"""
XGBoost View Module (P, Q, Omega) for 9 FF3 Selected Assets
Generates monthly Black-Litterman views using XGBoost forecasts.

Outputs:
    data/bl/bl_input/ (P.csv, Q.csv, Omega.csv, assets.csv, constraints.json)
    data/bl/diagnostics/ (views_timeseries.csv, model_fits.csv, predicted_returns.csv)
    data/bl/views_readable.csv
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(HERE, ".."))
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
FF3_DATA_DIR = os.path.join(ROOT_DIR, "data", "ff3")
BL_DATA_DIR = os.path.join(ROOT_DIR, "data", "bl")
BL_INPUT_DIR = os.path.join(BL_DATA_DIR, "bl_input")
BL_DIAG_DIR = os.path.join(BL_DATA_DIR, "diagnostics")

PATH_FF3_SELECTED = os.path.join(FF3_DATA_DIR, "ff3_selected.csv")
PATH_RETURNS_WIDE = os.path.join(RAW_DATA_DIR, "monthly_returns_10y_wide.csv")
PATH_PRICES_CACHE = os.path.join(RAW_DATA_DIR, ".cache", "prices_cache.parquet")
PATH_VOLUME_CACHE = os.path.join(RAW_DATA_DIR, ".cache", "volume_cache.parquet")
PATH_FACTORS = os.path.join(FF3_DATA_DIR, "F-F_Research_Data_Factors.csv")

FEATURES = ["mom1m", "mom6m", "mom12m", "chmom", "retvol", "dolvol", "beta", "idiovol"]
GBRT_DEPTHS = (1, 2)
GBRT_LRS = (0.01, 0.1)
GBRT_MAX_TREES = 1000
GBRT_EARLY_STOP = 50
VAL_MONTHS = 12
REFIT_EVERY = 12
SEED = 42

N_LONG_SHORT = 2
VIEW_NAME = f"Top{N_LONG_SHORT}_minus_Bottom{N_LONG_SHORT}"
OMEGA_WINDOW = 24
OMEGA_MIN_OBS = 12
OMEGA_FLOOR = 1e-6


@dataclass(frozen=True)
class ViewResult:
    date: str
    assets: Tuple[str, ...]
    P: np.ndarray
    Q: np.ndarray
    Omega: np.ndarray
    view_names: Tuple[str, ...]
    predicted_returns: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_selection() -> pd.DataFrame:
    """Load the 9 stocks selected by the FF3 model."""
    sel = pd.read_csv(PATH_FF3_SELECTED)
    return sel.sort_values("rank").reset_index(drop=True)


def load_factors() -> pd.DataFrame:
    with open(PATH_FACTORS, "r", encoding="utf-8", errors="latin-1") as f:
        lines = f.read().splitlines()
    rows = []
    for ln in lines:
        p = [x.strip() for x in ln.split(",")]
        if len(p) == 5 and p[0].isdigit() and len(p[0]) == 6:
            rows.append(p)
    ff = pd.DataFrame(rows, columns=["ym", "Mkt_RF", "SMB", "HML", "RF"])
    ff["ym"] = pd.PeriodIndex(ff["ym"].str[:4] + "-" + ff["ym"].str[4:], freq="M")
    return ff.set_index("ym").astype(float) / 100.0


def _cum(ret: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    g = 1.0 + ret
    out = g.shift(last)
    for j in range(last + 1, first + 1):
        out = out * g.shift(j)
    return out - 1.0


def build_panel(ret, price, volume, factors) -> pd.DataFrame:
    f = factors.reindex(ret.index)
    mkt, ex = f["Mkt_RF"], ret.sub(f["RF"], axis=0)
    roll = lambda x: x.rolling(12, min_periods=12).mean()
    var_m = roll(mkt ** 2) - roll(mkt) ** 2
    beta = (roll(ex.mul(mkt, axis=0)) - roll(ex).mul(roll(mkt), axis=0)).div(var_m, axis=0)
    var_ex = roll(ex ** 2) - roll(ex) ** 2
    raw = {
        "mom1m": ret,
        "mom6m": _cum(ret, 5, 1),
        "mom12m": _cum(ret, 11, 1),
        "chmom": _cum(ret, 5, 1) - _cum(ret, 11, 6),
        "retvol": ret.rolling(12, min_periods=12).std(),
        "dolvol": np.log((price * volume).where(volume > 0)),
        "beta": beta,
        "idiovol": np.sqrt((var_ex - beta.pow(2).mul(var_m, axis=0)).clip(lower=0)),
    }
    cols = {}
    for name, wide in raw.items():
        ranked = (2.0 * wide.round(10).rank(axis=1, pct=True) - 1.0).where(wide.notna())
        cols[name] = ranked.stack(future_stack=True)
    panel = pd.DataFrame(cols)
    panel["has_history"] = raw["mom12m"].notna().stack(future_stack=True)
    panel["target"] = ret.shift(-1).stack(future_stack=True)
    panel.index.names = ["ym", "Ticker"]
    panel[FEATURES] = panel[FEATURES].fillna(0.0)
    return panel


class _Booster:
    def __init__(self, booster: xgb.Booster):
        self.booster = booster

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(xgb.DMatrix(X[FEATURES].to_numpy(dtype=np.float32))).astype(float)


def split_validation(train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    months = train.index.get_level_values("ym")
    val_start = months.max() - VAL_MONTHS + 1
    return train[months < val_start], train[months >= val_start]


def fit_xgb(train: pd.DataFrame) -> Tuple[_Booster, Dict[str, Any]]:
    fit_part, val_part = split_validation(train)
    a = lambda df: df[FEATURES].to_numpy(dtype=np.float32)
    best = None
    for depth in GBRT_DEPTHS:
        for lr in GBRT_LRS:
            params = {
                "objective": "reg:squarederror", "max_depth": depth, "eta": lr,
                "seed": SEED, "tree_method": "hist", "nthread": 0
            }
            dtr = xgb.DMatrix(a(fit_part), label=fit_part["target"].to_numpy())
            dval = xgb.DMatrix(a(val_part), label=val_part["target"].to_numpy())
            bst = xgb.train(params, dtr, num_boost_round=GBRT_MAX_TREES, evals=[(dval, "val")],
                            verbose_eval=False, early_stopping_rounds=GBRT_EARLY_STOP)
            n = bst.best_iteration + 1
            mse = float(np.mean((val_part["target"].to_numpy() - bst.predict(dval, iteration_range=(0, n))) ** 2))
            if best is None or mse < best[3]:
                best = (depth, lr, n, mse, params)
    depth, lr, n, mse, params = best
    d_full = xgb.DMatrix(a(train), label=train["target"].to_numpy())
    model = _Booster(xgb.train(params, d_full, num_boost_round=n))
    return model, {"max_depth": depth, "learning_rate": lr, "n_trees": n, "val_mse": mse}


def make_P(mu: np.ndarray, assets: List[str]) -> np.ndarray:
    ok = np.flatnonzero(~np.isnan(mu))
    order = ok[np.argsort(-mu[ok])]
    row = np.zeros(len(mu))
    row[order[:N_LONG_SHORT]] = 1.0 / N_LONG_SHORT
    row[order[-N_LONG_SHORT:]] = -1.0 / N_LONG_SHORT
    return row


def generate_views(
    eval_start: str = "2021-01",
    tau: float = 0.1,
    use_max_weight: bool = True,
    max_weight: float = 0.20
) -> Dict[str, ViewResult]:
    """Generate walk-forward BL views and export to data/bl/."""
    for d in [BL_DATA_DIR, BL_INPUT_DIR, BL_DIAG_DIR]:
        os.makedirs(d, exist_ok=True)

    sel = load_selection()
    assets = sel["Ticker"].tolist()

    ret_wide = pd.read_csv(PATH_RETURNS_WIDE, index_col=0, parse_dates=[0])
    ret_wide.index = ret_wide.index.to_period("M")

    price_daily = pd.read_parquet(PATH_PRICES_CACHE)
    price_wide = price_daily.resample("ME").last()
    price_wide.index = price_wide.index.to_period("M")

    volume_wide = pd.read_parquet(PATH_VOLUME_CACHE)
    volume_wide.index = volume_wide.index.to_period("M")

    factors = load_factors()

    common = ret_wide.index.intersection(factors.index).intersection(price_wide.index).intersection(volume_wide.index)
    ret_wide = ret_wide.loc[common]
    price_wide = price_wide.loc[common]
    volume_wide = volume_wide.loc[common]
    factors = factors.loc[common]

    train_universe = [t for t in ret_wide.columns if t in price_wide.columns and t in volume_wide.columns]
    panel = build_panel(ret_wide[train_universe], price_wide[train_universe], volume_wide[train_universe], factors)

    eval_start_p = pd.Period(eval_start, "M")
    history_start_p = max(eval_start_p - 24, panel.index.get_level_values("ym").min() + 12)
    decisions = pd.period_range(history_start_p, common.max(), freq="M")

    ym = panel.index.get_level_values("ym")
    model, rows, results, fits = None, [], {}, []

    for t in decisions:
        if model is None or (t - history_start_p).n % REFIT_EVERY == 0 or t == eval_start_p:
            train = panel[(ym <= t - 1) & panel["has_history"] & panel["target"].notna()]
            model, info = fit_xgb(train)
            fits.append({"fit_date": str(t), "n_obs": len(train), **info})

        X = panel.xs(t, level="ym").reindex(assets)
        mu = model.predict(X[FEATURES].fillna(0.0))
        P = make_P(mu, assets)
        Q = float(P @ np.nan_to_num(mu))

        next_month = t + 1
        realized = ret_wide.loc[next_month, assets].to_numpy() if next_month in ret_wide.index else np.full(len(assets), np.nan)
        Q_real = float(np.nansum(P * realized)) if next_month in ret_wide.index else np.nan
        error = Q_real - Q if not np.isnan(Q_real) else np.nan

        past = [r["error"] for r in rows if not np.isnan(r["error"])][-OMEGA_WINDOW:]
        if len(past) >= OMEGA_MIN_OBS:
            omega = max(float(np.mean(np.square(past))), OMEGA_FLOOR)
        else:
            hist_cov = ret_wide.loc[:t, assets].dropna().tail(36).cov().to_numpy()
            omega = max(float((P @ (tau * hist_cov) @ P.T)), OMEGA_FLOOR)

        phase = "eval" if t >= eval_start_p else "history"
        long_str = " ".join([assets[i] for i in np.where(P > 0)[0]])
        short_str = " ".join([assets[i] for i in np.where(P < 0)[0]])

        rows.append({
            "date": str(t), "hold_until": str(next_month), "phase": phase,
            "Q": Q, "Q_real": Q_real, "error": error, "Omega": omega,
            "long": long_str, "short": short_str
        })

        if phase == "eval":
            results[str(t)] = ViewResult(
                str(t), tuple(assets), P[None, :], np.array([Q]), np.array([[omega]]),
                (VIEW_NAME,), mu, {"hold_until": str(next_month), "use_max_weight": use_max_weight, "max_weight": max_weight}
            )

    # Save to data/bl
    constraints = {"use_max_weight": use_max_weight, "max_weight": max_weight if use_max_weight else None, "tau": tau}
    export_views(sel, results, pd.DataFrame(rows), pd.DataFrame(fits), constraints)
    return results


def export_views(sel, results, rec, fits, constraints):
    a = sel[["rank", "Ticker", "PERMNO", "name", "marketCap"]].copy()
    a.to_csv(os.path.join(BL_INPUT_DIR, "assets.csv"), index=False, encoding="utf-8-sig")

    ev = list(results.values())
    base = [{"date": r.date, "hold_until": r.metadata["hold_until"]} for r in ev]
    pd.DataFrame([{**b, "view": VIEW_NAME, **dict(zip(r.assets, r.P[0]))} for b, r in zip(base, ev)]) \
        .to_csv(os.path.join(BL_INPUT_DIR, "P.csv"), index=False)
    pd.DataFrame([{**b, VIEW_NAME: r.Q[0]} for b, r in zip(base, ev)]) \
        .to_csv(os.path.join(BL_INPUT_DIR, "Q.csv"), index=False)
    pd.DataFrame([{**b, f"{VIEW_NAME}|{VIEW_NAME}": r.Omega[0, 0]} for b, r in zip(base, ev)]) \
        .to_csv(os.path.join(BL_INPUT_DIR, "Omega.csv"), index=False)
    with open(os.path.join(BL_INPUT_DIR, "constraints.json"), "w", encoding="utf-8") as f:
        json.dump(constraints, f, ensure_ascii=False, indent=2)

    evr = rec[rec["phase"] == "eval"]
    pd.DataFrame({
        "날짜(결정월)": evr["date"], "보유월": evr["hold_until"], "견해": VIEW_NAME,
        "롱 종목": evr["long"], "숏 종목": evr["short"],
        "예측 스프레드 Q(%/월)": (evr["Q"] * 100).round(3),
        "실현 스프레드(%/월)": (evr["Q_real"] * 100).round(3),
        "Omega": evr["Omega"],
    }).to_csv(os.path.join(BL_DATA_DIR, "views_readable.csv"), index=False, encoding="utf-8-sig")

    rec.to_csv(os.path.join(BL_DIAG_DIR, "views_timeseries.csv"), index=False)
    fits.to_csv(os.path.join(BL_DIAG_DIR, "model_fits.csv"), index=False)
    pd.DataFrame({r.date: r.predicted_returns for r in ev}, index=list(sel["Ticker"])).T \
        .rename_axis("date").to_csv(os.path.join(BL_DIAG_DIR, "predicted_returns.csv"))


def load_views() -> Dict[str, ViewResult]:
    """Load views from data/bl/bl_input/."""
    P = pd.read_csv(os.path.join(BL_INPUT_DIR, "P.csv"))
    Q = pd.read_csv(os.path.join(BL_INPUT_DIR, "Q.csv")).set_index("date")
    Om = pd.read_csv(os.path.join(BL_INPUT_DIR, "Omega.csv")).set_index("date")
    with open(os.path.join(BL_INPUT_DIR, "constraints.json"), encoding="utf-8") as f:
        cons = json.load(f)
    assets = tuple(c for c in P.columns[3:])
    return {
        row["date"]: ViewResult(
            row["date"], assets, row[P.columns[3:]].to_numpy(dtype=float)[None, :],
            Q.loc[[row["date"]], [VIEW_NAME]].to_numpy(dtype=float)[0],
            Om.loc[[row["date"]]].iloc[:, 1:].to_numpy(dtype=float).reshape(1, 1),
            (VIEW_NAME,), np.array([]), {"hold_until": row["hold_until"], **cons}
        )
        for _, row in P.iterrows()
    }


if __name__ == "__main__":
    v = generate_views()
    print(f"[Done] Generated views for {len(v)} evaluation months in data/bl/")
