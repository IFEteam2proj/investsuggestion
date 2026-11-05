"""
Black-Litterman (BL) Portfolio Optimization with XGBoost View Generation
Universe: 9 Stocks Selected by Fama-French 3-Factor (FF3) Model
    NVDA, LLY, PWR, PGR, AVGO, ANET, KLAC, LRCX, FTNT

Workflow:
1. Load 9 FF3 selected stocks and market data (returns, prices, volumes, factors).
2. Engineer 8 price/volume features cross-sectionally.
3. Train XGBoost model with time-series validation and walk-forward refitting.
4. Generate Black-Litterman Views (P, Q, Omega) for each decision month:
   - P: Top-k Long (+1/k) / Bottom-k Short (-1/k)
   - Q: Forecasted view spread (Q = P * mu)
   - Omega: Rolling 24-month forecast error variance
5. Compute Equilibrium Prior:
   - Pi = delta * Sigma * w_mkt
6. Compute BL Posterior Distribution:
   - E[R] = Pi + tau * Sigma * P^T * (P * tau * Sigma * P^T + Omega)^(-1) * (Q - P * Pi)
   - Sigma_BL = Sigma + M
7. Estimate Asset Weights:
   - Unconstrained BL weights
   - Constrained Long-Only BL weights (sum w = 1, 0 <= w <= max_weight)
8. Save intermediate data to `data/bl/` and final results to `results/`.
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import xgboost as xgb

# Project Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
FF3_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'ff3')
BL_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'bl')
BL_INPUT_DIR = os.path.join(BL_DATA_DIR, 'bl_input')
BL_DIAG_DIR = os.path.join(BL_DATA_DIR, 'diagnostics')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

PATH_FF3_SELECTED = os.path.join(FF3_DATA_DIR, 'ff3_selected.csv')
PATH_UNIVERSE = os.path.join(RAW_DATA_DIR, 'stock_universe_top80.csv')
PATH_RETURNS_WIDE = os.path.join(RAW_DATA_DIR, 'monthly_returns_10y_wide.csv')
PATH_PRICES_CACHE = os.path.join(RAW_DATA_DIR, '.cache', 'prices_cache.parquet')
PATH_VOLUME_CACHE = os.path.join(RAW_DATA_DIR, '.cache', 'volume_cache.parquet')
PATH_FACTORS = os.path.join(FF3_DATA_DIR, 'F-F_Research_Data_Factors.csv')

# XGBoost Hyperparameters
FEATURES = ["mom1m", "mom6m", "mom12m", "chmom", "retvol", "dolvol", "beta", "idiovol"]
GBRT_DEPTHS = (1, 2)
GBRT_LRS = (0.01, 0.1)
GBRT_MAX_TREES = 1000
GBRT_EARLY_STOP = 50
VAL_MONTHS = 12
SEED = 42

# Black-Litterman Defaults
DEFAULT_TAU = 0.1
DEFAULT_DELTA = 3.0
DEFAULT_MAX_WEIGHT = 0.20
DEFAULT_N_LONG_SHORT = 2
DEFAULT_REFIT_EVERY = 12
DEFAULT_EVAL_START = "2021-01"
OMEGA_WINDOW = 24
OMEGA_MIN_OBS = 12
OMEGA_FLOOR = 1e-6


def ensure_directories():
    for d in [RAW_DATA_DIR, FF3_DATA_DIR, BL_DATA_DIR, BL_INPUT_DIR, BL_DIAG_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)


# =============================================================================
# 1. Data Loading & Feature Engineering
# =============================================================================
def load_selected_assets() -> pd.DataFrame:
    """Load the 9 stocks selected by the FF3 model."""
    if not os.path.exists(PATH_FF3_SELECTED):
        raise FileNotFoundError(f"Missing FF3 selection file: {PATH_FF3_SELECTED}. Run run_ff3.py first.")
    sel = pd.read_csv(PATH_FF3_SELECTED)
    return sel.sort_values('rank').reset_index(drop=True)


def load_factors() -> pd.DataFrame:
    """Parse Kenneth French factors."""
    with open(PATH_FACTORS, 'r', encoding='utf-8', errors='latin-1') as f:
        lines = f.read().splitlines()
    rows = []
    for ln in lines:
        p = [x.strip() for x in ln.split(',')]
        if len(p) == 5 and p[0].isdigit() and len(p[0]) == 6:
            rows.append(p)
    ff = pd.DataFrame(rows, columns=['ym', 'Mkt_RF', 'SMB', 'HML', 'RF'])
    ff['ym'] = pd.PeriodIndex(ff['ym'].str[:4] + '-' + ff['ym'].str[4:], freq='M')
    return ff.set_index('ym').astype(float) / 100.0


def _cum(ret: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    """t-first ~ t-last cumulative return."""
    g = 1.0 + ret
    out = g.shift(last)
    for j in range(last + 1, first + 1):
        out = out * g.shift(j)
    return out - 1.0


def build_panel(ret: pd.DataFrame, price: pd.DataFrame, volume: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Construct cross-sectionally ranked feature panel."""
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


# =============================================================================
# 2. XGBoost Model Training
# =============================================================================
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
                "objective": "reg:squarederror",
                "max_depth": depth,
                "eta": lr,
                "seed": SEED,
                "tree_method": "hist",
                "nthread": 0
            }
            dtr = xgb.DMatrix(a(fit_part), label=fit_part["target"].to_numpy())
            dval = xgb.DMatrix(a(val_part), label=val_part["target"].to_numpy())
            bst = xgb.train(
                params, dtr,
                num_boost_round=GBRT_MAX_TREES,
                evals=[(dval, "val")],
                verbose_eval=False,
                early_stopping_rounds=GBRT_EARLY_STOP
            )
            n = bst.best_iteration + 1
            mse = float(np.mean((val_part["target"].to_numpy() - bst.predict(dval, iteration_range=(0, n))) ** 2))
            if best is None or mse < best[3]:
                best = (depth, lr, n, mse, params)

    depth, lr, n, mse, params = best
    d_full = xgb.DMatrix(a(train), label=train["target"].to_numpy())
    model = _Booster(xgb.train(params, d_full, num_boost_round=n))
    return model, {"max_depth": depth, "learning_rate": lr, "n_trees": n, "val_mse": mse}


def make_P(mu: np.ndarray, assets: List[str], n_long_short: int) -> np.ndarray:
    """Relative View Vector: Top k Long (+1/k), Bottom k Short (-1/k)."""
    ok = np.flatnonzero(~np.isnan(mu))
    order = ok[np.argsort(-mu[ok])]
    row = np.zeros(len(mu))
    row[order[:n_long_short]] = 1.0 / n_long_short
    row[order[-n_long_short:]] = -1.0 / n_long_short
    return row


# =============================================================================
# 3. Black-Litterman Master Formula & Optimization
# =============================================================================
def black_litterman_estimates(
    Pi: np.ndarray,
    Sigma: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Black-Litterman posterior expected returns and covariance matrix.
    E[R] = Pi + tau * Sigma * P^T * (P * tau * Sigma * P^T + Omega)^(-1) * (Q - P * Pi)
    """
    tau_Sigma = tau * Sigma
    P_tau_Sigma = P @ tau_Sigma
    view_cov = P_tau_Sigma @ P.T + Omega
    view_cov_inv = np.linalg.inv(view_cov)

    # Combined posterior returns
    excess_view = Q - (P @ Pi)
    er_bl = Pi + (tau_Sigma @ P.T @ view_cov_inv @ excess_view).flatten()

    # Posterior covariance: Sigma_BL = Sigma + M
    # M = tau * Sigma - tau * Sigma * P^T * (P * tau * Sigma * P^T + Omega)^(-1) * P * tau * Sigma
    M = tau_Sigma - tau_Sigma @ P.T @ view_cov_inv @ P_tau_Sigma
    Sigma_BL = Sigma + M

    return er_bl, Sigma_BL


def optimize_portfolio_weights(
    er: np.ndarray,
    cov: np.ndarray,
    delta: float,
    max_weight: float | None = 0.20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate optimal portfolio weights:
    1. Unconstrained BL weights: w = (1/delta) * cov^(-1) * er (normalized)
    2. Constrained Long-Only weights: max w^T er - (delta/2) w^T cov w
       s.t. sum w = 1, 0 <= w_i <= max_weight
    """
    n_assets = len(er)

    # 1. Unconstrained analytical weights
    try:
        cov_inv = np.linalg.inv(cov)
        w_uncon = (cov_inv @ er) / delta
        if w_uncon.sum() != 0:
            w_uncon_norm = w_uncon / w_uncon.sum()
        else:
            w_uncon_norm = np.ones(n_assets) / n_assets
    except np.linalg.LinAlgError:
        w_uncon_norm = np.ones(n_assets) / n_assets

    # 2. Constrained optimization
    def obj(w):
        return -(w @ er - 0.5 * delta * (w @ cov @ w))

    upper_bound = max_weight if max_weight is not None else 1.0
    # Minimum feasible bound for sum(w) = 1 is 1 / n_assets
    upper_bound = max(upper_bound, 1.0 / n_assets + 1e-4)

    bounds = [(0.0, upper_bound) for _ in range(n_assets)]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    init_w = np.ones(n_assets) / n_assets

    res = minimize(obj, x0=init_w, method='SLSQP', bounds=bounds, constraints=constraints)
    if res.success:
        w_const = res.x
        w_const = np.clip(w_const, 0.0, upper_bound)
        w_const /= w_const.sum()
    else:
        # Fallback to unconstrained clipped
        w_const = np.clip(w_uncon_norm, 0.0, upper_bound)
        w_const /= w_const.sum()

    return w_uncon_norm, w_const


# =============================================================================
# 4. Walk-Forward Black-Litterman Pipeline
# =============================================================================
def run_bl_pipeline(
    tau: float = DEFAULT_TAU,
    delta: float = DEFAULT_DELTA,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    n_long_short: int = DEFAULT_N_LONG_SHORT,
    refit_every: int = DEFAULT_REFIT_EVERY,
    eval_start: str = DEFAULT_EVAL_START
) -> None:
    ensure_directories()
    t_start = time.time()

    print("=" * 75)
    print(" Black-Litterman Portfolio Optimization with XGBoost Views")
    print(f" Universe: 9 FF3 Selected Assets | Tau: {tau} | Risk Aversion (Delta): {delta}")
    print(f" Long/Short per View: Top {n_long_short} / Bottom {n_long_short} | Max Weight Cap: {max_weight*100:.0f}%")
    print("=" * 75)

    # 1. Load 9 selected assets
    sel_df = load_selected_assets()
    assets = sel_df['Ticker'].tolist()
    permnos = sel_df['PERMNO'].tolist()
    asset_names = dict(zip(assets, sel_df['name']))
    print(f"[Universe] Selected 9 stocks: {', '.join(assets)}")

    # 2. Load returns, prices, volumes, factors
    ret_wide = pd.read_csv(PATH_RETURNS_WIDE, index_col=0, parse_dates=[0])
    ret_wide.index = ret_wide.index.to_period('M')

    price_daily = pd.read_parquet(PATH_PRICES_CACHE)
    price_wide = price_daily.resample('ME').last()
    price_wide.index = price_wide.index.to_period('M')

    volume_wide = pd.read_parquet(PATH_VOLUME_CACHE)
    volume_wide.index = volume_wide.index.to_period('M')

    factors = load_factors()

    # Align periods
    common_months = ret_wide.index.intersection(factors.index).intersection(price_wide.index).intersection(volume_wide.index)
    ret_wide = ret_wide.loc[common_months]
    price_wide = price_wide.loc[common_months]
    volume_wide = volume_wide.loc[common_months]
    factors = factors.loc[common_months]

    # Universe for ML training (broad cross-section for robust feature learning)
    training_universe = [t for t in ret_wide.columns if t in price_wide.columns and t in volume_wide.columns]
    print(f"[Data] Aligned {len(common_months)} months ({common_months.min()} ~ {common_months.max()}).")
    print(f"[ML] Training features cross-section: {len(training_universe)} stocks.")

    # 3. Build Panel
    panel = build_panel(ret_wide[training_universe], price_wide[training_universe], volume_wide[training_universe], factors)

    # Benchmark market-cap weights for the 9 assets
    universe_df = pd.read_csv(PATH_UNIVERSE).set_index('symbol')
    mc_series = universe_df.loc[assets, 'marketCap']
    w_mkt = (mc_series / mc_series.sum()).to_numpy()

    # 4. Walk-forward Schedule
    eval_start_p = pd.Period(eval_start, 'M')
    # Start error tracking 24 months before eval_start so Omega has initial observations
    history_start_p = max(eval_start_p - 24, panel.index.get_level_values('ym').min() + 12)
    decisions = pd.period_range(history_start_p, common_months.max(), freq='M')

    print(f"[Schedule] View History Start: {history_start_p} | Evaluation Start: {eval_start_p} | End: {common_months.max()}")

    view_name = f"Top{n_long_short}_minus_Bottom{n_long_short}"
    ym = panel.index.get_level_values("ym")

    model = None
    view_history = []
    fits_log = []
    bl_results_list = []
    backtest_records = []

    for t in decisions:
        # Refit model every `refit_every` months or on the first run
        if model is None or (t - history_start_p).n % refit_every == 0 or t == eval_start_p:
            train_sub = panel[(ym <= t - 1) & panel["has_history"] & panel["target"].notna()]
            model, fit_info = fit_xgb(train_sub)
            fits_log.append({"decision_month": str(t), "n_train_obs": len(train_sub), **fit_info})

        # Feature slice for the 9 assets at decision month t
        X_t = panel.xs(t, level="ym").reindex(assets)
        mu_t = model.predict(X_t[FEATURES].fillna(0.0))  # (9,)

        # View P and Q
        P_t = make_P(mu_t, assets, n_long_short).reshape(1, -1)  # (1, 9)
        Q_t = np.array([float((P_t @ mu_t)[0])])

        # Realized next-month return of view (if available)
        next_month = t + 1
        if next_month in ret_wide.index:
            realized_rets = ret_wide.loc[next_month, assets].to_numpy()
            Q_real = float(np.nansum(P_t[0] * realized_rets))
            view_error = Q_real - Q_t[0]
        else:
            realized_rets = np.full(len(assets), np.nan)
            Q_real = np.nan
            view_error = np.nan

        # Omega calculation: rolling MSE of past errors
        past_errors = [v["error"] for v in view_history if not np.isnan(v["error"])][-OMEGA_WINDOW:]

        # Covariance matrix: rolling 36 months of asset returns up to t
        hist_window = ret_wide.loc[:t, assets].dropna().tail(36)
        if len(hist_window) >= 12:
            Sigma_t = hist_window.cov().to_numpy()
        else:
            Sigma_t = ret_wide.loc[:t, assets].cov().to_numpy()

        if len(past_errors) >= OMEGA_MIN_OBS:
            omega_val = max(float(np.mean(np.square(past_errors))), OMEGA_FLOOR)
        else:
            # He-Litterman proportional prior when insufficient error samples
            omega_val = float((P_t @ (tau * Sigma_t) @ P_t.T)[0, 0])
            omega_val = max(omega_val, OMEGA_FLOOR)

        Omega_t = np.array([[omega_val]])

        phase = "eval" if t >= eval_start_p else "history"

        # Long and Short asset symbols
        long_symbols = [assets[i] for i in np.where(P_t[0] > 0)[0]]
        short_symbols = [assets[i] for i in np.where(P_t[0] < 0)[0]]

        view_entry = {
            "date": str(t),
            "hold_month": str(next_month),
            "phase": phase,
            "Q": Q_t[0],
            "Q_real": Q_real,
            "error": view_error,
            "Omega": omega_val,
            "long_assets": " ".join(long_symbols),
            "short_assets": " ".join(short_symbols)
        }
        view_history.append(view_entry)

        # In evaluation phase, solve Black-Litterman weights and record backtest
        if phase == "eval":
            # 1. Equilibrium Prior Pi = delta * Sigma * w_mkt
            Pi_t = delta * (Sigma_t @ w_mkt)

            # 2. Black-Litterman posterior
            er_bl_t, Sigma_bl_t = black_litterman_estimates(Pi_t, Sigma_t, P_t, Q_t, Omega_t, tau)

            # 3. Portfolio Weights
            w_uncon, w_const = optimize_portfolio_weights(er_bl_t, Sigma_bl_t, delta, max_weight=max_weight)

            # Store result for this month
            bl_results_list.append({
                "date": str(t),
                "hold_month": str(next_month),
                "P": P_t[0],
                "Q": Q_t[0],
                "Omega": omega_val,
                "Pi": Pi_t,
                "mu_ml": mu_t,
                "er_bl": er_bl_t,
                "w_mkt": w_mkt,
                "w_uncon": w_uncon,
                "w_const": w_const,
                "w_eq": np.ones(len(assets)) / len(assets)
            })

            # Realized backtest performance in next_month
            if not np.isnan(realized_rets).any():
                ret_bl_const = float(np.sum(w_const * realized_rets))
                ret_bl_uncon = float(np.sum(w_uncon * realized_rets))
                ret_mkt = float(np.sum(w_mkt * realized_rets))
                ret_eq = float(np.mean(realized_rets))
                rf_rate = float(factors.loc[next_month, 'RF']) if next_month in factors.index else 0.0

                backtest_records.append({
                    "decision_month": str(t),
                    "hold_month": str(next_month),
                    "ret_bl_const": ret_bl_const,
                    "ret_bl_uncon": ret_bl_uncon,
                    "ret_mkt_benchmark": ret_mkt,
                    "ret_eq_benchmark": ret_eq,
                    "rf": rf_rate
                })

    # =========================================================================
    # 5. Export Intermediate Data (`data/bl/`)
    # =========================================================================
    print("\n[Export] Saving intermediate data to data/bl/...")
    # assets.csv
    assets_meta = sel_df[['rank', 'Ticker', 'PERMNO', 'name', 'marketCap']].copy()
    assets_meta['marketCap_weight'] = w_mkt.round(4)
    assets_meta.to_csv(os.path.join(BL_INPUT_DIR, 'assets.csv'), index=False, encoding='utf-8-sig')

    # constraints.json
    constraints_data = {
        "universe_assets": assets,
        "n_assets": len(assets),
        "tau": tau,
        "risk_aversion_delta": delta,
        "max_weight_constraint": max_weight,
        "n_long_short": n_long_short,
        "view_type": view_name,
        "eval_start": eval_start,
        "refit_every_months": refit_every
    }
    with open(os.path.join(BL_INPUT_DIR, 'constraints.json'), 'w', encoding='utf-8') as f:
        json.dump(constraints_data, f, ensure_ascii=False, indent=2)

    # P.csv, Q.csv, Omega.csv
    p_rows = []
    q_rows = []
    om_rows = []
    for r in bl_results_list:
        p_row = {"date": r["date"], "hold_month": r["hold_month"], "view": view_name}
        for a_sym, p_val in zip(assets, r["P"]):
            p_row[a_sym] = round(p_val, 4)
        p_rows.append(p_row)
        q_rows.append({"date": r["date"], "hold_month": r["hold_month"], view_name: round(r["Q"], 6)})
        om_rows.append({"date": r["date"], "hold_month": r["hold_month"], f"{view_name}|{view_name}": r["Omega"]})

    pd.DataFrame(p_rows).to_csv(os.path.join(BL_INPUT_DIR, 'P.csv'), index=False)
    pd.DataFrame(q_rows).to_csv(os.path.join(BL_INPUT_DIR, 'Q.csv'), index=False)
    pd.DataFrame(om_rows).to_csv(os.path.join(BL_INPUT_DIR, 'Omega.csv'), index=False)

    # Diagnostics
    df_views = pd.DataFrame(view_history)
    df_views.to_csv(os.path.join(BL_DIAG_DIR, 'views_timeseries.csv'), index=False)
    pd.DataFrame(fits_log).to_csv(os.path.join(BL_DIAG_DIR, 'model_fits.csv'), index=False)

    # views_readable.csv
    eval_views = df_views[df_views["phase"] == "eval"]
    readable_df = pd.DataFrame({
        "결정월": eval_views["date"],
        "보유월": eval_views["hold_month"],
        "견해명": view_name,
        "롱 종목 (Long)": eval_views["long_assets"],
        "숏 종목 (Short)": eval_views["short_assets"],
        "예측 스프레드 Q (%/월)": (eval_views["Q"] * 100).round(3),
        "실현 스프레드 (%/월)": (eval_views["Q_real"] * 100).round(3),
        "예측 오차 (%/월)": (eval_views["error"] * 100).round(3),
        "불확실성 Omega": eval_views["Omega"]
    })
    readable_df.to_csv(os.path.join(BL_DATA_DIR, 'views_readable.csv'), index=False, encoding='utf-8-sig')

    # Diagnostics summary JSON
    valid_eval_views = eval_views.dropna(subset=['Q_real'])
    hit_rate = float((np.sign(valid_eval_views['Q']) == np.sign(valid_eval_views['Q_real'])).mean()) if len(valid_eval_views) else 0.0
    diag_summary = {
        "n_eval_months": len(eval_views),
        "view_hit_rate": hit_rate,
        "mean_Q_pct_month": float(valid_eval_views['Q'].mean() * 100) if len(valid_eval_views) else 0.0,
        "mean_Q_real_pct_month": float(valid_eval_views['Q_real'].mean() * 100) if len(valid_eval_views) else 0.0,
        "mean_absolute_error_pct": float(valid_eval_views['error'].abs().mean() * 100) if len(valid_eval_views) else 0.0,
        "parameters": constraints_data
    }
    with open(os.path.join(BL_DIAG_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(diag_summary, f, ensure_ascii=False, indent=2)

    # =========================================================================
    # 6. Export Results (`results/`)
    # =========================================================================
    print("[Export] Saving final results to results/...")

    # A. Latest Asset Weights (`results/bl_weights_latest.csv`)
    latest_res = bl_results_list[-1]
    df_latest = pd.DataFrame({
        "Ticker": assets,
        "Name": [asset_names[a] for a in assets],
        "MarketCap_Weight": latest_res["w_mkt"].round(4),
        "Equal_Weight": latest_res["w_eq"].round(4),
        "Prior_Return_Ann": (latest_res["Pi"] * 12).round(4),
        "XGBoost_Return_Ann": (latest_res["mu_ml"] * 12).round(4),
        "BL_Expected_Return_Ann": (latest_res["er_bl"] * 12).round(4),
        "BL_Unconstrained_Weight": latest_res["w_uncon"].round(4),
        "BL_Constrained_Weight": latest_res["w_const"].round(4),
        "Tilt_vs_MarketCap": (latest_res["w_const"] - latest_res["w_mkt"]).round(4)
    })
    df_latest.to_csv(os.path.join(RESULTS_DIR, 'bl_weights_latest.csv'), index=False, encoding='utf-8-sig')

    # B. Historical Weights (`results/bl_weights_history.csv`)
    hist_weight_rows = []
    for r in bl_results_list:
        row = {"date": r["date"], "hold_month": r["hold_month"]}
        for a_sym, w_val in zip(assets, r["w_const"]):
            row[f"{a_sym}_weight"] = round(w_val, 4)
        hist_weight_rows.append(row)
    df_hist_weights = pd.DataFrame(hist_weight_rows)
    df_hist_weights.to_csv(os.path.join(RESULTS_DIR, 'bl_weights_history.csv'), index=False, encoding='utf-8-sig')

    # C. Backtest Performance Table (`results/bl_backtest_performance.csv`)
    df_backtest = pd.DataFrame(backtest_records)
    if not df_backtest.empty:
        # Calculate cumulative returns
        df_backtest["cum_bl_const"] = (1.0 + df_backtest["ret_bl_const"]).cumprod() - 1.0
        df_backtest["cum_bl_uncon"] = (1.0 + df_backtest["ret_bl_uncon"]).cumprod() - 1.0
        df_backtest["cum_mkt_benchmark"] = (1.0 + df_backtest["ret_mkt_benchmark"]).cumprod() - 1.0
        df_backtest["cum_eq_benchmark"] = (1.0 + df_backtest["ret_eq_benchmark"]).cumprod() - 1.0
        df_backtest.round(4).to_csv(os.path.join(RESULTS_DIR, 'bl_backtest_performance.csv'), index=False)

    # D. Portfolio Summary Metrics (`results/bl_portfolio_summary.csv`)
    def calc_metrics(r_series, rf_series):
        ex = r_series - rf_series
        ann_ret = float(r_series.mean() * 12)
        ann_vol = float(r_series.std() * np.sqrt(12))
        ann_rf = float(rf_series.mean() * 12)
        sharpe = (ann_ret - ann_rf) / ann_vol if ann_vol > 0 else np.nan
        cum_series = (1.0 + r_series).cumprod()
        peak = cum_series.cummax()
        mdd = float(((cum_series - peak) / peak).min())
        return {
            "Annualized_Return": round(ann_ret, 4),
            "Annualized_Volatility": round(ann_vol, 4),
            "Sharpe_Ratio": round(sharpe, 3),
            "Max_Drawdown": round(mdd, 4),
            "Total_Cumulative_Return": round(float(cum_series.iloc[-1] - 1.0), 4)
        }

    summary_records = []
    if not df_backtest.empty:
        rf_col = df_backtest["rf"]
        summary_records.append({
            "Portfolio": "Black-Litterman Constrained (Cap 20%)",
            "N_Assets": len(assets),
            "Evaluation_Months": len(df_backtest),
            **calc_metrics(df_backtest["ret_bl_const"], rf_col)
        })
        summary_records.append({
            "Portfolio": "Black-Litterman Unconstrained",
            "N_Assets": len(assets),
            "Evaluation_Months": len(df_backtest),
            **calc_metrics(df_backtest["ret_bl_uncon"], rf_col)
        })
        summary_records.append({
            "Portfolio": "Market-Cap Weighted Benchmark",
            "N_Assets": len(assets),
            "Evaluation_Months": len(df_backtest),
            **calc_metrics(df_backtest["ret_mkt_benchmark"], rf_col)
        })
        summary_records.append({
            "Portfolio": "Equal-Weighted Benchmark (1/9)",
            "N_Assets": len(assets),
            "Evaluation_Months": len(df_backtest),
            **calc_metrics(df_backtest["ret_eq_benchmark"], rf_col)
        })

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 'bl_portfolio_summary.csv'), index=False, encoding='utf-8-sig')

    # =========================================================================
    # 7. Console Summary Presentation
    # =========================================================================
    print("\n" + "=" * 75)
    print(f" [Latest Decision Month: {latest_res['date']} -> Hold Month: {latest_res['hold_month']}]")
    print(f" View: Long {df_views.iloc[-1]['long_assets']} vs Short {df_views.iloc[-1]['short_assets']}")
    print(f" Forecasted Spread Q: {latest_res['Q']*100:.2f}%/month | Uncertainty Omega: {latest_res['Omega']:.6f}")
    print("=" * 75)
    print(" Black-Litterman Asset Allocation Weights (9 FF3 Stocks):")
    print("=" * 75)
    show_cols = ["Ticker", "Name", "MarketCap_Weight", "BL_Constrained_Weight", "Tilt_vs_MarketCap", "Prior_Return_Ann", "BL_Expected_Return_Ann"]
    print(df_latest[show_cols].to_string(index=False))

    if not df_summary.empty:
        print("\n" + "=" * 75)
        print(f" Out-of-Sample Performance Comparison ({eval_start} ~ {common_months.max()}):")
        print("=" * 75)
        print(df_summary[["Portfolio", "Annualized_Return", "Annualized_Volatility", "Sharpe_Ratio", "Max_Drawdown", "Total_Cumulative_Return"]].to_string(index=False))

    print("\n" + "=" * 75)
    print(" [Saved Files Summary]")
    print(f" 1. Final Latest Weights:       {os.path.join(RESULTS_DIR, 'bl_weights_latest.csv')}")
    print(f" 2. Historical Weight Series:   {os.path.join(RESULTS_DIR, 'bl_weights_history.csv')}")
    print(f" 3. Portfolio Summary:          {os.path.join(RESULTS_DIR, 'bl_portfolio_summary.csv')}")
    print(f" 4. Out-of-Sample Backtest:     {os.path.join(RESULTS_DIR, 'bl_backtest_performance.csv')}")
    print(f" 5. BL Intermediate Inputs:     {BL_INPUT_DIR}")
    print(f" 6. BL Diagnostic Views:        {BL_DIAG_DIR}")
    print("=" * 75)
    print(f"[Done] Black-Litterman optimization pipeline finished in {time.time()-t_start:.2f} seconds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Black-Litterman Portfolio Optimization with XGBoost Views for 9 FF3 stocks.")
    parser.add_argument('--tau', type=float, default=DEFAULT_TAU, help=f"BL tau parameter (default: {DEFAULT_TAU})")
    parser.add_argument('--delta', type=float, default=DEFAULT_DELTA, help=f"Risk aversion coefficient delta (default: {DEFAULT_DELTA})")
    parser.add_argument('--risk-aversion', type=float, default=None, help="Alias for --delta")
    parser.add_argument('--max-weight', type=float, default=DEFAULT_MAX_WEIGHT, help=f"Max single stock weight (default: {DEFAULT_MAX_WEIGHT})")
    parser.add_argument('--n-long-short', type=int, default=DEFAULT_N_LONG_SHORT, help=f"Top-k / Bottom-k assets per view (default: {DEFAULT_N_LONG_SHORT})")
    parser.add_argument('--refit-every', type=int, default=DEFAULT_REFIT_EVERY, help=f"Refit XGBoost model every N months (default: {DEFAULT_REFIT_EVERY})")
    parser.add_argument('--eval-start', type=str, default=DEFAULT_EVAL_START, help=f"Evaluation start month (default: {DEFAULT_EVAL_START})")

    args = parser.parse_args()
    delta_val = args.risk_aversion if args.risk_aversion is not None else args.delta

    run_bl_pipeline(
        tau=args.tau,
        delta=delta_val,
        max_weight=args.max_weight,
        n_long_short=args.n_long_short,
        refit_every=args.refit_every,
        eval_start=args.eval_start
    )
