"""
HTS Quantitative Trading Terminal: US Top 80% FF3-XGBoost-Black-Litterman Dashboard
Clean, high-legibility quantitative investment interface with live execution and multi-step forecasting.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb

# =============================================================================
# Path & Directory Configurations
# =============================================================================
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
FF3_DATA_DIR = os.path.join(DATA_DIR, 'ff3')
BL_DATA_DIR = os.path.join(DATA_DIR, 'bl')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

PATH_PRICES_CACHE = os.path.join(RAW_DATA_DIR, '.cache', 'prices_cache.parquet')
PATH_VOLUME_CACHE = os.path.join(RAW_DATA_DIR, '.cache', 'volume_cache.parquet')
PATH_UNIVERSE = os.path.join(RAW_DATA_DIR, 'stock_universe_top80.csv')
PATH_RETURNS_WIDE = os.path.join(RAW_DATA_DIR, 'monthly_returns_10y_wide.csv')
PATH_FF3_SELECTED = os.path.join(FF3_DATA_DIR, 'ff3_selected.csv')
PATH_FF3_ALL = os.path.join(FF3_DATA_DIR, 'ff3_regression_all.csv')
PATH_BL_LATEST = os.path.join(RESULTS_DIR, 'bl_weights_latest.csv')
PATH_BL_SUMMARY = os.path.join(RESULTS_DIR, 'bl_portfolio_summary.csv')
PATH_BL_HISTORY = os.path.join(RESULTS_DIR, 'bl_weights_history.csv')
PATH_BL_BACKTEST = os.path.join(RESULTS_DIR, 'bl_backtest_performance.csv')
PATH_VIEWS_READABLE = os.path.join(BL_DATA_DIR, 'views_readable.csv')
PATH_FACTORS = os.path.join(FF3_DATA_DIR, 'F-F_Research_Data_Factors.csv')

# =============================================================================
# Streamlit Page Settings & Clean CSS
# =============================================================================
st.set_page_config(
    page_title="QUANT HTS TERMINAL | BL-XGBOOST",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto"
)

# Responsive, High-Contrast Dashboard CSS (Auto-adapts between Desktop, Tablet & Mobile)
RESPONSIVE_CSS = """
<style>
    /* 1. Base Typography & Clean Background */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 100% !important;
    }

    /* 2. Metric Cards (Desktop Default) */
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 14px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }

    /* 3. Smooth Horizontal Scroll for Tabs */
    div[data-testid="stTabs"] {
        overflow-x: auto;
    }
    button[data-baseweb="tab"] {
        font-size: 14px !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
    }

    /* 4. Responsive Mobile / Tablet Layout (@media max-width: 768px) */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.8rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-bottom: 1.5rem !important;
        }

        /* Metric cards wrap into 2 columns on mobile instead of crushing into 5 slivers */
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 10px !important;
        }
        div[data-testid="column"] {
            flex: 1 1 calc(50% - 10px) !important;
            min-width: 140px !important;
        }

        /* Typography scaling for mobile */
        h1 {
            font-size: 20px !important;
        }
        h2, h3, h4 {
            font-size: 16px !important;
        }

        /* Metric text downscale */
        div[data-testid="stMetric"] {
            padding: 8px 10px !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 18px !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 11px !important;
        }

        /* Tab buttons touch-friendly on mobile */
        button[data-baseweb="tab"] {
            font-size: 12px !important;
            padding: 6px 10px !important;
        }
    }

    /* 5. Small Mobile Phones (@media max-width: 480px) */
    @media (max-width: 480px) {
        div[data-testid="column"] {
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
    }
</style>
"""
st.markdown(RESPONSIVE_CSS, unsafe_allow_html=True)


# =============================================================================
# Helper Functions: Data Loading & Cache
# =============================================================================
@st.cache_data(ttl=60)
def load_all_terminal_data():
    """Load latest files generated by the pipeline."""
    data = {}
    if os.path.exists(PATH_UNIVERSE):
        data['universe'] = pd.read_csv(PATH_UNIVERSE)
    else:
        data['universe'] = pd.DataFrame()

    if os.path.exists(PATH_FF3_SELECTED):
        data['selected'] = pd.read_csv(PATH_FF3_SELECTED)
    else:
        data['selected'] = pd.DataFrame()

    if os.path.exists(PATH_FF3_ALL):
        data['all_reg'] = pd.read_csv(PATH_FF3_ALL)
    else:
        data['all_reg'] = pd.DataFrame()

    if os.path.exists(PATH_BL_LATEST):
        data['weights_latest'] = pd.read_csv(PATH_BL_LATEST)
    else:
        data['weights_latest'] = pd.DataFrame()

    if os.path.exists(PATH_BL_SUMMARY):
        data['summary'] = pd.read_csv(PATH_BL_SUMMARY)
    else:
        data['summary'] = pd.DataFrame()

    if os.path.exists(PATH_BL_HISTORY):
        data['history'] = pd.read_csv(PATH_BL_HISTORY)
    else:
        data['history'] = pd.DataFrame()

    if os.path.exists(PATH_BL_BACKTEST):
        data['backtest'] = pd.read_csv(PATH_BL_BACKTEST)
    else:
        data['backtest'] = pd.DataFrame()

    if os.path.exists(PATH_VIEWS_READABLE):
        data['views'] = pd.read_csv(PATH_VIEWS_READABLE)
    else:
        data['views'] = pd.DataFrame()

    PATH_PRICES_CACHE_DIR = os.path.join(RAW_DATA_DIR, 'prices_cache.parquet')
    PATH_PRICES_CACHE_DOT = os.path.join(RAW_DATA_DIR, '.cache', 'prices_cache.parquet')
    if os.path.exists(PATH_PRICES_CACHE_DIR):
        data['prices_daily'] = pd.read_parquet(PATH_PRICES_CACHE_DIR)
    elif os.path.exists(PATH_PRICES_CACHE_DOT):
        data['prices_daily'] = pd.read_parquet(PATH_PRICES_CACHE_DOT)
    else:
        data['prices_daily'] = pd.DataFrame()

    PATH_VOL_CACHE_DIR = os.path.join(RAW_DATA_DIR, 'volume_cache.parquet')
    PATH_VOL_CACHE_DOT = os.path.join(RAW_DATA_DIR, '.cache', 'volume_cache.parquet')
    if os.path.exists(PATH_VOL_CACHE_DIR):
        data['volume_daily'] = pd.read_parquet(PATH_VOL_CACHE_DIR)
    elif os.path.exists(PATH_VOL_CACHE_DOT):
        data['volume_daily'] = pd.read_parquet(PATH_VOL_CACHE_DOT)
    else:
        data['volume_daily'] = pd.DataFrame()

    if os.path.exists(PATH_FACTORS):
        with open(PATH_FACTORS, 'r', encoding='utf-8', errors='latin-1') as f:
            lines = f.read().splitlines()
        rows = []
        for ln in lines:
            p = [x.strip() for x in ln.split(',')]
            if len(p) == 5 and p[0].isdigit() and len(p[0]) == 6:
                rows.append(p)
        ff = pd.DataFrame(rows, columns=['ym', 'MKT_RF', 'SMB', 'HML', 'RF'])
        ff['ym'] = pd.PeriodIndex(ff['ym'].str[:4] + '-' + ff['ym'].str[4:], freq='M')
        data['factors_ff'] = ff.set_index('ym').astype(float) / 100.0
    else:
        data['factors_ff'] = pd.DataFrame()

    if not data['prices_daily'].empty:
        m_px = data['prices_daily'].resample('ME').last()
        m_ret = m_px.pct_change().iloc[1:]
        m_px.index = m_px.index.to_period('M')
        m_ret.index = m_ret.index.to_period('M')
        data['monthly_prices_full'] = m_px
        data['returns_monthly_full'] = m_ret
    else:
        data['monthly_prices_full'] = pd.DataFrame()
        data['returns_monthly_full'] = pd.DataFrame()

    if not data['volume_daily'].empty:
        m_vol = data['volume_daily'].resample('ME').sum()
        m_vol.index = m_vol.index.to_period('M')
        data['monthly_volume_full'] = m_vol
    else:
        data['monthly_volume_full'] = pd.DataFrame()

    if os.path.exists(PATH_RETURNS_WIDE):
        ret_wide = pd.read_csv(PATH_RETURNS_WIDE, index_col=0, parse_dates=[0])
        ret_wide.index = ret_wide.index.to_period('M')
        data['returns_monthly'] = ret_wide
    else:
        data['returns_monthly'] = data['returns_monthly_full']

    return data


def execute_pipeline_command(args_list: List[str] = None):
    """Execute master quantitative pipeline synchronously and display log."""
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, 'run_monthly_pipeline.py')]
    if args_list:
        cmd.extend(args_list)
    with st.spinner("⏳ 실행 중: 주가 수집 -> FF3 팩터 회귀 -> XGBoost 견해 -> Black-Litterman 최적화..."):
        res = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True)
    if res.returncode == 0:
        st.success("✅ 파이프라인 전체 실행 완료! 최신 결과가 즉시 반영되었습니다.")
        st.cache_data.clear()
    else:
        st.error(f"❌ 파이프라인 실행 중 오류 발생 (코드 {res.returncode}):")
        st.code(res.stderr or res.stdout)


def _cum_ret(ret: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    g = 1.0 + ret
    out = g.shift(last)
    for j in range(last + 1, first + 1):
        out = out * g.shift(j)
    return out - 1.0


@st.cache_data(ttl=300, show_spinner=False)
def run_point_in_time_quant_model(
    selected_date_str: str,
    tau: float,
    delta: float,
    max_weight: float,
    t_cut: float
) -> Dict:
    """
    Point-in-Time (기준일 시점) 동적 퀀트 엔진:
    지정된 기준일(selected_date_str)까지의 데이터만 사용하여
    1) FF3 3팩터 OLS 종목 선정 (10년 롤링 윈도우)
    2) XGBoost 크로스섹션 견해(P, Q, Omega) 생성
    3) Black-Litterman 사후 기대수익률 및 최적 비중(SLSQP) 산출
    4) 기준일 당시 실제 종가 매핑
    """
    data = load_all_terminal_data()
    px_daily = data['prices_daily']
    m_ret_full = data['returns_monthly_full']
    m_px_full = data['monthly_prices_full']
    m_vol_full = data['monthly_volume_full']
    ff = data['factors_ff']
    univ = data['universe']

    if px_daily.empty or m_ret_full.empty or ff.empty:
        return {
            'ok': False,
            'warning': '기초 시계열/팩터 데이터가 부족하여 사전 계산된 최신 결과를 표시합니다.',
            'weights': data['weights_latest'],
            'selected': data['selected'],
            'all_reg': data['all_reg'],
            'prices_daily': px_daily,
            'returns_monthly': data['returns_monthly'],
            'window_months': 120,
            'port_stats': None,
        }

    as_of_ts = pd.Timestamp(selected_date_str)
    as_of_ym = pd.Period(selected_date_str, freq='M')

    px_pit = px_daily.loc[:as_of_ts]
    ret_pit = m_ret_full.loc[:as_of_ym]
    if px_pit.empty or len(ret_pit) < 24:
        return {
            'ok': False,
            'warning': f'{selected_date_str} 기준 가용 월간 데이터가 부족합니다 (최소 24개월 필요). 2011년 이후 날짜를 선택해주세요.',
            'weights': data['weights_latest'],
            'selected': data['selected'],
            'all_reg': data['all_reg'],
            'prices_daily': px_daily,
            'returns_monthly': data['returns_monthly'],
            'window_months': 0,
            'port_stats': None,
        }

    # -------------------------------------------------------------------------
    # 1. Point-in-Time FF3 3-Factor OLS Screening (Trailing up to 120M)
    # -------------------------------------------------------------------------
    sub_ret = ret_pit.tail(120)
    common = sub_ret.index.intersection(ff.index)
    R = sub_ret.loc[common].dropna(axis=1)
    F = ff.loc[common]
    T = len(common)

    if T < 24 or R.shape[1] < 5:
        return {
            'ok': False,
            'warning': f'{selected_date_str} 기준 FF3 팩터 공통 구간이 부족합니다 (T={T}).',
            'weights': data['weights_latest'],
            'selected': data['selected'],
            'all_reg': data['all_reg'],
            'prices_daily': px_pit,
            'returns_monthly': ret_pit,
            'window_months': T,
            'port_stats': None,
        }

    Y = R.sub(F['RF'], axis=0).to_numpy()
    X = np.column_stack([np.ones(T), F[['MKT_RF', 'SMB', 'HML']].to_numpy()])
    XtX_inv = np.linalg.inv(X.T @ X)
    B = XtX_inv @ (X.T @ Y)
    E = Y - X @ B
    s2 = (E ** 2).sum(axis=0) / max(T - 4, 1)
    se = np.sqrt(np.outer(np.diag(XtX_inv), s2))
    Tst = B / np.where(se > 0, se, np.nan)
    denom_r2 = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - (E ** 2).sum(axis=0) / np.where(denom_r2 > 0, denom_r2, np.nan)

    res = pd.DataFrame({
        'Ticker': R.columns,
        'alpha': B[0],
        'b_mkt': B[1],
        's_smb': B[2],
        'h_hml': B[3],
        't_alpha': Tst[0],
        'r2': r2,
        'resid_sd': np.sqrt(s2),
        'nobs': T,
        'ex_mean_m': Y.mean(axis=0),
        'ex_sd_m': Y.std(axis=0, ddof=1),
        'zero_ret_share': (np.abs(R.to_numpy()) < 1e-9).mean(axis=0),
    })

    # Map universe metadata (PERMNO, name, marketCap)
    if not univ.empty:
        tkr_col = 'yahoo_ticker' if 'yahoo_ticker' in univ.columns else ('symbol' if 'symbol' in univ.columns else univ.columns[0])
        u_map = univ.drop_duplicates(subset=[tkr_col]).set_index(tkr_col)
        res['PERMNO'] = res['Ticker'].map(u_map['PERMNO']) if 'PERMNO' in u_map.columns else np.nan
        res['name'] = res['Ticker'].map(u_map['name']).fillna(res['Ticker']) if 'name' in u_map.columns else res['Ticker']
        res['marketCap'] = res['Ticker'].map(u_map['marketCap']).fillna(1e9) if 'marketCap' in u_map.columns else 1e9
    else:
        res['PERMNO'] = np.nan
        res['name'] = res['Ticker']
        res['marketCap'] = 1e9

    res['PERMNO'] = pd.to_numeric(res['PERMNO'], errors='coerce').astype('Int64')
    res['alpha_ann'] = res['alpha'] * 12.0
    res['sharpe_raw'] = res['ex_mean_m'] / np.where(res['ex_sd_m'] > 0, res['ex_sd_m'], np.nan) * np.sqrt(12.0)

    # Sigma Guard & Liquidity Gate (identical to run_ff3.py)
    sd_tail = res['ex_sd_m'].quantile(0.05)
    sigma_flag = (
        (res['ex_sd_m'] < 0.015) |
        (res['zero_ret_share'] > 0.30) |
        ~np.isfinite(res['sharpe_raw']) |
        ((res['sharpe_raw'].abs() > 3.0) & (res['ex_sd_m'] < sd_tail))
    )
    res['sharpe'] = res['sharpe_raw'].where(~sigma_flag)
    liq_threshold = res['marketCap'].quantile(0.20)
    res['liq_pass'] = res['marketCap'] >= liq_threshold
    res['gate_pass'] = (
        (res['alpha'] > 0) &
        (res['t_alpha'] >= t_cut) &
        res['liq_pass'] &
        res['sharpe'].notna()
    )

    res = res.sort_values('t_alpha', ascending=False).reset_index(drop=True)
    res.insert(0, 'rank_all', np.arange(1, len(res) + 1))

    sel = res[res['gate_pass']].copy().reset_index(drop=True)
    warning_msg = None
    if len(sel) < 2:
        warning_msg = f"기준일({selected_date_str})에서 t(alpha) ≥ {t_cut:.1f} 게이트를 통과한 종목이 {len(sel)}개뿐이어서, t(alpha) 상위 5개 종목으로 대체 산출했습니다."
        sel = res[res['sharpe'].notna() & res['liq_pass']].head(5).copy().reset_index(drop=True)
    sel.insert(0, 'rank', np.arange(1, len(sel) + 1))

    # -------------------------------------------------------------------------
    # 2. Point-in-Time Covariance & Equilibrium Prior (Pi)
    # -------------------------------------------------------------------------
    assets = sel['Ticker'].tolist()
    n_assets = len(assets)
    ret_assets = ret_pit[assets].dropna(how='all')
    ret_cov = ret_assets.dropna().tail(36)
    if len(ret_cov) < 12:
        ret_cov = ret_assets.fillna(0.0).tail(36)

    Sigma = ret_cov.cov().to_numpy()
    # Small ridge for numerical stability
    Sigma = Sigma + np.eye(n_assets) * 1e-6

    mc_arr = sel['marketCap'].to_numpy(dtype=float)
    w_mkt = mc_arr / mc_arr.sum() if mc_arr.sum() > 0 else np.ones(n_assets) / n_assets
    Pi = delta * (Sigma @ w_mkt)

    # -------------------------------------------------------------------------
    # 3. Point-in-Time XGBoost Views (P, Q, Omega) & Black-Litterman Posterior
    # -------------------------------------------------------------------------
    FEATURES = ["mom1m", "mom6m", "mom12m", "chmom", "retvol", "dolvol", "beta", "idiovol"]
    mu_xgb = Pi.copy()
    omega_val = 0.0025

    try:
        sub_ret_ml = ret_pit[assets]
        sub_px_ml = m_px_full.loc[:as_of_ym, assets] if not m_px_full.empty else pd.DataFrame(100.0, index=sub_ret_ml.index, columns=assets)
        if not m_vol_full.empty:
            valid_vol_cols = [a for a in assets if a in m_vol_full.columns]
            sub_vol_ml = m_vol_full.loc[:as_of_ym].reindex(columns=assets).fillna(1e6)
        else:
            sub_vol_ml = pd.DataFrame(1e6, index=sub_ret_ml.index, columns=assets)

        f_ml = ff.reindex(sub_ret_ml.index).ffill()
        mkt_s = f_ml['MKT_RF']
        ex_ml = sub_ret_ml.sub(f_ml['RF'], axis=0)
        roll_fn = lambda x: x.rolling(12, min_periods=12).mean()
        var_m = roll_fn(mkt_s ** 2) - roll_fn(mkt_s) ** 2
        beta_ml = (roll_fn(ex_ml.mul(mkt_s, axis=0)) - roll_fn(ex_ml).mul(roll_fn(mkt_s), axis=0)).div(var_m, axis=0)
        var_ex = roll_fn(ex_ml ** 2) - roll_fn(ex_ml) ** 2

        raw_feats = {
            "mom1m": sub_ret_ml,
            "mom6m": _cum_ret(sub_ret_ml, 5, 1),
            "mom12m": _cum_ret(sub_ret_ml, 11, 1),
            "chmom": _cum_ret(sub_ret_ml, 5, 1) - _cum_ret(sub_ret_ml, 11, 6),
            "retvol": sub_ret_ml.rolling(12, min_periods=12).std(),
            "dolvol": np.log((sub_px_ml.reindex(sub_ret_ml.index) * sub_vol_ml.reindex(sub_ret_ml.index)).clip(lower=1.0)),
            "beta": beta_ml,
            "idiovol": np.sqrt((var_ex - beta_ml.pow(2).mul(var_m, axis=0)).clip(lower=0)),
        }
        cols_dict = {}
        for fname, wide in raw_feats.items():
            ranked = (2.0 * wide.round(10).rank(axis=1, pct=True) - 1.0).where(wide.notna())
            cols_dict[fname] = ranked.stack(future_stack=True)
        panel = pd.DataFrame(cols_dict)
        panel["has_history"] = raw_feats["mom12m"].notna().stack(future_stack=True)
        panel["target"] = sub_ret_ml.shift(-1).stack(future_stack=True)
        panel.index.names = ["ym", "Ticker"]
        panel[FEATURES] = panel[FEATURES].fillna(0.0)

        p_months = panel.index.get_level_values("ym")
        pred_ym = p_months.max()
        train_df = panel[(p_months < pred_ym) & panel["has_history"] & panel["target"].notna()]
        pred_df = panel.loc[pred_ym].reindex(assets)

        train_yms = train_df.index.get_level_values("ym").unique().sort_values()
        if len(train_yms) >= 18 and len(pred_df) == n_assets:
            val_cut = train_yms[-min(12, max(4, len(train_yms) // 5))]
            fit_part = train_df[train_df.index.get_level_values("ym") < val_cut]
            val_part = train_df[train_df.index.get_level_values("ym") >= val_cut]

            params = {
                "objective": "reg:squarederror",
                "max_depth": 3,
                "eta": 0.05,
                "seed": 42,
                "tree_method": "hist",
                "nthread": 0,
            }
            dtr = xgb.DMatrix(fit_part[FEATURES].to_numpy(dtype=np.float32), label=fit_part["target"].to_numpy())
            dval = xgb.DMatrix(val_part[FEATURES].to_numpy(dtype=np.float32), label=val_part["target"].to_numpy())
            bst = xgb.train(
                params, dtr,
                num_boost_round=150,
                evals=[(dval, "val")],
                early_stopping_rounds=20,
                verbose_eval=False
            )
            best_n = bst.best_iteration + 1
            d_full = xgb.DMatrix(train_df[FEATURES].to_numpy(dtype=np.float32), label=train_df["target"].to_numpy())
            final_bst = xgb.train(params, d_full, num_boost_round=best_n)
            mu_xgb = final_bst.predict(xgb.DMatrix(pred_df[FEATURES].to_numpy(dtype=np.float32))).astype(float)

            val_preds = bst.predict(dval, iteration_range=(0, best_n))
            val_mse = float(np.mean((val_part["target"].to_numpy() - val_preds) ** 2))
            omega_val = float(np.clip(val_mse, 1e-4, 0.05))
    except Exception:
        mu_xgb = Pi + sel['alpha'].to_numpy() * 0.5
        omega_val = 0.0025

    # Construct Relative View P (Top k Long, Bottom k Short)
    k_ls = max(1, min(3, n_assets // 3))
    order = np.argsort(-mu_xgb)
    P_row = np.zeros(n_assets)
    P_row[order[:k_ls]] = 1.0 / k_ls
    P_row[order[-k_ls:]] = -1.0 / k_ls
    P = P_row.reshape(1, -1)
    Q = np.array([float(P_row @ mu_xgb)])
    Omega = np.array([[omega_val]])

    # Black-Litterman Posterior Formula
    tau_Sigma = tau * Sigma
    P_tau_Sigma = P @ tau_Sigma
    view_cov_inv = np.linalg.inv(P_tau_Sigma @ P.T + Omega)
    er_bl = Pi + (tau_Sigma @ P.T @ view_cov_inv @ (Q - P @ Pi)).flatten()
    M_post = tau_Sigma - tau_Sigma @ P.T @ view_cov_inv @ P_tau_Sigma
    Sigma_BL = Sigma + M_post

    # Constrained Optimization (0 <= w_i <= max_weight, sum w_i = 1)
    def obj(w):
        return -(w @ er_bl - 0.5 * delta * (w @ Sigma_BL @ w))

    upper = max(max_weight, 1.0 / n_assets + 1e-4)
    bounds = [(0.0, upper) for _ in range(n_assets)]
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    init_w = np.ones(n_assets) / n_assets
    opt = minimize(obj, init_w, method='SLSQP', bounds=bounds, constraints=cons)
    if opt.success:
        w_const = np.clip(opt.x, 0.0, upper)
        w_const /= w_const.sum()
    else:
        w_const = init_w

    # Point-in-Time actual closing prices as of selected_date
    pit_prices = []
    for a in assets:
        if a in px_pit.columns:
            s = px_pit[a].dropna()
            pit_prices.append(round(float(s.iloc[-1]), 2) if not s.empty else 100.0)
        else:
            pit_prices.append(100.0)

    df_weights_pit = pd.DataFrame({
        'Ticker': assets,
        'Name': sel['name'].tolist(),
        'Last_Price': pit_prices,
        'MarketCap_Weight': np.round(w_mkt, 4),
        'Equal_Weight': np.round(np.ones(n_assets) / n_assets, 4),
        'Prior_Return_Ann': np.round(Pi * 12.0, 4),
        'XGBoost_Return_Ann': np.round(mu_xgb * 12.0, 4),
        'BL_Expected_Return_Ann': np.round(er_bl * 12.0, 4),
        'BL_Constrained_Weight': np.round(w_const, 4),
        'Tilt_vs_MarketCap': np.round(w_const - w_mkt, 4),
    })

    port_ret_ann = float(w_const @ er_bl * 12.0)
    port_vol_ann = float(np.sqrt(w_const @ Sigma @ w_const) * np.sqrt(12.0))
    port_sharpe = (port_ret_ann - 0.025) / port_vol_ann if port_vol_ann > 0 else 0.0

    # Trailing 36M realized max drawdown of this point-in-time portfolio
    port_hist_m = (ret_cov @ w_const)
    cum_w = (1.0 + port_hist_m).cumprod()
    dd_series = cum_w / cum_w.cummax() - 1.0
    port_mdd = float(dd_series.min()) if len(dd_series) > 0 else -0.20

    return {
        'ok': True,
        'warning': warning_msg,
        'weights': df_weights_pit,
        'selected': sel,
        'all_reg': res,
        'prices_daily': px_pit,
        'returns_monthly': ret_pit,
        'window_months': T,
        'actual_price_date': px_pit.index[-1].strftime('%Y-%m-%d'),
        'port_stats': {
            'ann_ret_pct': port_ret_ann * 100.0,
            'ann_vol_pct': port_vol_ann * 100.0,
            'sharpe': port_sharpe,
            'mdd_pct': port_mdd * 100.0,
        },
    }


# =============================================================================
# Load Terminal Data
# =============================================================================
terminal_data = load_all_terminal_data()
df_summary = terminal_data['summary']
df_history = terminal_data['history']
df_backtest = terminal_data['backtest']
df_views = terminal_data['views']
df_univ = terminal_data['universe']

max_avail_date = (
    terminal_data['prices_daily'].index[-1].date()
    if not terminal_data['prices_daily'].empty
    else datetime.now().date()
)

# =============================================================================
# Sidebar: Control Terminal & Model Hyperparameters
# =============================================================================
st.sidebar.markdown("### 🎛️ 퀀트 제어판 (Controls)")
st.sidebar.markdown("---")

# Portfolio Capital Setting
capital = st.sidebar.number_input("💰 운용 총 자본금 ($ USD)", min_value=1000, max_value=100_000_000, value=100_000, step=10_000)

# Rebalancing Date (Point-in-Time supported from 2011-01-01 to latest data date)
selected_date = st.sidebar.date_input(
    "📅 리밸런싱 기준일",
    value=max_avail_date,
    min_value=date(2011, 1, 1),
    max_value=max(datetime.now().date(), max_avail_date),
    help="기준일을 변경하면 해당 날짜까지의 데이터만 사용하여 FF3 종목 선정, XGBoost 견해, BL 비중, 당시 실제 주가를 즉시 재계산합니다."
)

st.sidebar.markdown("#### ⚙️ Black-Litterman & FF3 파라미터")
param_tau = st.sidebar.slider("타우 (Tau, 견해 비중계수)", min_value=0.01, max_value=0.50, value=0.10, step=0.01)
param_delta = st.sidebar.slider("위험회피계수 (Delta)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
param_max_weight = st.sidebar.slider("단일 종목 최대 비중 상한 (Cap)", min_value=0.10, max_value=0.40, value=0.20, step=0.02)
param_t_cut = st.sidebar.slider("FF3 알파 t-통계량 게이트 Cut", min_value=1.5, max_value=3.0, value=2.0, step=0.1)

st.sidebar.markdown("---")
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    btn_recalc = st.sidebar.button("🔄 비중 재계산", use_container_width=True)
with col_btn2:
    btn_pipeline = st.sidebar.button("🚀 전체 파이프라인", use_container_width=True)

if btn_pipeline:
    execute_pipeline_command([
        '--tau', str(param_tau),
        '--delta', str(param_delta),
        '--max-weight', str(param_max_weight),
        '--t-cut', str(param_t_cut)
    ])
    st.rerun()

if btn_recalc:
    run_point_in_time_quant_model.clear()
    st.toast(f"🔄 {selected_date.strftime('%Y-%m-%d')} 기준으로 FF3 종목 선정 및 BL 비중을 재계산했습니다.")

# Run Point-in-Time Dynamic Model (Recomputes automatically when date or any slider changes!)
with st.spinner(f"⏳ {selected_date.strftime('%Y-%m-%d')} 기준 FF3 종목 선정 및 BL 최적화 계산 중..."):
    pit_res = run_point_in_time_quant_model(
        selected_date_str=selected_date.strftime('%Y-%m-%d'),
        tau=param_tau,
        delta=param_delta,
        max_weight=param_max_weight,
        t_cut=param_t_cut
    )

df_weights_active = pit_res['weights']
df_selected = pit_res['selected']
df_all_reg = pit_res['all_reg']
df_prices_daily = pit_res['prices_daily']
df_returns_monthly = pit_res['returns_monthly']

# =============================================================================
# Top Ticker & Metric Status Cards (Native, Clean, High Contrast)
# =============================================================================
n_selected = len(df_selected) if not df_selected.empty else 0
total_mc_trillion = (df_univ['marketCap'].sum() / 1e12) if not df_univ.empty else 0.0

if pit_res.get('port_stats'):
    ps = pit_res['port_stats']
    ann_ret_pct = ps['ann_ret_pct']
    ann_vol_pct = ps['ann_vol_pct']
    sharpe_val = ps['sharpe']
    mdd_pct = ps['mdd_pct']
elif not df_summary.empty:
    bl_row = df_summary[df_summary['Portfolio'].str.contains('Constrained')].iloc[0]
    ann_ret_pct = bl_row['Annualized_Return'] * 100
    ann_vol_pct = bl_row['Annualized_Volatility'] * 100
    sharpe_val = bl_row['Sharpe_Ratio']
    mdd_pct = bl_row['Max_Drawdown'] * 100
else:
    ann_ret_pct, ann_vol_pct, sharpe_val, mdd_pct = 49.38, 28.15, 1.64, -27.31

st.title("📈 미국 상위 80% 주식 퀀트 투자 터미널")
actual_px_dt = pit_res.get('actual_price_date', selected_date.strftime('%Y-%m-%d'))
win_m = pit_res.get('window_months', 120)
is_historical_mode = selected_date < (max_avail_date - timedelta(days=7))
mode_badge = f"🕰️ 과거 시점 백테스트 모드 ({actual_px_dt} 종가 기준)" if is_historical_mode else f"🟢 최신 기준일 모드 ({actual_px_dt} 종가 기준)"
st.caption(f"{mode_badge} | FF3 회귀 윈도우: {win_m}개월 | t-Cut: {param_t_cut:.1f} | τ={param_tau:.2f}, δ={param_delta:.1f}, Cap={param_max_weight*100:.0f}%")

if pit_res.get('warning'):
    st.warning(pit_res['warning'])

m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
m_c1.metric("선정 종목 수", f"{n_selected} 종목", f"유니버스 {len(df_all_reg)}개 중")
m_c2.metric("BL 포트폴리오 연수익률", f"{ann_ret_pct:.1f}%", "사후 기대 E[R]")
m_c3.metric("포트폴리오 연 변동성", f"{ann_vol_pct:.1f}%", "직전 36M 공분산")
m_c4.metric("기대 샤프 지수", f"{sharpe_val:.2f}", "무위험 2.5% 기준")
m_c5.metric("직전 36M 최대 낙폭", f"{mdd_pct:.1f}%", f"{actual_px_dt} 기준")

st.markdown("---")

# =============================================================================
# Main Multi-Tab Workspace
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 [1. 포트폴리오 주문 및 자산배분 매트릭스]",
    "📈 [2. 수익률 다중 모델 시계열 & ML 예측]",
    "🏛️ [3. FF3 팩터 랭킹 및 유니버스 전광판]",
    "🧠 [4. XGBoost 견해 진단 및 장기 백테스트]"
])

# -----------------------------------------------------------------------------
# TAB 1: ORDER & ALLOCATION MATRIX
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("📋 포트폴리오 목표 비중 및 매수 주문 수량 계산기")
    st.caption("현재 주가를 반영한 목표 비중 배정 금액 및 필요 주문 주수(Shares) 산출표")

    latest_prices = {}
    # 1. First priority: Read directly from df_weights_active if 'Last_Price' exists
    if 'Last_Price' in df_weights_active.columns:
        for _, row in df_weights_active.iterrows():
            p_val = row.get('Last_Price')
            if pd.notna(p_val) and float(p_val) > 0:
                latest_prices[row['Ticker']] = round(float(p_val), 2)

    # 2. Second priority: Read latest valid closing quote from df_prices_daily
    if not df_prices_daily.empty:
        for tkr in df_weights_active['Ticker']:
            if tkr in df_prices_daily.columns:
                s_px = df_prices_daily[tkr].dropna()
                if not s_px.empty:
                    latest_prices[tkr] = round(float(s_px.iloc[-1]), 2)

    # 3. Third priority: Live fetch from yfinance if any ticker is still missing
    missing_tkrs = [t for t in df_weights_active['Ticker'] if t not in latest_prices or latest_prices[t] <= 0]
    if missing_tkrs:
        try:
            import yfinance as yf
            live_data = yf.download(missing_tkrs, period='5d', progress=False)
            close_data = live_data['Close'] if 'Close' in live_data else live_data
            for tkr in missing_tkrs:
                if tkr in close_data.columns:
                    s_live = close_data[tkr].dropna()
                    if not s_live.empty:
                        latest_prices[tkr] = round(float(s_live.iloc[-1]), 2)
        except Exception:
            pass

    df_grid = df_weights_active.copy()
    df_grid['Last_Price'] = df_grid['Ticker'].map(latest_prices).fillna(100.0).round(2)
    df_grid['Alloc_USD'] = (df_grid['BL_Constrained_Weight'] * capital).round(2)
    df_grid['Target_Shares'] = (df_grid['Alloc_USD'] / df_grid['Last_Price']).fillna(0).astype(int)

    # Clean native Streamlit Dataframe (NO HTML raw code, no broken <tr> tags!)
    display_table = pd.DataFrame({
        "티커": df_grid['Ticker'],
        "기업명": df_grid.get('Name', df_grid['Ticker']),
        "현재가 ($)": df_grid['Last_Price'],
        "BL 목표 비중": df_grid['BL_Constrained_Weight'] * 100,
        "시총 기준 비중": df_grid['MarketCap_Weight'] * 100,
        "비중 틸트": df_grid['Tilt_vs_MarketCap'] * 100,
        "배정 금액 ($)": df_grid['Alloc_USD'],
        "주문 수량": df_grid['Target_Shares'],
        "사전 균형수익 (Π)": df_grid.get('Prior_Return_Ann', 0) * 100,
        "ML 견해수익 (μ)": df_grid.get('XGBoost_Return_Ann', 0) * 100,
        "BL 사후기대 (E[R])": df_grid.get('BL_Expected_Return_Ann', 0) * 100,
    })

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
        height=380,
        column_config={
            "현재가 ($)": st.column_config.NumberColumn(format="$%.2f"),
            "BL 목표 비중": st.column_config.NumberColumn(format="%.2f%%"),
            "시총 기준 비중": st.column_config.NumberColumn(format="%.2f%%"),
            "비중 틸트": st.column_config.NumberColumn(format="%+.2f%%p"),
            "배정 금액 ($)": st.column_config.NumberColumn(format="$%,.2f"),
            "주문 수량": st.column_config.NumberColumn(format="%d 주"),
            "사전 균형수익 (Π)": st.column_config.NumberColumn(format="%+.2f%%"),
            "ML 견해수익 (μ)": st.column_config.NumberColumn(format="%+.2f%%"),
            "BL 사후기대 (E[R])": st.column_config.NumberColumn(format="%+.2f%%"),
        }
    )

    # Visual Comparison: Clean White Charts
    col_c1, col_c2 = st.columns([6, 5])
    
    with col_c1:
        st.markdown("##### 📊 시가총액 가중 vs Black-Litterman 20% 캡 최적 비중")
        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(
            x=df_grid['Ticker'],
            y=df_grid['MarketCap_Weight'] * 100,
            name='시총 비중 (Benchmark)',
            marker_color='#94a3b8'
        ))
        fig_w.add_trace(go.Bar(
            x=df_grid['Ticker'],
            y=df_grid['BL_Constrained_Weight'] * 100,
            name='BL 최적 비중 (Constrained)',
            marker_color='#2563eb'
        ))
        fig_w.add_hline(y=param_max_weight * 100, line_dash="dash", line_color="#dc2626",
                        annotation_text=f"상한 한도 ({param_max_weight*100:.0f}%)", annotation_position="top right")
        fig_w.update_layout(
            template='plotly_white',
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            barmode='group',
            yaxis=dict(title='비중 (%)'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_w, use_container_width=True)

    with col_c2:
        st.markdown("##### 🧬 포트폴리오 자산 간 상관계수 히트맵 (36M Correlation)")
        if not df_returns_monthly.empty:
            valid_tkrs = [t for t in df_grid['Ticker'] if t in df_returns_monthly.columns]
            corr_mat = df_returns_monthly[valid_tkrs].dropna().tail(36).corr().round(2)
            fig_corr = px.imshow(
                corr_mat,
                text_auto=True,
                color_continuous_scale='Blues',
                template='plotly_white'
            )
            fig_corr.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_corr, use_container_width=True)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# TAB 2: RETURN FORECASTING & TRAJECTORY CENTER
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("🔮 개별 종목 및 포트폴리오 다중 모델 수익률 & 주가 궤적 예측")
    st.caption("블랙-리터만 결합 모형, XGBoost 머신러닝, AR(2) 자기회귀, 지수평활(ETS)을 결합한 미래 복리 궤적 및 신뢰구간 분석.")

    col_ctrl1, col_ctrl2 = st.columns([6, 6])
    with col_ctrl1:
        selected_stock = st.selectbox("🎯 분석 및 예측 종목 선택", df_grid['Ticker'].tolist(), index=0)
    with col_ctrl2:
        horizon_months = st.slider("예측 시계 (Forecast Horizon, 개월)", min_value=3, max_value=24, value=12, step=3)

    view_mode = st.radio(
        "📊 예측 곡선 표시 형식 (View Mode)",
        ["💵 예상 주가 궤적 ($ USD) & 90% 신뢰구간 팬 (권장)", "📈 누적 복리 수익률 궤적 (%)", "📊 월간 수익률 변동성 시뮬레이션 (%)"],
        horizontal=True
    )

    col_fc1, col_fc2 = st.columns([7, 5])

    with col_fc1:
        if not df_returns_monthly.empty and selected_stock in df_returns_monthly.columns:
            ts_series = df_returns_monthly[selected_stock].dropna().astype(float)
            
            # 1. Historical monthly prices and dates
            if not df_prices_daily.empty and selected_stock in df_prices_daily.columns:
                stock_daily_px = df_prices_daily[selected_stock].dropna()
                stock_m_px = stock_daily_px.resample('ME').last().tail(12)
                hist_dates = [d.strftime('%Y-%m') for d in stock_m_px.index]
                hist_prices = stock_m_px.values
                current_price = float(hist_prices[-1])
                last_dt = stock_m_px.index[-1]
            else:
                current_price = 100.0
                hist_dates = [f"M-{12-i}" for i in range(12)]
                hist_prices = np.full(12, 100.0)
                last_dt = datetime.now()

            # Future monthly dates connected seamlessly from last_dt
            future_dates = [hist_dates[-1]] + [(last_dt + pd.DateOffset(months=i)).strftime('%Y-%m') for i in range(1, horizon_months + 1)]
            
            # Annual & monthly model parameters
            row_stk = df_grid[df_grid['Ticker'] == selected_stock].iloc[0]
            bl_ann_ret = float(row_stk['BL_Expected_Return_Ann'])
            xgb_ann_ret = float(row_stk['XGBoost_Return_Ann'])
            bl_m_ret = bl_ann_ret / 12.0
            xgb_m_ret = xgb_ann_ret / 12.0
            stock_vol_ann = float(ts_series.tail(36).std() * np.sqrt(12.0))
            if np.isnan(stock_vol_ann) or stock_vol_ann <= 0:
                stock_vol_ann = 0.30

            # Time Series Predictions (AR, ETS)
            try:
                ar_model = AutoReg(ts_series.iloc[-48:], lags=2).fit()
                pred_ar_m = ar_model.predict(start=len(ts_series.iloc[-48:]), end=len(ts_series.iloc[-48:]) + horizon_months - 1).values
            except Exception:
                pred_ar_m = np.full(horizon_months, bl_m_ret)

            try:
                es_model = ExponentialSmoothing(ts_series.iloc[-36:], initialization_method='estimated').fit()
                pred_es_m = es_model.forecast(horizon_months).values
            except Exception:
                pred_es_m = np.full(horizon_months, bl_m_ret)

            # Compounded trajectories starting at current_price
            steps = np.arange(horizon_months + 1)
            t_years = steps / 12.0
            
            # BL & XGB price trajectories
            p_bl = current_price * ((1.0 + bl_m_ret) ** steps)
            p_xgb = current_price * ((1.0 + xgb_m_ret) ** steps)

            # AR price trajectory
            p_ar = [current_price]
            for r in pred_ar_m:
                p_ar.append(p_ar[-1] * (1.0 + r))
            p_ar = np.array(p_ar)

            # ETS price trajectory
            p_es = [current_price]
            for r in pred_es_m:
                p_es.append(p_es[-1] * (1.0 + r))
            p_es = np.array(p_es)

            # Geometric Brownian Motion Confidence Fan Cone
            drift_gbm = (bl_ann_ret - 0.5 * (stock_vol_ann ** 2)) * t_years
            p_p90 = current_price * np.exp(drift_gbm + 1.645 * stock_vol_ann * np.sqrt(t_years))
            p_p10 = current_price * np.exp(drift_gbm - 1.645 * stock_vol_ann * np.sqrt(t_years))
            p_p75 = current_price * np.exp(drift_gbm + 0.674 * stock_vol_ann * np.sqrt(t_years))
            p_p25 = current_price * np.exp(drift_gbm - 0.674 * stock_vol_ann * np.sqrt(t_years))

            # Stochastic Simulation Path (Realistic market fluctuation path)
            np.random.seed(int(abs(hash(selected_stock)) % 100000))
            z_sim = np.random.randn(horizon_months)
            p_sim = [current_price]
            for i in range(horizon_months):
                step_r = bl_m_ret + (stock_vol_ann / np.sqrt(12.0)) * z_sim[i]
                p_sim.append(p_sim[-1] * (1.0 + step_r))
            p_sim = np.array(p_sim)

            fig_stk = go.Figure()

            if "예상 주가 궤적" in view_mode:
                # 1. Price Mode
                fig_stk.add_trace(go.Scatter(
                    x=future_dates, y=p_p90, mode='lines', line=dict(width=0),
                    showlegend=False, hoverinfo='skip'
                ))
                fig_stk.add_trace(go.Scatter(
                    x=future_dates, y=p_p10, mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(37, 99, 235, 0.10)',
                    name='90% 신뢰구간 (P10 ~ P90)'
                ))
                fig_stk.add_trace(go.Scatter(
                    x=future_dates, y=p_p75, mode='lines', line=dict(width=0),
                    showlegend=False, hoverinfo='skip'
                ))
                fig_stk.add_trace(go.Scatter(
                    x=future_dates, y=p_p25, mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(37, 99, 235, 0.22)',
                    name='50% 신뢰구간 (P25 ~ P75)'
                ))
                fig_stk.add_trace(go.Scatter(
                    x=hist_dates, y=hist_prices,
                    mode='lines+markers', name='과거 실제 주가 (Actual)',
                    line=dict(color='#475569', width=2.5)
                ))
                fig_stk.add_trace(go.Scatter(
                    x=future_dates, y=p_bl,
                    mode='lines+markers', name=f'BL 기대 주가 (+{bl_ann_ret*100:.1f}%)',
                    line=dict(color='#2563eb', width=3.5)
                ))
                fig_stk.add_trace(go.Scatter(
                    x=future_dates, y=p_xgb,
                    mode='lines+markers', name=f'XGBoost ML 주가 (+{xgb_ann_ret*100:.1f}%)',
                    line=dict(color='#16a34a', width=2.5, dash='dash')
                ))
                fig_stk.add_trace(go.Scatter(
                    x=future_dates, y=p_ar,
                    mode='lines', name='AR(2) 시계열 궤적',
                    line=dict(color='#d97706', width=2, dash='dot')
                ))
                fig_stk.add_trace(go.Scatter(
                    x=future_dates, y=p_sim,
                    mode='lines', name='몬테카를로 실현 시뮬레이션 경로',
                    line=dict(color='#9333ea', width=1.8, dash='longdash')
                ))
                fig_stk.add_hline(y=current_price, line_dash="dash", line_color="#94a3b8", annotation_text=f"현재가 ${current_price:,.2f}")
                fig_stk.update_layout(yaxis=dict(title='주가 ($ USD)'))

            elif "누적 복리" in view_mode:
                # 2. Cumulative Return Mode (%)
                cum_hist_pct = (hist_prices / current_price - 1.0) * 100
                cum_bl_pct = (p_bl / current_price - 1.0) * 100
                cum_xgb_pct = (p_xgb / current_price - 1.0) * 100
                cum_ar_pct = (p_ar / current_price - 1.0) * 100
                cum_p90_pct = (p_p90 / current_price - 1.0) * 100
                cum_p10_pct = (p_p10 / current_price - 1.0) * 100
                cum_sim_pct = (p_sim / current_price - 1.0) * 100

                fig_stk.add_trace(go.Scatter(
                    x=future_dates, y=cum_p90_pct, mode='lines', line=dict(width=0), showlegend=False
                ))
                fig_stk.add_trace(go.Scatter(
                    x=future_dates, y=cum_p10_pct, mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(37, 99, 235, 0.12)', name='90% 신뢰구간'
                ))
                fig_stk.add_trace(go.Scatter(
                    x=hist_dates, y=cum_hist_pct, mode='lines+markers', name='과거 실제 누적수익률',
                    line=dict(color='#475569', width=2.5)
                ))
                fig_stk.add_trace(go.Scatter(
                    x=future_dates, y=cum_bl_pct, mode='lines+markers', name='BL 복리 누적 기대치',
                    line=dict(color='#2563eb', width=3.5)
                ))
                fig_stk.add_trace(go.Scatter(
                    x=future_dates, y=cum_xgb_pct, mode='lines+markers', name='XGBoost 누적 예측률',
                    line=dict(color='#16a34a', width=2.5, dash='dash')
                ))
                fig_stk.add_trace(go.Scatter(
                    x=future_dates, y=cum_ar_pct, mode='lines', name='AR(2) 누적 궤적',
                    line=dict(color='#d97706', width=2, dash='dot')
                ))
                fig_stk.add_trace(go.Scatter(
                    x=future_dates, y=cum_sim_pct, mode='lines', name='변동성 반영 시뮬레이션',
                    line=dict(color='#9333ea', width=1.8, dash='longdash')
                ))
                fig_stk.add_hline(y=0.0, line_dash="dash", line_color="#94a3b8", annotation_text="기준점 (0%)")
                fig_stk.update_layout(yaxis=dict(title='누적 수익률 (%)'))

            else:
                # 3. Monthly Return Mode (%)
                m_hist_pct = ts_series.tail(12) * 100
                m_future_dates = [(last_dt + pd.DateOffset(months=i)).strftime('%Y-%m') for i in range(1, horizon_months + 1)]
                m_bl_pct = np.full(horizon_months, bl_m_ret * 100)
                m_xgb_pct = np.full(horizon_months, xgb_m_ret * 100)
                m_vol_pct = (stock_vol_ann / np.sqrt(12.0)) * 100

                fig_stk.add_trace(go.Scatter(
                    x=m_future_dates, y=m_bl_pct + m_vol_pct, mode='lines', line=dict(width=0), showlegend=False
                ))
                fig_stk.add_trace(go.Scatter(
                    x=m_future_dates, y=m_bl_pct - m_vol_pct, mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(37, 99, 235, 0.12)', name='월간 변동성 범위 (±1σ)'
                ))
                fig_stk.add_trace(go.Scatter(
                    x=hist_dates, y=m_hist_pct, mode='lines+markers', name='과거 월간 실현치',
                    line=dict(color='#475569', width=2)
                ))
                fig_stk.add_trace(go.Scatter(
                    x=[hist_dates[-1]] + m_future_dates, y=[m_hist_pct.iloc[-1]] + list(m_bl_pct),
                    mode='lines+markers', name='BL 월간 기대치 (드리프트)',
                    line=dict(color='#2563eb', width=3)
                ))
                fig_stk.add_trace(go.Scatter(
                    x=[hist_dates[-1]] + m_future_dates, y=[m_hist_pct.iloc[-1]] + list(m_xgb_pct),
                    mode='lines+markers', name='XGBoost 월간 예측치',
                    line=dict(color='#16a34a', width=2, dash='dot')
                ))
                sim_m_returns = [bl_m_ret * 100 + m_vol_pct * z_sim[i] for i in range(horizon_months)]
                fig_stk.add_trace(go.Scatter(
                    x=[hist_dates[-1]] + m_future_dates, y=[m_hist_pct.iloc[-1]] + sim_m_returns,
                    mode='lines+markers', name='시뮬레이션 월간 실현 경로',
                    line=dict(color='#9333ea', width=2, dash='dash')
                ))
                fig_stk.add_hline(y=0.0, line_dash="dash", line_color="#94a3b8")
                fig_stk.update_layout(yaxis=dict(title='월간 수익률 (%)'))

            fig_stk.update_layout(
                template='plotly_white',
                height=380,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_stk, use_container_width=True)

            # Stock Key Target Metrics
            s_c1, s_c2, s_c3, s_c4, s_c5 = st.columns(5)
            s_c1.metric("현재가", f"${current_price:,.2f}")
            s_c2.metric(f"{horizon_months}M BL 목표가", f"${p_bl[-1]:,.2f}", f"+{(p_bl[-1]/current_price - 1)*100:.1f}%")
            s_c3.metric("Bull Case (상위 10%)", f"${p_p90[-1]:,.2f}", f"+{(p_p90[-1]/current_price - 1)*100:.1f}%")
            s_c4.metric("Bear Case (하위 10%)", f"${p_p10[-1]:,.2f}", f"{(p_p10[-1]/current_price - 1)*100:+.1f}%")
            s_c5.metric("연환산 변동성 (σ)", f"{stock_vol_ann*100:.1f}%", "36M 실측치")

    with col_fc2:
        st.markdown("##### ⚖️ 기대수익률 3원 분해 (Decomposition)")
        fig_decomp = go.Figure()
        fig_decomp.add_trace(go.Bar(
            y=df_grid['Ticker'],
            x=df_grid['Prior_Return_Ann'] * 100,
            name='시장 균형 Prior (Π)',
            orientation='h',
            marker_color='#94a3b8'
        ))
        fig_decomp.add_trace(go.Bar(
            y=df_grid['Ticker'],
            x=df_grid['XGBoost_Return_Ann'] * 100,
            name='XGBoost ML 견해 (μ)',
            orientation='h',
            marker_color='#16a34a'
        ))
        fig_decomp.add_trace(go.Bar(
            y=df_grid['Ticker'],
            x=df_grid['BL_Expected_Return_Ann'] * 100,
            name='BL 사후 결합 (E[R])',
            orientation='h',
            marker_color='#2563eb'
        ))
        fig_decomp.update_layout(
            template='plotly_white',
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            barmode='group',
            xaxis=dict(title='연환산 기대수익률 (%)'),
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_decomp, use_container_width=True)

    # 2. Portfolio Monte Carlo Simulation Fan Chart
    st.markdown("---")
    st.markdown("##### 🎲 전체 포트폴리오 몬테카를로 자산가치 궤적 시뮬레이션 (1,000 Paths, Correlated Gaussian)")
    
    if not df_returns_monthly.empty and not df_grid.empty:
        valid_tkrs = [t for t in df_grid['Ticker'] if t in df_returns_monthly.columns]
        mu_vec = (df_grid['BL_Expected_Return_Ann'].to_numpy() / 12.0)
        Sigma_mat = df_returns_monthly[valid_tkrs].dropna().tail(36).cov().to_numpy()
        w_vec = df_grid['BL_Constrained_Weight'].to_numpy()

        S_sim = 1000
        H_sim = horizon_months
        L_chol = np.linalg.cholesky(Sigma_mat)

        np.random.seed(42)
        Z_norm = np.random.randn(S_sim, H_sim, len(valid_tkrs))
        R_sim = mu_vec + np.einsum('shn,mn->shm', Z_norm, L_chol)
        R_p_sim = np.sum(R_sim * w_vec, axis=2)

        # Cumulative wealth trajectory starting from initial capital
        W_paths = capital * np.cumprod(1.0 + R_p_sim, axis=1)
        W_paths = np.column_stack([np.full(S_sim, capital), W_paths])

        time_steps = [f"M+{i}" for i in range(H_sim + 1)]
        p5 = np.percentile(W_paths, 5, axis=0)
        p25 = np.percentile(W_paths, 25, axis=0)
        p50 = np.percentile(W_paths, 50, axis=0)
        p75 = np.percentile(W_paths, 75, axis=0)
        p95 = np.percentile(W_paths, 95, axis=0)

        fig_fan = go.Figure()
        # 5% ~ 95% Fan Band
        fig_fan.add_trace(go.Scatter(
            x=time_steps, y=p95, mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ))
        fig_fan.add_trace(go.Scatter(
            x=time_steps, y=p5, mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(37, 99, 235, 0.12)',
            name='90% 신뢰구간 (P5 ~ P95)'
        ))
        # 25% ~ 75% Fan Band
        fig_fan.add_trace(go.Scatter(
            x=time_steps, y=p75, mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ))
        fig_fan.add_trace(go.Scatter(
            x=time_steps, y=p25, mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(37, 99, 235, 0.28)',
            name='50% 신뢰구간 (P25 ~ P75)'
        ))
        # Median Path
        fig_fan.add_trace(go.Scatter(
            x=time_steps, y=p50, mode='lines+markers',
            line=dict(color='#1d4ed8', width=3),
            name='중간값 궤적 (P50 Median)'
        ))
        # Initial baseline
        fig_fan.add_hline(y=capital, line_dash="dash", line_color="#475569", annotation_text=f"원금 (${capital:,.0f})")

        fig_fan.update_layout(
            template='plotly_white',
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(title='포트폴리오 평가 자산 ($ USD)'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_fan, use_container_width=True)

        c_mc1, c_mc2, c_mc3, c_mc4 = st.columns(4)
        c_mc1.metric("12M 중앙값 예상 자산", f"${p50[-1]:,.0f}", f"+{(p50[-1]/capital - 1)*100:.1f}%")
        c_mc2.metric("상위 5% Bull Case (P95)", f"${p95[-1]:,.0f}", f"+{(p95[-1]/capital - 1)*100:.1f}%")
        c_mc3.metric("하위 5% Bear Case (VaR 95%)", f"${p5[-1]:,.0f}", f"{(p5[-1]/capital - 1)*100:+.1f}%")
        prob_win = (W_paths[:, -1] > capital).mean() * 100
        c_mc4.metric("원금 이상 달성 확률", f"{prob_win:.1f}%", "Monte Carlo 1,000회")


# -----------------------------------------------------------------------------
# TAB 3: FF3 FACTOR RANKING BOARD
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("🏛️ Fama-French 3-Factor (FF3) 유니버스 전광판")
    st.caption("미국 시총 상위 80% 주식들의 10개년 Balanced Panel OLS 회귀 분석 및 게이트 통과 내역")

    col_srch, col_filt = st.columns([6, 6])
    with col_srch:
        search_txt = st.text_input("🔍 종목명 / 티커 검색", "")
    with col_filt:
        gate_only = st.checkbox("게이트 통과(Selected) 종목만 보기", value=False)

    df_display_reg = df_all_reg.copy() if not df_all_reg.empty else pd.DataFrame()
    if not df_display_reg.empty:
        if search_txt:
            df_display_reg = df_display_reg[
                df_display_reg['Ticker'].str.contains(search_txt.upper(), na=False) |
                df_display_reg['name'].str.contains(search_txt, case=False, na=False)
            ]
        if gate_only:
            df_display_reg = df_display_reg[df_display_reg['gate_pass'] == True]

        st.dataframe(
            df_display_reg[[
                'rank_all', 'Ticker', 'PERMNO', 'name', 't_alpha', 'alpha_ann', 'sharpe',
                'b_mkt', 's_smb', 'h_hml', 'r2', 'gate_pass'
            ]].rename(columns={
                'rank_all': '순위', 'Ticker': '티커', 'name': '기업명',
                't_alpha': 't(alpha)', 'alpha_ann': '연환산 알파', 'sharpe': '샤프지수',
                'b_mkt': '시장 베타', 's_smb': 'SMB 베타', 'h_hml': 'HML 베타',
                'r2': 'R-squared', 'gate_pass': '게이트 통과'
            }),
            use_container_width=True,
            hide_index=True,
            height=400
        )

        st.markdown("##### 🌌 알파 t-통계량 vs 시장 베타 (Market Beta) 산포도")
        fig_scatter = px.scatter(
            df_all_reg.dropna(subset=['t_alpha', 'b_mkt']),
            x='b_mkt', y='t_alpha',
            color='gate_pass',
            hover_name='Ticker',
            hover_data=['name', 'alpha_ann', 'sharpe'],
            color_discrete_map={True: '#2563eb', False: '#94a3b8'},
            template='plotly_white',
            labels={'b_mkt': '시장 베타 (Beta)', 't_alpha': '알파 유의성 t(alpha)', 'gate_pass': '게이트 통과'}
        )
        fig_scatter.add_hline(y=2.0, line_dash="dash", line_color="#dc2626", annotation_text="t = 2.0 Gate")
        fig_scatter.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 4: XGBOOST VIEWS & HISTORICAL BACKTEST
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("🧠 XGBoost 머신러닝 견해(Views) 및 실증 백테스트")
    st.caption("68개월간의 Out-of-Sample 실증 누적 자산 성장 곡선 및 위험/수익 지표")

    col_bt1, col_bt2 = st.columns([7, 5])
    
    with col_bt1:
        st.markdown("##### 📈 68개월(2021~2026) 누적 자산 성장 백테스트 곡선")
        if not df_backtest.empty:
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=df_backtest['hold_month'], y=df_backtest['cum_bl_const'] * 100,
                mode='lines', name='BL 제약 포트폴리오 (Cap 20%)',
                line=dict(color='#2563eb', width=3)
            ))
            fig_bt.add_trace(go.Scatter(
                x=df_backtest['hold_month'], y=df_backtest['cum_mkt_benchmark'] * 100,
                mode='lines', name='시가총액 가중 벤치마크',
                line=dict(color='#64748b', width=1.5, dash='dash')
            ))
            fig_bt.add_trace(go.Scatter(
                x=df_backtest['hold_month'], y=df_backtest['cum_eq_benchmark'] * 100,
                mode='lines', name='동일가중 (1/N) 벤치마크',
                line=dict(color='#d97706', width=1.5, dash='dot')
            ))
            fig_bt.update_layout(
                template='plotly_white',
                height=340,
                margin=dict(l=20, r=20, t=30, b=20),
                yaxis=dict(title='누적 수익률 (%)'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bt, use_container_width=True)

    with col_bt2:
        st.markdown("##### 📊 포트폴리오 장기 리스크/수익 성과 요약")
        if not df_summary.empty:
            st.dataframe(
                df_summary.rename(columns={
                    'Portfolio': '포트폴리오', 'Annualized_Return': '연환산 수익률',
                    'Annualized_Volatility': '연환산 변동성', 'Sharpe_Ratio': '샤프지수',
                    'Max_Drawdown': '최대 낙폭(MDD)', 'Total_Cumulative_Return': '누적 총수익률'
                }),
                use_container_width=True,
                hide_index=True,
                height=220
            )

    st.markdown("---")
    st.markdown("##### 📡 월별 XGBoost 롱/숏 견해 예측치($Q$) vs 실제 실현 스프레드($Q_{real}$)")
    if not df_views.empty:
        st.dataframe(df_views.tail(12), use_container_width=True, hide_index=True, height=280)
