"""
Fama-French 3-Factor (FF3) Stock Ranking & Selection Pipeline
Based on methodology from ff3_selection_2000_2010.ipynb

Model:
    R_i - RF = alpha_i + b_i * (MKT - RF) + s_i * SMB + h_i * HML + e_i

Ranking:
    t(alpha) descending

Gates:
    1. alpha > 0
    2. t(alpha) >= T_CUT (default: 2.0)
    3. Liquidity: Bottom LIQ_CUT (default: 20%) excluded
    4. Sigma guard: Sharpe ratio / volatility stability checks
"""

import os
import sys
import io
import time
import zipfile
import argparse
import urllib.request
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
FF3_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'ff3')

PATH_STOCK_MONTHLY = os.path.join(RAW_DATA_DIR, 'monthly_returns_10y.csv')
PATH_UNIVERSE = os.path.join(RAW_DATA_DIR, 'stock_universe_top80.csv')
PATH_FF_FACTORS = os.path.join(FF3_DATA_DIR, 'F-F_Research_Data_Factors.csv')

OUTPUT_SELECTED = os.path.join(FF3_DATA_DIR, 'ff3_selected.csv')
OUTPUT_REGRESSION_ALL = os.path.join(FF3_DATA_DIR, 'ff3_regression_all.csv')
OUTPUT_PERFORMANCE = os.path.join(FF3_DATA_DIR, 'ff3_insample_performance.csv')

KENNETH_FRENCH_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip'


def ensure_directories():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(FF3_DATA_DIR, exist_ok=True)


def download_or_load_ff3_factors(force_download=False):
    """Download official Kenneth French FF3 monthly factors or load local cache."""
    ensure_directories()
    if not os.path.exists(PATH_FF_FACTORS) or force_download:
        print("[FF3 Factors] Downloading official Kenneth French 3-Factor dataset...")
        try:
            req = urllib.request.Request(KENNETH_FRENCH_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                z = zipfile.ZipFile(io.BytesIO(resp.read()))
                filename = [f for f in z.namelist() if f.endswith('.csv')][0]
                content = z.open(filename).read().decode('latin-1')

            with open(PATH_FF_FACTORS, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[FF3 Factors] Saved factors to {PATH_FF_FACTORS}")
        except Exception as e:
            if os.path.exists(PATH_FF_FACTORS):
                print(f"[FF3 Factors] Warning: Download failed ({e}), using existing factor file.")
            else:
                raise RuntimeError(f"Failed to fetch Fama-French factor data: {e}")

    # Parse Kenneth French CSV file format
    with open(PATH_FF_FACTORS, 'r', encoding='utf-8', errors='latin-1') as f:
        lines = f.read().splitlines()

    rows = []
    for ln in lines:
        p = [x.strip() for x in ln.split(',')]
        if len(p) == 5 and p[0].isdigit() and len(p[0]) == 6:
            rows.append(p)

    ff = pd.DataFrame(rows, columns=['ym', 'MKT_RF', 'SMB', 'HML', 'RF'])
    ff['ym'] = pd.PeriodIndex(ff['ym'].str[:4] + '-' + ff['ym'].str[4:], freq='M')
    ff = ff.set_index('ym').astype(float) / 100.0  # Convert percentage to decimal

    print(f"[FF3 Factors] Loaded factors from {ff.index.min()} to {ff.index.max()} ({len(ff)} months).")
    return ff


def run_ff3_ranking(t_cut=2.0, liq_cut=0.20, force_factors=False):
    ensure_directories()
    start_time = time.time()
    print("=" * 70)
    print(" Fama-French 3-Factor Security Ranking & Selection Pipeline")
    print(f" Ranking Metric: t(alpha) descending | T_CUT: {t_cut} | LIQ_CUT: {liq_cut*100:.0f}%")
    print("=" * 70)

    # 1. Load F-F 3 factors
    ff3 = download_or_load_ff3_factors(force_download=force_factors)

    # 2. Load stock monthly returns and universe metadata
    if not os.path.exists(PATH_STOCK_MONTHLY):
        raise FileNotFoundError(f"Missing monthly returns file: {PATH_STOCK_MONTHLY}. Run update_returns.py first.")

    raw_ret = pd.read_csv(PATH_STOCK_MONTHLY)
    raw_ret['ym'] = pd.PeriodIndex(raw_ret['YearMonth'], freq='M')

    universe_df = pd.DataFrame()
    if os.path.exists(PATH_UNIVERSE):
        universe_df = pd.read_csv(PATH_UNIVERSE)
        if 'symbol' in universe_df.columns:
            universe_df = universe_df.set_index('symbol')

    # 3. Determine common sample period
    sample_start = max(raw_ret['ym'].min(), ff3.index.min())
    sample_end = min(raw_ret['ym'].max(), ff3.index.max())
    print(f"\n[Sample] Analysis Window: {sample_start} to {sample_end}")

    sub_ret = raw_ret[(raw_ret['ym'] >= sample_start) & (raw_ret['ym'] <= sample_end)].copy()
    sub_ff = ff3.loc[sample_start:sample_end].copy()
    n_months = len(sub_ff)
    print(f"[Sample] Total months in regression: {n_months} months")

    # 4. Construct Balanced Panel (stocks valid across all sample months)
    piv_ret = sub_ret.pivot(index='ym', columns='Ticker', values='Return')
    valid_tickers = piv_ret.columns[piv_ret.notna().sum() == n_months]
    R = piv_ret[valid_tickers].copy()
    months = R.index

    print(f"[Universe] Total available stocks: {piv_ret.shape[1]} -> Balanced panel: {len(valid_tickers)} stocks")

    # 5. FF3 Multi-Asset Matrix OLS Regression
    # Model: R - RF = alpha + b*MKT_RF + s*SMB + h*HML + e
    F = sub_ff.loc[months, ['MKT_RF', 'SMB', 'HML']].to_numpy()
    rf = sub_ff.loc[months, 'RF'].to_numpy()
    Y = R.to_numpy() - rf[:, None]  # Excess returns (T x N)

    T = len(months)
    X = np.column_stack([np.ones(T), F])  # (T x 4)
    XtX_inv = np.linalg.inv(X.T @ X)
    B = XtX_inv @ X.T @ Y  # (4 x N)
    E = Y - X @ B          # Residuals (T x N)
    s2 = (E ** 2).sum(axis=0) / (T - 4)
    se = np.sqrt(np.outer(np.diag(XtX_inv), s2))
    Tst = B / se
    r2 = 1 - (E ** 2).sum(axis=0) / ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)

    # 6. Build Stock Metrics DataFrame
    res = pd.DataFrame({
        'Ticker': R.columns,
        'alpha': B[0],
        'b_mkt': B[1],
        's_smb': B[2],
        'h_hml': B[3],
        't_alpha': Tst[0],
        't_mkt': Tst[1],
        't_smb': Tst[2],
        't_hml': Tst[3],
        'r2': r2,
        'resid_sd': np.sqrt(s2),
        'se_alpha': se[0],
        'nobs': T,
        'ex_mean_m': Y.mean(axis=0),
        'ex_sd_m': Y.std(axis=0, ddof=1),
        'zero_ret_share': (np.abs(R.to_numpy()) < 1e-9).mean(axis=0),
    })

    # Join metadata
    if not universe_df.empty:
        res['PERMNO'] = res['Ticker'].map(universe_df['PERMNO'])
        res['name'] = res['Ticker'].map(universe_df['name'])
        res['marketCap'] = res['Ticker'].map(universe_df['marketCap'])
    else:
        res['PERMNO'] = np.nan
        res['name'] = res['Ticker']
        res['marketCap'] = 1.0

    res['PERMNO'] = pd.to_numeric(res['PERMNO'], errors='coerce').astype('Int64')
    res['alpha_ann'] = res['alpha'] * 12
    res['p_alpha_one'] = stats.t.sf(res['t_alpha'], df=T - 4)
    res['appraisal_ratio'] = res['alpha'] / res['resid_sd']
    res['sharpe_raw'] = res['ex_mean_m'] / res['ex_sd_m'] * np.sqrt(12)

    # 7. Sigma Guard (Filter out unstable / exploding Sharpe ratios)
    SIGMA_FLOOR_ABS = 0.015
    ZERO_RET_MAX = 0.30
    SHARPE_CAP = 3.0
    SIGMA_TAIL_PCTILE = 0.05
    sd_tail = res['ex_sd_m'].quantile(SIGMA_TAIL_PCTILE)

    sigma_flag = (
        (res['ex_sd_m'] < SIGMA_FLOOR_ABS) |
        (res['zero_ret_share'] > ZERO_RET_MAX) |
        ~np.isfinite(res['sharpe_raw']) |
        ((res['sharpe_raw'].abs() > SHARPE_CAP) & (res['ex_sd_m'] < sd_tail))
    )
    res['sharpe_valid'] = ~sigma_flag
    res['sharpe'] = res['sharpe_raw'].where(res['sharpe_valid'])
    print(f"[Gate] Sigma Guard filtered: {int(sigma_flag.sum())} stocks")

    # 8. Liquidity Gate
    liq_threshold = res['marketCap'].quantile(liq_cut)
    res['liq_pass'] = res['marketCap'] >= liq_threshold
    print(f"[Gate] Liquidity (Market Cap Top {100-liq_cut*100:.0f}%): {res['liq_pass'].sum()} passed")

    # 9. Gate Pass Evaluation: alpha > 0 & t(alpha) >= T_CUT & liq_pass & sharpe_valid
    res['gate_pass'] = (
        (res['alpha'] > 0) &
        (res['t_alpha'] >= t_cut) &
        res['liq_pass'] &
        res['sharpe'].notna()
    )

    # Multiple testing statistics
    m = len(res)
    p_vals = np.sort(res['p_alpha_one'].to_numpy())
    bh = np.where(p_vals <= np.arange(1, m + 1) / m * 0.05)[0]
    n_fdr = int(bh.max() + 1) if len(bh) else 0
    exp_chance = m * stats.t.sf(t_cut, df=T - 4)

    print(f"[Gate] Passed All Gates (alpha > 0, t(alpha) >= {t_cut}, liq, sigma): {res['gate_pass'].sum()} stocks")
    print(f"[Statistics] BH-FDR 5% discoveries: {n_fdr} | Expected by chance: {exp_chance:.1f} | Bonferroni p-cutoff: {0.05/m:.2e}")

    # 10. Sort by t(alpha) descending
    res = res.sort_values('t_alpha', ascending=False).reset_index(drop=True)
    res.insert(0, 'rank_all', np.arange(1, len(res) + 1))

    # Selected Subset
    sel = res[res['gate_pass']].copy().reset_index(drop=True)
    sel.insert(0, 'rank', np.arange(1, len(sel) + 1))

    # 11. In-Sample Portfolio Performance Evaluation
    def eval_portfolio(ex_series, name, n_assets):
        Xp = np.column_stack([np.ones(len(ex_series)), sub_ff.loc[ex_series.index, ['MKT_RF', 'SMB', 'HML']]])
        ip = np.linalg.inv(Xp.T @ Xp)
        b_p = ip @ Xp.T @ ex_series.to_numpy()
        e_p = ex_series.to_numpy() - Xp @ b_p
        s_p = np.sqrt(np.diag(ip) * (e_p @ e_p) / (len(ex_series) - 4))
        return {
            'portfolio': name,
            'n_names': n_assets,
            'months': len(ex_series),
            'ex_ret_ann': ex_series.mean() * 12,
            'vol_ann': ex_series.std() * np.sqrt(12),
            'sharpe': ex_series.mean() / ex_series.std() * np.sqrt(12),
            'ff3_alpha_ann': b_p[0] * 12,
            't_alpha': b_p[0] / s_p[0],
            'b_mkt': b_p[1],
            's_smb': b_p[2],
            'h_hml': b_p[3]
        }

    EX = R.sub(rf, axis=0)
    perf_records = []
    if len(sel) > 0:
        perf_records.append(eval_portfolio(EX[sel['Ticker']].mean(axis=1), 'Selected Portfolio (EW)', len(sel)))
    perf_records.append(eval_portfolio(EX.mean(axis=1), 'Balanced Universe (EW)', EX.shape[1]))

    mkt = sub_ff.loc[months, 'MKT_RF']
    perf_records.append({
        'portfolio': 'Market Benchmark (Mkt-RF)',
        'n_names': 1,
        'months': len(mkt),
        'ex_ret_ann': mkt.mean() * 12,
        'vol_ann': mkt.std() * np.sqrt(12),
        'sharpe': mkt.mean() / mkt.std() * np.sqrt(12),
        'ff3_alpha_ann': 0.0,
        't_alpha': np.nan,
        'b_mkt': 1.0,
        's_smb': 0.0,
        'h_hml': 0.0
    })
    perf_df = pd.DataFrame(perf_records)

    # 12. Save Results
    # Format floats cleanly
    float_cols = ['alpha', 'b_mkt', 's_smb', 'h_hml', 't_alpha', 't_mkt', 't_smb', 't_hml', 'r2', 'resid_sd', 'alpha_ann', 'sharpe', 'sharpe_raw', 'appraisal_ratio']
    for c in float_cols:
        if c in res.columns:
            res[c] = res[c].round(4)
        if c in sel.columns:
            sel[c] = sel[c].round(4)

    sel.to_csv(OUTPUT_SELECTED, index=False, encoding='utf-8-sig')
    res.to_csv(OUTPUT_REGRESSION_ALL, index=False, encoding='utf-8-sig')
    perf_df.round(4).to_csv(OUTPUT_PERFORMANCE, index=False, encoding='utf-8-sig')

    # Display Top Results
    print("\n" + "=" * 70)
    print(" Top 20 Ranked Stocks by t(alpha) Descending:")
    print("=" * 70)
    show_cols = ['rank_all', 'Ticker', 'PERMNO', 'name', 't_alpha', 'alpha_ann', 'sharpe', 'b_mkt', 's_smb', 'h_hml', 'r2', 'gate_pass']
    print(res[show_cols].head(20).to_string(index=False))

    print("\n" + "=" * 70)
    print(" In-Sample Portfolio Performance Comparison:")
    print("=" * 70)
    print(perf_df.round(3).to_string(index=False))

    print("\n" + "=" * 70)
    print(f" [Outputs Saved to data/ff3/]")
    print(f" 1. Gate Selected Stocks:       {OUTPUT_SELECTED} ({len(sel)} stocks)")
    print(f" 2. Full Regression & Ranking:  {OUTPUT_REGRESSION_ALL} ({len(res)} stocks)")
    print(f" 3. Portfolio Performance:      {OUTPUT_PERFORMANCE}")
    print(f" 4. Fama-French Factors:        {PATH_FF_FACTORS}")
    print("=" * 70)
    print(f"[Done] FF3 ranking finished in {time.time()-start_time:.2f} seconds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FF3 Security Selection and Ranking.")
    parser.add_argument('--t-cut', type=float, default=2.0, help="t-statistic threshold for alpha gate (default: 2.0).")
    parser.add_argument('--liq-cut', type=float, default=0.20, help="Liquidity bottom cut percentile (default: 0.20).")
    parser.add_argument('--update-factors', action='store_true', help="Force re-download Kenneth French factors.")

    args = parser.parse_args()
    run_ff3_ranking(t_cut=args.t_cut, liq_cut=args.liq_cut, force_factors=args.update_factors)
