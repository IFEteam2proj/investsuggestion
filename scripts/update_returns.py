"""
Script to collect and update US Top 80% Market Cap stock returns:
- Past 1-Year Daily Returns
- Past 10-Year Monthly Returns
- Past 20-Year Annual Returns
Features authentic CRSP PERMNO mapping and incremental update caching.
"""

import os
import sys
import json
import time
import argparse
import urllib.request
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
CACHE_DIR = os.path.join(DATA_DIR, '.cache')
PERMNO_CACHE_FILE = os.path.join(CACHE_DIR, 'permno_map.json')
PRICES_CACHE_FILE = os.path.join(CACHE_DIR, 'prices_cache.parquet')
UNIVERSE_CSV_FILE = os.path.join(DATA_DIR, 'stock_universe_top80.csv')

DAILY_RETURNS_CSV = os.path.join(DATA_DIR, 'daily_returns_1y.csv')
MONTHLY_RETURNS_CSV = os.path.join(DATA_DIR, 'monthly_returns_10y.csv')
ANNUAL_RETURNS_CSV = os.path.join(DATA_DIR, 'annual_returns_20y.csv')

DAILY_RETURNS_WIDE_CSV = os.path.join(DATA_DIR, 'daily_returns_1y_wide.csv')
MONTHLY_RETURNS_WIDE_CSV = os.path.join(DATA_DIR, 'monthly_returns_10y_wide.csv')
ANNUAL_RETURNS_WIDE_CSV = os.path.join(DATA_DIR, 'annual_returns_20y_wide.csv')


def ensure_directories():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def load_or_build_permno_map():
    """Load cached ticker->PERMNO map or generate from SEC + CRSP crosswalks."""
    ensure_directories()
    if os.path.exists(PERMNO_CACHE_FILE):
        try:
            with open(PERMNO_CACHE_FILE, 'r', encoding='utf-8') as f:
                permno_map = json.load(f)
            print(f"[PERMNO] Loaded {len(permno_map)} ticker-PERMNO mappings from cache.")
            return permno_map
        except Exception as e:
            print(f"[PERMNO] Error reading cache ({e}), rebuilding...")

    print("[PERMNO] Building crosswalk between CRSP PERMNO and SEC tickers...")
    try:
        url_crsp = 'https://raw.githubusercontent.com/Wenzhi-Ding/Std_Security_Code/main/other/gvkey_permco_permno.pq'
        url_cg = 'https://raw.githubusercontent.com/Wenzhi-Ding/Std_Security_Code/main/other/cik_gvkey.pq'
        url_ct = 'https://raw.githubusercontent.com/Wenzhi-Ding/Std_Security_Code/main/other/cik_ticker.pq'

        df_crsp = pd.read_parquet(url_crsp).dropna(subset=['LPERMNO'])
        df_crsp['GVKEY'] = df_crsp['GVKEY'].astype(int)
        df_crsp['PERMNO'] = df_crsp['LPERMNO'].astype(int)

        df_cg = pd.read_parquet(url_cg).dropna(subset=['gvkey'])
        df_cg['cik'] = df_cg['cik'].astype(int)
        df_cg['gvkey'] = df_cg['gvkey'].astype(int)

        cg_crsp = df_cg.merge(df_crsp, left_on='gvkey', right_on='GVKEY')

        # SEC official company tickers
        req_sec = urllib.request.Request(
            'https://www.sec.gov/files/company_tickers.json',
            headers={'User-Agent': 'InvestmentResearch user@investment.com'}
        )
        with urllib.request.urlopen(req_sec) as resp:
            sec_data = json.loads(resp.read().decode())
        df_sec = pd.DataFrame(list(sec_data.values()))
        df_sec['cik_str'] = df_sec['cik_str'].astype(int)

        sec_mapped = df_sec.merge(cg_crsp, left_on='cik_str', right_on='cik').drop_duplicates(subset=['ticker'])
        map1 = dict(zip(sec_mapped['ticker'], sec_mapped['PERMNO']))

        # Standard cik_ticker table
        df_ct = pd.read_parquet(url_ct).dropna(subset=['Ticker'])
        df_ct['CIK'] = df_ct['CIK'].astype(int)
        ct_mapped = df_ct.merge(cg_crsp, left_on='CIK', right_on='cik').sort_values(
            by=['N_Records', 'LTDate'], ascending=False
        ).drop_duplicates(subset=['Ticker'])
        map2 = dict(zip(ct_mapped['Ticker'], ct_mapped['PERMNO']))

        full_map = {**map2, **map1}
        json_map = {k: int(v) for k, v in full_map.items()}

        with open(PERMNO_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(json_map, f, indent=2)
        print(f"[PERMNO] Saved {len(json_map)} ticker-PERMNO mappings to cache.")
        return json_map
    except Exception as e:
        print(f"[PERMNO] Warning: Failed to build crosswalk ({e}). Proceeding with empty mapping.")
        return {}


def get_top80_universe(top_pct=0.80, permno_map=None):
    """
    Fetch US listed stocks from Nasdaq screener, calculate cumulative market cap,
    and filter stocks representing the top `top_pct` (default 80%) of total market cap.
    """
    ensure_directories()
    print(f"[Universe] Fetching US listed stocks to calculate top {top_pct*100:.0f}% market cap universe...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    url_nasdaq = 'https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000'

    stocks = []
    try:
        req = urllib.request.Request(url_nasdaq, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        rows = data.get('data', {}).get('table', {}).get('rows', [])
        for r in rows:
            sym = r.get('symbol', '').strip()
            mc_str = r.get('marketCap', '').replace(',', '').strip()
            if mc_str and mc_str.isdigit():
                stocks.append({
                    'symbol': sym,
                    'name': r.get('name', '').strip(),
                    'marketCap': int(mc_str),
                    'lastSale': r.get('lastsale', '').strip()
                })
    except Exception as e:
        print(f"[Universe] Warning: Failed to fetch live Nasdaq screener ({e}).")
        if os.path.exists(UNIVERSE_CSV_FILE):
            print("[Universe] Falling back to previously saved universe CSV.")
            return pd.read_csv(UNIVERSE_CSV_FILE)
        else:
            raise RuntimeError("No universe data available. Please check internet connection.")

    df_stocks = pd.DataFrame(stocks).sort_values(by='marketCap', ascending=False).reset_index(drop=True)
    total_mc = df_stocks['marketCap'].sum()
    df_stocks['cum_mc'] = df_stocks['marketCap'].cumsum()
    df_stocks['cum_pct'] = df_stocks['cum_mc'] / total_mc

    # Filter top 80% (include the marginal stock that crosses the threshold)
    top_universe = df_stocks[df_stocks['cum_pct'].shift(1, fill_value=0) < top_pct].copy()

    # Map Yahoo Ticker format (convert dot/slash to hyphen, e.g. BRK.B -> BRK-B)
    top_universe['yahoo_ticker'] = top_universe['symbol'].apply(lambda s: s.replace('.', '-').replace('/', '-'))

    # Map PERMNO
    if permno_map:
        top_universe['PERMNO'] = top_universe['symbol'].map(permno_map)
        # If dot was replaced, try original or mapped
        missing_mask = top_universe['PERMNO'].isna()
        if missing_mask.any():
            top_universe.loc[missing_mask, 'PERMNO'] = top_universe.loc[missing_mask, 'yahoo_ticker'].map(permno_map)

    top_universe['PERMNO'] = pd.to_numeric(top_universe['PERMNO'], errors='coerce').astype('Int64')

    # Format market cap for display
    top_universe['marketCap_Billion'] = (top_universe['marketCap'] / 1e9).round(2)
    top_universe['cum_pct_formatted'] = (top_universe['cum_pct'] * 100).round(2).astype(str) + '%'

    # Save universe CSV
    cols_to_save = ['symbol', 'yahoo_ticker', 'PERMNO', 'name', 'marketCap', 'marketCap_Billion', 'cum_pct', 'cum_pct_formatted']
    top_universe[cols_to_save].to_csv(UNIVERSE_CSV_FILE, index=False, encoding='utf-8-sig')

    matched_permno = top_universe['PERMNO'].notna().sum()
    print(f"[Universe] Filtered {len(top_universe)} stocks accounting for {top_universe.iloc[-1]['cum_pct']*100:.2f}% of total US market cap.")
    print(f"[Universe] Total Market Cap: ${total_mc/1e12:.2f}T, Cutoff: ${top_universe.iloc[-1]['marketCap_Billion']:.2f}B.")
    print(f"[Universe] Authentic CRSP PERMNO matched: {matched_permno} / {len(top_universe)} ({matched_permno/len(top_universe)*100:.1f}%).")
    print(f"[Universe] Saved universe metadata to {UNIVERSE_CSV_FILE}")

    return top_universe


def fetch_and_update_prices(tickers, force=False, batch_size=50):
    """
    Download or incrementally update 20-year daily adjusted close prices for all tickers.
    Cached locally in `data/.cache/prices_cache.parquet`.
    """
    ensure_directories()
    now = datetime.now()
    start_date_20y = (now - timedelta(days=20 * 365 + 30)).strftime('%Y-%m-%d')
    today_str = now.strftime('%Y-%m-%d')

    cached_prices = None
    if os.path.exists(PRICES_CACHE_FILE) and not force:
        try:
            cached_prices = pd.read_parquet(PRICES_CACHE_FILE)
            cached_prices.index = pd.to_datetime(cached_prices.index)
            print(f"[Cache] Found existing price cache with {len(cached_prices.columns)} tickers from {cached_prices.index.min().date()} to {cached_prices.index.max().date()}.")
        except Exception as e:
            print(f"[Cache] Failed to load price cache ({e}), will re-download.")
            cached_prices = None

    if cached_prices is not None:
        last_date = cached_prices.index.max()
        # Check how many days missing
        days_gap = (now.date() - last_date.date()).days
        cached_tickers = set(cached_prices.columns)
        missing_tickers = [t for t in tickers if t not in cached_tickers]

        if days_gap <= 0 and not missing_tickers:
            print("[Update] Cache is completely up to date with the latest market date.")
            return cached_prices

        # Incremental download for existing tickers if there are new trading days
        if days_gap > 0:
            inc_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"[Update] Performing incremental download from {inc_start} to {today_str} for {len(tickers)} tickers...")
            inc_data = download_batch_prices(tickers, start=inc_start, end=today_str, batch_size=batch_size)
            if not inc_data.empty:
                # Merge with cached prices
                combined = pd.concat([cached_prices, inc_data])
                combined = combined[~combined.index.duplicated(keep='last')].sort_index()
                cached_prices = combined

        # If there are brand new tickers not in cache, download their full 20 years
        if missing_tickers:
            print(f"[Update] Downloading 20-year history for {len(missing_tickers)} new tickers in universe...")
            new_data = download_batch_prices(missing_tickers, start=start_date_20y, end=today_str, batch_size=batch_size)
            if not new_data.empty:
                cached_prices = pd.concat([cached_prices, new_data], axis=1)

        # De-duplicate columns if any overlap occurred
        cached_prices = cached_prices.loc[:, ~cached_prices.columns.duplicated(keep='last')]

        # Save updated cache
        cached_prices = cached_prices.sort_index()
        cached_prices.to_parquet(PRICES_CACHE_FILE)
        print(f"[Cache] Updated cache saved to {PRICES_CACHE_FILE} ({len(cached_prices)} days, {len(cached_prices.columns)} tickers).")
        return cached_prices

    else:
        # Full initial download
        print(f"[Download] Downloading 20-year history for all {len(tickers)} tickers (from {start_date_20y} to {today_str})...")
        prices_df = download_batch_prices(tickers, start=start_date_20y, end=today_str, batch_size=batch_size)
        prices_df = prices_df.sort_index()
        prices_df.to_parquet(PRICES_CACHE_FILE)
        print(f"[Cache] Initial cache saved to {PRICES_CACHE_FILE} ({len(prices_df)} days, {len(prices_df.columns)} tickers).")
        return prices_df


def download_batch_prices(tickers, start, end, batch_size=50):
    """Batch download adjusted close prices using yfinance with retry."""
    all_series = []
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in range(0, len(tickers), batch_size):
        chunk = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  -> Downloading batch {batch_num}/{total_batches} ({len(chunk)} tickers)...", end="", flush=True)
        t0 = time.time()
        try:
            df = yf.download(
                chunk,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True
            )
            elapsed = time.time() - t0
            if df.empty:
                print(f" Empty response ({elapsed:.1f}s)")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                close = df['Close'].copy()
            else:
                close = df[['Close']].copy()
                close.columns = chunk

            all_series.append(close)
            print(f" Done ({elapsed:.1f}s, {len(close)} rows)")
        except Exception as e:
            print(f" Error: {e}")

    if not all_series:
        return pd.DataFrame()

    combined = pd.concat(all_series, axis=1)
    # Remove duplicate columns if any
    combined = combined.loc[:, ~combined.columns.duplicated(keep='first')]
    return combined


def calculate_returns(prices_df, universe_df):
    """
    Compute rolling returns:
    1) 1-Year Daily Returns
    2) 10-Year Monthly Returns (month-end to month-end)
    3) 20-Year Annual Returns (year-end to year-end)
    """
    print("[Returns] Calculating returns for 1Y daily, 10Y monthly, and 20Y annual...")
    prices = prices_df.copy()
    latest_date = prices.index.max()

    ticker_to_symbol = dict(zip(universe_df['yahoo_ticker'], universe_df['symbol']))
    symbol_to_permno = dict(zip(universe_df['symbol'], universe_df['PERMNO']))

    # 1. Past 1-Year Daily Returns
    cutoff_1y = latest_date - pd.DateOffset(years=1)
    df_1y = prices.loc[prices.index >= cutoff_1y]
    daily_returns_wide = df_1y.pct_change().dropna(how='all')

    # 2. Past 10-Year Monthly Returns
    cutoff_10y = latest_date - pd.DateOffset(years=10)
    df_10y = prices.loc[prices.index >= cutoff_10y]
    # Resample to month-end business days (last price of month)
    monthly_prices_10y = df_10y.resample('ME').last()
    monthly_returns_wide = monthly_prices_10y.pct_change().dropna(how='all')

    # 3. Past 20-Year Annual Returns
    cutoff_20y = latest_date - pd.DateOffset(years=20)
    df_20y = prices.loc[prices.index >= cutoff_20y]
    # Resample to year-end business days (last price of year)
    annual_prices_20y = df_20y.resample('YE').last()
    annual_returns_wide = annual_prices_20y.pct_change().dropna(how='all')

    return (
        (daily_returns_wide, df_1y),
        (monthly_returns_wide, monthly_prices_10y),
        (annual_returns_wide, annual_prices_20y),
        ticker_to_symbol,
        symbol_to_permno
    )


def build_long_panel(returns_wide, prices_wide, ticker_to_symbol, symbol_to_permno, freq_type='daily'):
    """
    Transform wide matrix (Date x Ticker) into standardized Long Panel format:
    Columns: Date / Year, PERMNO, Ticker, Return, AdjClose
    """
    # Melt returns
    df_ret = returns_wide.reset_index().rename(columns={'index': 'Date', 'Date': 'Date'})
    melted_ret = df_ret.melt(id_vars='Date', var_name='YahooTicker', value_name='Return')

    # Melt prices
    df_prc = prices_wide.reset_index().rename(columns={'index': 'Date', 'Date': 'Date'})
    melted_prc = df_prc.melt(id_vars='Date', var_name='YahooTicker', value_name='AdjClose')

    # Merge
    panel = pd.merge(melted_ret, melted_prc, on=['Date', 'YahooTicker'], how='left')
    panel = panel.dropna(subset=['Return']).copy()

    # Map standard symbol and PERMNO
    panel['Ticker'] = panel['YahooTicker'].map(ticker_to_symbol).fillna(panel['YahooTicker'])
    panel['PERMNO'] = pd.to_numeric(panel['Ticker'].map(symbol_to_permno), errors='coerce').astype('Int64')

    if freq_type == 'daily':
        panel['Date'] = panel['Date'].dt.strftime('%Y-%m-%d')
        cols = ['Date', 'PERMNO', 'Ticker', 'Return', 'AdjClose']
        panel = panel.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
    elif freq_type == 'monthly':
        panel['YearMonth'] = panel['Date'].dt.strftime('%Y-%m')
        panel['Date'] = panel['Date'].dt.strftime('%Y-%m-%d')
        cols = ['Date', 'YearMonth', 'PERMNO', 'Ticker', 'Return', 'AdjClose']
        panel = panel.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
    elif freq_type == 'annual':
        panel['Year'] = panel['Date'].dt.strftime('%Y')
        panel['Date'] = panel['Date'].dt.strftime('%Y-%m-%d')
        cols = ['Year', 'Date', 'PERMNO', 'Ticker', 'Return', 'AdjClose']
        panel = panel.sort_values(by=['Ticker', 'Year']).reset_index(drop=True)

    # Format numeric values cleanly
    panel['Return'] = panel['Return'].round(6)
    panel['AdjClose'] = panel['AdjClose'].round(4)

    return panel[cols]


def export_csv_files(daily_tuple, monthly_tuple, annual_tuple, ticker_to_symbol, symbol_to_permno):
    """Export long panel and wide matrix CSV files to data directory."""
    daily_returns_wide, daily_prices = daily_tuple
    monthly_returns_wide, monthly_prices = monthly_tuple
    annual_returns_wide, annual_prices = annual_tuple

    print("[Export] Building long-format panel datasets...")
    daily_panel = build_long_panel(daily_returns_wide, daily_prices, ticker_to_symbol, symbol_to_permno, 'daily')
    monthly_panel = build_long_panel(monthly_returns_wide, monthly_prices, ticker_to_symbol, symbol_to_permno, 'monthly')
    annual_panel = build_long_panel(annual_returns_wide, annual_prices, ticker_to_symbol, symbol_to_permno, 'annual')

    # Save Long format CSVs
    daily_panel.to_csv(DAILY_RETURNS_CSV, index=False, encoding='utf-8-sig')
    monthly_panel.to_csv(MONTHLY_RETURNS_CSV, index=False, encoding='utf-8-sig')
    annual_panel.to_csv(ANNUAL_RETURNS_CSV, index=False, encoding='utf-8-sig')

    # Save Wide format CSVs for matrix computations
    # Rename columns to standard symbols
    daily_wide_renamed = daily_returns_wide.rename(columns=ticker_to_symbol).round(6)
    monthly_wide_renamed = monthly_returns_wide.rename(columns=ticker_to_symbol).round(6)
    annual_wide_renamed = annual_returns_wide.rename(columns=ticker_to_symbol).round(6)

    daily_wide_renamed.to_csv(DAILY_RETURNS_WIDE_CSV, index=True, encoding='utf-8-sig')
    monthly_wide_renamed.to_csv(MONTHLY_RETURNS_WIDE_CSV, index=True, encoding='utf-8-sig')
    annual_wide_renamed.to_csv(ANNUAL_RETURNS_WIDE_CSV, index=True, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print(" [Data Generation Complete] Summary of Generated CSV Files:")
    print("=" * 60)
    print(f" 1. 1Y Daily Returns (Panel):   {DAILY_RETURNS_CSV} ({len(daily_panel):,} rows)")
    print(f" 2. 10Y Monthly Returns (Panel): {MONTHLY_RETURNS_CSV} ({len(monthly_panel):,} rows)")
    print(f" 3. 20Y Annual Returns (Panel):  {ANNUAL_RETURNS_CSV} ({len(annual_panel):,} rows)")
    print(f" 4. 1Y Daily Returns (Wide):    {DAILY_RETURNS_WIDE_CSV}")
    print(f" 5. 10Y Monthly Returns (Wide):  {MONTHLY_RETURNS_WIDE_CSV}")
    print(f" 6. 20Y Annual Returns (Wide):   {ANNUAL_RETURNS_WIDE_CSV}")
    print(f" 7. Stock Universe Metadata:    {UNIVERSE_CSV_FILE}")
    print("=" * 60)


def run_pipeline(force=False, top_pct=0.80, batch_size=50):
    start_time = time.time()
    print("============================================================")
    print(" Starting US Top 80% Stock Return Collection & Update Pipeline")
    print(f" Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Mode: {'Full Re-download (--force)' if force else 'Incremental Update'}")
    print("============================================================")

    # 1. PERMNO Map
    permno_map = load_or_build_permno_map()

    # 2. Universe
    universe_df = get_top80_universe(top_pct=top_pct, permno_map=permno_map)
    yahoo_tickers = universe_df['yahoo_ticker'].tolist()

    # 3. Prices
    prices_df = fetch_and_update_prices(yahoo_tickers, force=force, batch_size=batch_size)

    # 4. Returns
    daily_tup, monthly_tup, annual_tup, t_to_s, s_to_p = calculate_returns(prices_df, universe_df)

    # 5. Export
    export_csv_files(daily_tup, monthly_tup, annual_tup, t_to_s, s_to_p)

    total_time = time.time() - start_time
    print(f"\n[Success] Entire pipeline completed in {total_time:.2f} seconds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect and update US Top 80% Market Cap stock returns (1Y daily, 10Y monthly, 20Y annual).")
    parser.add_argument('--force', action='store_true', help="Force full 20-year re-download instead of incremental update.")
    parser.add_argument('--top-pct', type=float, default=0.80, help="Cumulative market cap fraction (default: 0.80 for top 80%%).")
    parser.add_argument('--batch-size', type=int, default=50, help="Batch size for Yahoo Finance download (default: 50).")

    args = parser.parse_args()
    run_pipeline(force=args.force, top_pct=args.top_pct, batch_size=args.batch_size)
