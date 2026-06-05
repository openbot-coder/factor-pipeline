"""
Download OHLCV data for CSI 500 constituents via Tencent API → DuckDB.
"""
import time, sys, random
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import akshare as ak
import duckdb

def to_tx_symbol(code: str) -> str:
    code = str(code).zfill(6)
    prefix = 'sz' if code.startswith(('0', '3', '8', '4')) else 'sh'
    return prefix + code


def fetch_stock(code: str) -> pd.DataFrame:
    symbol = to_tx_symbol(code)
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_hist_tx(
                symbol=symbol,
                start_date='20210101',
                end_date='20241231',
                adjust='qfq'
            )
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.rename(columns={
                'date': 'date', 'open': 'open', 'close': 'close',
                'high': 'high', 'low': 'low', 'amount': 'amount',
            })
            # Compute volume from amount / close (rough proxy)
            if 'volume' not in df.columns:
                df['volume'] = (df['amount'] / df['close'] * 1e8).round(0)
            # Compute turnover_rate roughly
            if 'turnover_rate' not in df.columns:
                df['turnover_rate'] = 0.0

            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            # Symbol = original code (without prefix)
            df['symbol'] = code.zfill(6)
            df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover_rate']]
            return df
        except Exception as e:
            if attempt < 2:
                time.sleep(random.uniform(0.5, 1.5) * (attempt + 1))
    return pd.DataFrame()


def main():
    print('Fetching CSI 500 constituents...')
    cons_df = ak.index_stock_cons_weight_csindex(symbol='000905')
    cons_df = cons_df[cons_df['日期'] == cons_df['日期'].max()]
    codes = cons_df['成分券代码'].tolist()
    # Limit to first 100 for faster testing
    codes = codes[:100]
    print(f'Got {len(codes)} constituents (limited from {len(cons_df)})')

    all_data = []
    total = len(codes)
    failed = 0

    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {pool.submit(fetch_stock, c): c for c in codes}
        done = 0
        for fut in as_completed(futures):
            df = fut.result()
            if not df.empty:
                all_data.append(df)
            else:
                failed += 1
            done += 1
            bar = '=' * (done * 50 // total)
            sys.stdout.write(f'\r[{bar:<50}] {done}/{total} (failed: {failed})')
            sys.stdout.flush()

    print(f'\nFetched {len(all_data)} stocks, {failed} failed')

    if not all_data:
        print('No data fetched!')
        return

    combined = pd.concat(all_data, ignore_index=True)
    print(f'Total rows: {len(combined):,}')
    print(f'Stocks: {combined["symbol"].nunique()}, Days: {combined["date"].nunique()}')

    # Save intermediate CSV
    combined.to_csv('data/csi500_raw.csv', index=False)
    print('Saved intermediate CSV')

    # Store in DuckDB
    con = duckdb.connect('data/ohlcv_csi500.duckdb')
    con.execute('DROP TABLE IF EXISTS daily_ohlcv')
    con.execute('''
        CREATE TABLE daily_ohlcv (
            date DATE, symbol VARCHAR,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume DOUBLE, amount DOUBLE, turnover_rate DOUBLE
        )
    ''')
    con.execute("CREATE INDEX idx_sym ON daily_ohlcv (symbol, date)")
    con.execute("CREATE INDEX idx_dt ON daily_ohlcv (date)")
    con.execute("INSERT INTO daily_ohlcv BY NAME SELECT * FROM combined")
    con.close()
    print('Saved to data/ohlcv_csi500.duckdb')


if __name__ == '__main__':
    main()
