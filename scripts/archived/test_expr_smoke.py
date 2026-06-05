"""Quick smoke test for expr_engine."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'factors'))
from expr_engine import ExprEngine

engine = ExprEngine(db_path='data/ohlcv.duckdb', table='daily_ohlcv', code_col='symbol')

# 1. Simple expression
sql = engine.compile_sql('($close - $high) / $close')
print('=== SQL ===')
print(sql)
print()

# 2. Complex expression (Alpha001-like)
sql2 = engine.compile_sql('-1 * Corr(Rank(Delta(Log($volume), 1)), Rank(($close - $open) / $open), 6)')
print('=== Alpha001-like SQL ===')
print(sql2)
print()

# 3. Actually run a simple factor against the DB
df = engine.compute_sql('($close - $high) / $close', start='2024-06-01', end='2024-06-30')
print('=== Factor Result (first 10 rows) ===')
print(df.head(10))
