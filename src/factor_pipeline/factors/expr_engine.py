"""Qlib-style Expression Engine — parse factor expressions, compile to DuckDB SQL or Pandas.

Supports Qlib expression syntax:
    $close, $high, $low, $open, $volume, $amount
    Ref($close, 1), Mean($close, 20), Std($close, 60), Sum($volume, 5)
    Delta($close, 1), Max($high, 20), Min($low, 20)
    Rank($close), Ts_Rank($close, 10), Corr($high, $volume, 20)
    Log($close), Abs(x), Iif(cond, A, B)

Usage:
    from expr_engine import ExprEngine

    engine = ExprEngine(db_path="data/ohlcv.duckdb")

    # DuckDB SQL
    sql = engine.compile_sql("($close - Ref($close, 1)) / Ref($close, 1)")

    # Pandas DataFrame
    df = engine.compute_pandas("($close - Ref($close, 1)) / Ref($close, 1)")

    # One-liner: expression → factor DataFrame
    df = engine.factor("($close - $high) / $close", name="my_factor")

    # CLI
    python expr_engine.py "($close - Ref($close, 1)) / Ref($close, 1)" --backend duckdb
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

import numpy as np
import pandas as pd


# =============================================================================
# 1. Tokenizer
# =============================================================================

class TT(Enum):
    NUMBER = auto()
    IDENT = auto()
    DOLLAR_VAR = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    GT = auto()
    LT = auto()
    GTE = auto()
    LTE = auto()
    EQ = auto()
    NEQ = auto()
    EOF = auto()


@dataclass
class Token:
    tt: TT
    value: str
    pos: int = 0


QLIB_COLUMNS = {"open", "high", "low", "close", "volume", "amount"}


def tokenize(expr: str) -> list[Token]:
    """Tokenize a Qlib expression string."""
    tokens: list[Token] = []
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch == '$':
            start = i + 1
            i = start  # advance past '$' before reading name
            while i < n and (expr[i].isalnum() or expr[i] == '_'):
                i += 1
            var_name = expr[start:i]
            if var_name.lower() not in QLIB_COLUMNS:
                raise SyntaxError(f"Unknown $ variable: ${var_name} at pos {start}")
            tokens.append(Token(TT.DOLLAR_VAR, var_name.lower(), start))
            continue
        if ch.isdigit() or (ch == '.' and i + 1 < n and expr[i + 1].isdigit()):
            start = i
            has_dot = ch == '.'
            i += 1
            while i < n and (expr[i].isdigit() or (expr[i] == '.' and not has_dot)):
                if expr[i] == '.':
                    has_dot = True
                i += 1
            tokens.append(Token(TT.NUMBER, expr[start:i], start))
            continue
        if ch.isalpha() or ch == '_':
            start = i
            while i < n and (expr[i].isalnum() or expr[i] == '_'):
                i += 1
            word = expr[start:i]
            tokens.append(Token(TT.IDENT, word, start))
            continue
        # Two-char comparison operators first
        if ch == '>' and i + 1 < n and expr[i + 1] == '=':
            tokens.append(Token(TT.GTE, '>=', i)); i += 2; continue
        if ch == '<' and i + 1 < n and expr[i + 1] == '=':
            tokens.append(Token(TT.LTE, '<=', i)); i += 2; continue
        if ch == '!' and i + 1 < n and expr[i + 1] == '=':
            tokens.append(Token(TT.NEQ, '!=', i)); i += 2; continue
        if ch == '=' and i + 1 < n and expr[i + 1] == '=':
            tokens.append(Token(TT.EQ, '==', i)); i += 2; continue
        # Single-char ops
        _SIMPLE = {'+': TT.PLUS, '-': TT.MINUS, '*': TT.STAR, '/': TT.SLASH,
                    '(': TT.LPAREN, ')': TT.RPAREN, ',': TT.COMMA,
                    '>': TT.GT, '<': TT.LT}
        if ch in _SIMPLE:
            tokens.append(Token(_SIMPLE[ch], ch, i)); i += 1; continue
        raise SyntaxError(f"Unexpected character '{ch}' at position {i}")
    tokens.append(Token(TT.EOF, '', len(expr)))
    return tokens


# =============================================================================
# 2. AST Nodes
# =============================================================================

class NodeKind(Enum):
    COLUMN = auto()
    LITERAL = auto()
    BINARY = auto()
    UNARY = auto()
    FUNC = auto()
    CMP = auto()


@dataclass
class ASTNode:
    kind: NodeKind
    value: Any = None
    children: list['ASTNode'] = field(default_factory=list)

    def __repr__(self):
        if self.kind == NodeKind.COLUMN:
            return f"${self.value}"
        if self.kind == NodeKind.LITERAL:
            return str(self.value)
        if self.kind == NodeKind.UNARY:
            return f"(-{self.children[0]})"
        if self.kind == NodeKind.BINARY:
            return f"({self.children[0]} {self.value} {self.children[1]})"
        if self.kind == NodeKind.FUNC:
            args = ", ".join(repr(c) for c in self.children)
            return f"{self.value}({args})"
        if self.kind == NodeKind.CMP:
            return f"({self.children[0]} {self.value} {self.children[1]})"
        return f"Node({self.kind}, {self.value})"


# =============================================================================
# 3. Parser (recursive descent)
# =============================================================================

class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, tt: TT) -> Token:
        tok = self.peek()
        if tok.tt != tt:
            raise SyntaxError(
                f"Expected {tt.name} but got {tok.tt.name} ('{tok.value}') at pos {tok.pos}"
            )
        return self.advance()

    def parse(self) -> ASTNode:
        node = self.comparison()
        if self.peek().tt != TT.EOF:
            raise SyntaxError(f"Unexpected token: {self.peek()}")
        return node

    def comparison(self) -> ASTNode:
        left = self.addition()
        if self.peek().tt in (TT.GT, TT.LT, TT.GTE, TT.LTE, TT.EQ, TT.NEQ):
            op = self.advance()
            right = self.addition()
            return ASTNode(NodeKind.CMP, op.value, [left, right])
        return left

    def addition(self) -> ASTNode:
        left = self.multiplication()
        while self.peek().tt in (TT.PLUS, TT.MINUS):
            op = self.advance()
            right = self.multiplication()
            left = ASTNode(NodeKind.BINARY, op.value, [left, right])
        return left

    def multiplication(self) -> ASTNode:
        left = self.unary()
        while self.peek().tt in (TT.STAR, TT.SLASH):
            op = self.advance()
            right = self.unary()
            left = ASTNode(NodeKind.BINARY, op.value, [left, right])
        return left

    def unary(self) -> ASTNode:
        if self.peek().tt == TT.MINUS:
            self.advance()
            return ASTNode(NodeKind.UNARY, '-', [self.unary()])
        if self.peek().tt == TT.PLUS:
            self.advance()
            return self.unary()
        return self.primary()

    def primary(self) -> ASTNode:
        tok = self.peek()
        if tok.tt == TT.NUMBER:
            self.advance()
            val = float(tok.value)
            if val == int(val):
                val = int(val)
            return ASTNode(NodeKind.LITERAL, val)
        if tok.tt == TT.DOLLAR_VAR:
            self.advance()
            return ASTNode(NodeKind.COLUMN, tok.value)
        if tok.tt == TT.LPAREN:
            self.advance()
            node = self.comparison()
            self.expect(TT.RPAREN)
            return node
        if tok.tt == TT.IDENT:
            func_name = self.advance().value
            self.expect(TT.LPAREN)
            args: list[ASTNode] = []
            if self.peek().tt != TT.RPAREN:
                args.append(self.comparison())
                while self.peek().tt == TT.COMMA:
                    self.advance()
                    args.append(self.comparison())
            self.expect(TT.RPAREN)
            return ASTNode(NodeKind.FUNC, func_name, args)
        raise SyntaxError(f"Unexpected token {tok.tt.name} ('{tok.value}') at pos {tok.pos}")


def parse(expr: str) -> ASTNode:
    """Parse a Qlib expression string into an AST."""
    tokens = tokenize(expr)
    return Parser(tokens).parse()


# =============================================================================
# 4. DuckDB SQL Compiler
# =============================================================================

_SQL_CMPOP = {'>': '>', '<': '<', '>=': '>=', '<=': '<=', '==': '=', '!=': '!='}


class SQLCompiler:
    """Compile AST to DuckDB SQL with automatic CTE decomposition.

    Args:
        universe: Stock pool filter. Three modes:
            - None: no filter (all stocks in table)
            - Pool ID string (e.g. 'csi500', 'csi300', 'all_a_share'):
              auto-generate SQL to filter via instruments table
            - Raw SQL subquery returning (date, symbol) columns:
              used as INNER JOIN filter
        instruments_db: Path to DuckDB database containing instruments table.
            Required when universe is a pool ID string. E.g. '/path/to/quantdb.duckdb'
    """

    def __init__(self, table: str = "daily_kline", date_col: str = "date",
                 code_col: str = "code", universe: str | None = None,
                 instruments_db: str | None = None):
        self.table = table
        self.date_col = date_col
        self.code_col = code_col
        self.universe = universe
        self.instruments_db = instruments_db
        self._ctes: list[str] = []
        self._cte_counter = 0
        self._attach_done = False

    def _new_cte(self, select_body: str) -> str:
        self._cte_counter += 1
        name = f"step{self._cte_counter}"
        self._ctes.append(f"{name} AS (\n{select_body}\n)")
        return name

    def _base_cte(self, start: str, end: str) -> str:
        # Universe filter: append to WHERE clause if specified
        universe_filter = ""
        if self.universe:
            universe_filter = f"\n    AND {self.code_col} IN (\n{self._universe_sql()}\n    )"

        return f"""base AS (
    SELECT
        {self.date_col}, {self.code_col},
        COALESCE(open, 0) AS open, COALESCE(high, 0) AS high,
        COALESCE(low, 0) AS low, COALESCE(close, 0) AS close,
        COALESCE(volume, 0) AS volume, COALESCE(amount, 0) AS amount
    FROM {self.table}
    WHERE {self.date_col} >= '{start}' AND {self.date_col} <= '{end}' AND volume > 0{universe_filter}
)"""

    def _universe_sql(self) -> str:
        """Generate SQL subquery for universe filtering."""
        u = self.universe
        if not u:
            return ""

        # If it looks like a raw SQL query (contains SELECT), use as-is
        if u.upper().startswith("SELECT"):
            return u

        # Otherwise treat as pool_id → query instruments table
        # Reference attached database if instruments_db was provided
        db_prefix = "instruments_db." if self.instruments_db else ""
        return f"""SELECT stock_code FROM {db_prefix}instruments
        WHERE pool_id = '{u}'
          AND {self.date_col} >= entry_date
          AND (exit_date IS NULL OR {self.date_col} <= exit_date)"""

    def _over(self, func: str, window: int | None = None) -> str:
        over = f" OVER (PARTITION BY {self.code_col} ORDER BY {self.date_col}"
        if window and window > 1:
            over += f" ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        return func + over + ")"

    def _col_ref(self, node: ASTNode) -> str:
        if node.kind == NodeKind.COLUMN:
            return node.value
        raise ValueError(f"Expected column node, got {node.kind}")

    def _ensure_column(self, node: ASTNode) -> tuple[str, str]:
        """Ensure arg is a column reference. If complex (contains window func), create CTE.

        Returns (sql_expr, alias) where alias is the column name to reference.
        """
        if node.kind == NodeKind.COLUMN:
            return (node.value, node.value)
        # Check if the SQL contains OVER (window function) — can't nest in DuckDB
        sql_expr = self._sql(node)
        has_window = "OVER" in sql_expr.upper()
        alias = f"_col{self._cte_counter + 1}"
        from_cte = f"step{self._cte_counter}" if self._cte_counter > 0 else "base"
        if has_window:
            # Must create CTE to materialize the window expression
            self._new_cte(
                f"SELECT {self.date_col}, {self.code_col},\n"
                f"       ({sql_expr}) AS {alias}\n"
                f"    FROM {from_cte}"
            )
            return (alias, alias)
        else:
            # Simple scalar — can inline
            return (sql_expr, alias)

    def _sql(self, node: ASTNode) -> str:
        """Compile an AST node to a SQL expression fragment."""
        if node.kind == NodeKind.COLUMN:
            return node.value
        if node.kind == NodeKind.LITERAL:
            return str(node.value)
        if node.kind == NodeKind.UNARY:
            return f"(-({self._sql(node.children[0])}))"
        if node.kind == NodeKind.BINARY:
            return f"({self._sql(node.children[0])} {node.value} {self._sql(node.children[1])})"
        if node.kind == NodeKind.CMP:
            return f"({self._sql(node.children[0])} {SQL_CMPOP[node.value]} {self._sql(node.children[1])})"
        if node.kind == NodeKind.FUNC:
            return self._sql_func(node)
        raise ValueError(f"Unknown node: {node.kind}")

    def _sql_func(self, node: ASTNode) -> str:
        name = node.value.upper()
        args = node.children

        # --- Scalar ---
        if name == "LOG":
            return f"LN({self._sql(args[0])})"
        if name == "ABS":
            return f"ABS({self._sql(args[0])})"
        if name == "SIGN":
            return f"SIGN({self._sql(args[0])})"
        if name == "SQRT":
            return f"SQRT({self._sql(args[0])})"
        if name == "POWER":
            return f"POWER({self._sql(args[0])}, {self._sql(args[1])})"

        # --- Ref / Delay ---
        if name in ("REF", "DELAY"):
            col_sql, col_alias = self._ensure_column(args[0])
            n = int(args[1].value) if len(args) > 1 else 1
            func = f"LEAD({col_alias}, {abs(n)})" if n < 0 else f"LAG({col_alias}, {n})"
            return self._over(func)

        # --- Rolling window functions ---
        if name in ("MEAN", "MA", "TS_MEAN"):
            col_sql, col_alias = self._ensure_column(args[0])
            w = int(args[1].value) if len(args) > 1 else 20
            return self._over(f"AVG({col_alias})", w)
        if name in ("STD", "TS_STD", "STDDEV"):
            col_sql, col_alias = self._ensure_column(args[0])
            w = int(args[1].value) if len(args) > 1 else 20
            return self._over(f"STDDEV_SAMP({col_alias})", w)
        if name in ("SUM", "TS_SUM"):
            col_sql, col_alias = self._ensure_column(args[0])
            w = int(args[1].value) if len(args) > 1 else 20
            return self._over(f"SUM({col_alias})", w)
        if name in ("MAX", "TS_MAX"):
            col_sql, col_alias = self._ensure_column(args[0])
            w = int(args[1].value) if len(args) > 1 else 20
            return self._over(f"MAX({col_alias})", w)
        if name in ("MIN", "TS_MIN"):
            col_sql, col_alias = self._ensure_column(args[0])
            w = int(args[1].value) if len(args) > 1 else 20
            return self._over(f"MIN({col_alias})", w)

        # --- Delta (needs CTE: col - LAG(col, n)) ---
        if name == "DELTA":
            col_sql, col_alias = self._ensure_column(args[0])
            n = int(args[1].value) if len(args) > 1 else 1
            from_cte = f"step{self._cte_counter}" if self._cte_counter > 0 else "base"
            lag_alias = f"_lag_{col_alias}_{n}"
            self._new_cte(
                f"SELECT {self.date_col}, {self.code_col}, {col_alias},\n"
                f"       LAG({col_alias}, {n}) OVER (PARTITION BY {self.code_col} "
                f"ORDER BY {self.date_col}) AS {lag_alias}\n"
                f"    FROM {from_cte}"
            )
            return f"({col_alias} - {lag_alias})"

        # --- Rank ---
        if name in ("RANK", "CSRANK"):
            col_sql, col_alias = self._ensure_column(args[0])
            if col_sql != col_alias:
                # Complex expression — materialize via CTE
                from_cte = f"step{self._cte_counter}" if self._cte_counter > 0 else "base"
                self._new_cte(
                    f"SELECT {self.date_col}, {self.code_col},\n"
                    f"       ({col_sql}) AS {col_alias}\n"
                    f"    FROM {from_cte}"
                )
            return f"PERCENT_RANK() OVER (PARTITION BY {self.date_col} ORDER BY {col_alias})"

        # --- Ts_Rank (rolling percent rank — needs CTE) ---
        if name == "TS_RANK":
            col_sql, col_alias = self._ensure_column(args[0])
            w = int(args[1].value) if len(args) > 1 else 10
            from_cte = f"step{self._cte_counter}" if self._cte_counter > 0 else "base"
            rank_alias = f"_tsrank_{col_alias}"
            self._new_cte(
                f"SELECT {self.date_col}, {self.code_col}, {col_sql} AS {col_alias},\n"
                f"       PERCENT_RANK() OVER (\n"
                f"           PARTITION BY {self.code_col} ORDER BY {col_alias}\n"
                f"           ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW\n"
                f"       ) AS {rank_alias}\n"
                f"    FROM {from_cte}"
            )
            return rank_alias

        # --- Corr ---
        if name == "CORR":
            c1_sql, c1_alias = self._ensure_column(args[0])
            c2_sql, c2_alias = self._ensure_column(args[1])
            w = int(args[2].value) if len(args) > 2 else 20
            # If either arg is complex, need CTE to materialize
            if c1_sql != c1_alias or c2_sql != c2_alias:
                from_cte = f"step{self._cte_counter}" if self._cte_counter > 0 else "base"
                cols = f"{c1_sql} AS {c1_alias}, {c2_sql} AS {c2_alias}"
                self._new_cte(
                    f"SELECT {self.date_col}, {self.code_col}, {cols}\n"
                    f"    FROM {from_cte}"
                )
            return self._over(f"CORR({c1_alias}, {c2_alias})", w)

        # --- Cov ---
        if name == "COV":
            c1_sql, c1_alias = self._ensure_column(args[0])
            c2_sql, c2_alias = self._ensure_column(args[1])
            w = int(args[2].value) if len(args) > 2 else 20
            if c1_sql != c1_alias or c2_sql != c2_alias:
                from_cte = f"step{self._cte_counter}" if self._cte_counter > 0 else "base"
                cols = f"{c1_sql} AS {c1_alias}, {c2_sql} AS {c2_alias}"
                self._new_cte(
                    f"SELECT {self.date_col}, {self.code_col}, {cols}\n"
                    f"    FROM {from_cte}"
                )
            return self._over(f"COVAR_SAMP({c1}, {c2})", w)

        # --- Iif / If ---
        if name in ("IIF", "IF"):
            cond = self._sql(args[0])
            t = self._sql(args[1])
            f = self._sql(args[2]) if len(args) > 2 else "NULL"
            return f"CASE WHEN {cond} THEN {t} ELSE {f} END"

        # --- Count ---
        if name in ("COUNT", "TS_COUNT"):
            col = self._col_ref(args[0])
            w = int(args[1].value) if len(args) > 1 else 20
            return self._over(f"COUNT({col})", w)

        raise ValueError(f"Unknown function: {name}")

    def compile(self, ast: ASTNode, start: str = "2020-01-01",
                end: str = "2026-12-31") -> str:
        """Compile AST to full DuckDB SQL query."""
        self._ctes = []
        self._cte_counter = 0

        factor_expr = self._sql(ast)
        base = self._base_cte(start, end)

        # Check if factor contains window function (has OVER keyword)
        has_window = "OVER(" in factor_expr.upper() or "OVER (" in factor_expr.upper()

        if has_window:
            # Window functions can't be in WHERE clause directly
            # Add intermediate CTE to compute the factor, then filter in final SELECT
            prev_cte = f"step{self._cte_counter}" if self._cte_counter > 0 else "base"
            factor_cte = self._new_cte(
                f"SELECT {self.date_col}, {self.code_col},\n"
                f"       ({factor_expr}) AS _factor\n"
                f"FROM {prev_cte}"
            )
            cte_block = ",\n".join([base] + self._ctes)
            return f"""{self._attach_prefix()}WITH
{cte_block}
SELECT
    {self.date_col}, {self.code_col},
    _factor AS factor
FROM {factor_cte}
WHERE _factor IS NOT NULL
ORDER BY {self.date_col}, {self.code_col}"""
        else:
            cte_block = ",\n".join([base] + self._ctes)
            from_cte = f"step{self._cte_counter}" if self._cte_counter > 0 else "base"
            return f"""{self._attach_prefix()}WITH
{cte_block}
SELECT
    {self.date_col}, {self.code_col},
    ({factor_expr}) AS factor
FROM {from_cte}
WHERE ({factor_expr}) IS NOT NULL
ORDER BY {self.date_col}, {self.code_col}"""

    def _attach_prefix(self) -> str:
        """Generate ATTACH statement if needed for universe filter."""
        if self.universe and self.instruments_db and not self.universe.upper().startswith("SELECT"):
            return f"ATTACH '{self.instruments_db}' AS instruments_db;\n"
        return ""


SQL_CMPOP = _SQL_CMPOP  # expose for _sql in CMP branch


def compile_sql(expr: str, table: str = "daily_ohlcv", date_col: str = "date",
                code_col: str = "symbol", start: str = "2020-01-01",
                end: str = "2026-12-31", universe: str | None = None,
                instruments_db: str | None = None) -> str:
    """Compile a Qlib expression string to DuckDB SQL.

    Args:
        universe: Stock pool filter. None for no filter, pool_id string (e.g. 'csi500'),
                  or raw SQL subquery returning (date, symbol).
        instruments_db: Path to DuckDB with instruments table. Required when universe is pool_id.
    """
    ast = parse(expr)
    return SQLCompiler(table=table, date_col=date_col, code_col=code_col,
                       universe=universe, instruments_db=instruments_db).compile(ast, start, end)


# =============================================================================
# 5. Pandas Compiler
# =============================================================================

class PandasCompiler:
    """Compile AST to a pandas Series on a multi-stock DataFrame."""

    def __init__(self, df: pd.DataFrame, date_col: str = "date",
                 code_col: str = "code"):
        self.df = df.sort_values([date_col, code_col]).reset_index(drop=True)
        self.date_col = date_col
        self.code_col = code_col

    def _grp(self):
        return self.df.groupby(self.code_col, sort=False)

    def _pd(self, node: ASTNode) -> pd.Series:
        """Compile AST node to a pandas Series."""
        if node.kind == NodeKind.COLUMN:
            col = node.value
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not in DataFrame")
            return self.df[col].astype(float)

        if node.kind == NodeKind.LITERAL:
            return pd.Series(node.value, index=self.df.index, dtype=float)

        if node.kind == NodeKind.UNARY:
            return -self._pd(node.children[0])

        if node.kind == NodeKind.BINARY:
            l, r = self._pd(node.children[0]), self._pd(node.children[1])
            ops = {'+': lambda a, b: a + b, '-': lambda a, b: a - b,
                   '*': lambda a, b: a * b, '/': lambda a, b: a / b.replace(0, np.nan)}
            return ops[node.value](l, r)

        if node.kind == NodeKind.CMP:
            l, r = self._pd(node.children[0]), self._pd(node.children[1])
            cmps = {'>': l.__gt__, '<': l.__lt__, '>=': l.__ge__,
                    '<=': l.__le__, '==': l.__eq__, '!=': l.__ne__}
            return cmps[node.value](r).astype(float)

        if node.kind == NodeKind.FUNC:
            return self._pd_func(node)

        raise ValueError(f"Unknown node: {node.kind}")

    def _col(self, node: ASTNode) -> pd.Series:
        if node.kind == NodeKind.COLUMN and node.value in self.df.columns:
            return self.df[node.value].astype(float)
        raise ValueError(f"Expected column, got {node}")

    def _pd_func(self, node: ASTNode) -> pd.Series:
        name = node.value.upper()
        args = node.children
        g = self._grp()

        if name == "LOG":
            return np.log(self._pd(args[0]).replace(0, np.nan))
        if name == "ABS":
            return self._pd(args[0]).abs()
        if name == "SIGN":
            return np.sign(self._pd(args[0]))
        if name == "SQRT":
            return np.sqrt(self._pd(args[0]))
        if name == "POWER":
            return self._pd(args[0]) ** self._pd(args[1])

        if name in ("REF", "DELAY"):
            c = self._col(args[0])
            n = int(args[1].value) if len(args) > 1 else 1
            return g[c.name].shift(n)

        if name in ("MEAN", "MA", "TS_MEAN"):
            c = self._col(args[0])
            w = int(args[1].value) if len(args) > 1 else 20
            return g[c.name].rolling(w, min_periods=w).mean().droplevel(0)

        if name in ("STD", "TS_STD", "STDDEV"):
            c = self._col(args[0])
            w = int(args[1].value) if len(args) > 1 else 20
            return g[c.name].rolling(w, min_periods=w).std().droplevel(0)

        if name in ("SUM", "TS_SUM"):
            c = self._col(args[0])
            w = int(args[1].value) if len(args) > 1 else 20
            return g[c.name].rolling(w, min_periods=w).sum().droplevel(0)

        if name in ("MAX", "TS_MAX"):
            c = self._col(args[0])
            w = int(args[1].value) if len(args) > 1 else 20
            return g[c.name].rolling(w, min_periods=w).max().droplevel(0)

        if name in ("MIN", "TS_MIN"):
            c = self._col(args[0])
            w = int(args[1].value) if len(args) > 1 else 20
            return g[c.name].rolling(w, min_periods=w).min().droplevel(0)

        if name == "DELTA":
            c = self._col(args[0])
            n = int(args[1].value) if len(args) > 1 else 1
            return c - g[c.name].shift(n)

        if name in ("RANK", "CSRANK"):
            c = self._col(args[0])
            return self.df.groupby(self.date_col)[c.name].rank(pct=True)

        if name == "TS_RANK":
            c = self._col(args[0])
            w = int(args[1].value) if len(args) > 1 else 10
            def _rrank(s):
                return s.rolling(w, min_periods=w).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
            return g[c.name].apply(_rrank).droplevel(0)

        if name == "CORR":
            c1, c2 = self._col(args[0]), self._col(args[1])
            w = int(args[2].value) if len(args) > 2 else 20
            return g.apply(
                lambda g: g[c1.name].rolling(w, min_periods=w).corr(g[c2.name])
            ).droplevel(0).sort_index()

        if name == "COV":
            c1, c2 = self._col(args[0]), self._col(args[1])
            w = int(args[2].value) if len(args) > 2 else 20
            return g.apply(
                lambda g: g[c1.name].rolling(w, min_periods=w).cov(g[c2.name])
            ).droplevel(0).sort_index()

        if name in ("IIF", "IF"):
            cond = self._pd(args[0])
            t = self._pd(args[1])
            f = self._pd(args[2]) if len(args) > 2 else 0
            return pd.Series(np.where(cond > 0, t, f), index=self.df.index, dtype=float)

        if name in ("COUNT", "TS_COUNT"):
            c = self._col(args[0])
            w = int(args[1].value) if len(args) > 1 else 20
            return g[c.name].rolling(w, min_periods=1).count().droplevel(0)

        raise ValueError(f"Unknown function: {name}")

    def compute(self, ast: ASTNode) -> pd.Series:
        return self._pd(ast)


def compute_pandas(df: pd.DataFrame, expr: str, date_col: str = "date",
                   code_col: str = "symbol") -> pd.Series:
    """Compute a Qlib expression on a pandas DataFrame."""
    ast = parse(expr)
    return PandasCompiler(df, date_col=date_col, code_col=code_col).compute(ast)


# =============================================================================
# 6. High-level Engine
# =============================================================================

class ExprEngine:
    """High-level engine bridging expressions to DuckDB or Pandas computation.

    Args:
        universe: Stock pool filter. None for no filter, pool_id string (e.g. 'csi500'),
                  or raw SQL subquery returning (date, symbol).
        instruments_db: Path to DuckDB with instruments table. Required when universe is pool_id.
    """

    def __init__(self, db_path: str | None = None, table: str = "daily_ohlcv",
                 date_col: str = "date", code_col: str = "symbol",
                 universe: str | None = None, instruments_db: str | None = None):
        self.db_path = db_path
        self.table = table
        self.date_col = date_col
        self.code_col = code_col
        self.universe = universe
        self.instruments_db = instruments_db

    def compile_sql(self, expr: str, start: str = "2020-01-01",
                    end: str = "2026-12-31", universe: str | None = None) -> str:
        """Compile expression to DuckDB SQL.

        Args:
            universe: Override universe filter. If None, use engine default.
        """
        u = universe if universe is not None else self.universe
        return compile_sql(expr, self.table, self.date_col, self.code_col, start, end,
                           universe=u, instruments_db=self.instruments_db)

    def compute_sql(self, expr: str, start: str = "2020-01-01",
                    end: str = "2026-12-31", universe: str | None = None) -> pd.DataFrame:
        """Compute expression via DuckDB SQL.

        Args:
            universe: Override universe filter. If None, use engine default.
        """
        import duckdb
        sql = self.compile_sql(expr, start, end, universe=universe)
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            return con.execute(sql).fetchdf()
        finally:
            con.close()

    def compute_pandas(self, df: pd.DataFrame, expr: str) -> pd.Series:
        return compute_pandas(df, expr, self.date_col, self.code_col)

    def factor(self, expr: str, name: str = "factor", start: str = "2020-01-01",
               end: str = "2026-12-31", backend: str = "duckdb",
               universe: str | None = None) -> pd.DataFrame:
        """Expression → factor DataFrame.

        Args:
            universe: Override universe filter. If None, use engine default.
        """
        if backend == "duckdb":
            df = self.compute_sql(expr, start, end, universe=universe)
            if 'factor' in df.columns:
                df = df.rename(columns={'factor': name})
            return df
        raise ValueError("Pandas backend requires passing a DataFrame directly")

    def explain(self, expr: str) -> dict:
        """Return AST repr and generated SQL for inspection."""
        ast = parse(expr)
        sql = self.compile_sql(expr)
        return {"expression": expr, "ast": repr(ast), "sql": sql}


# =============================================================================
# 7. CLI
# =============================================================================

def _cli():
    import argparse
    p = argparse.ArgumentParser(description="Qlib Expression Engine")
    p.add_argument("expr", help="Qlib expression, e.g. '($close - Ref($close, 1)) / Ref($close, 1)'")
    p.add_argument("--backend", choices=["duckdb", "pandas"], default="duckdb")
    p.add_argument("--db", default=None, help="DuckDB path (for duckdb backend)")
    p.add_argument("--table", default="daily_kline")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--universe", default=None,
                   help="Stock pool filter: pool_id (e.g. 'csi500') or SQL subquery")
    p.add_argument("--instruments-db", default=None,
                   help="Path to DuckDB with instruments table (required when universe is pool_id)")
    p.add_argument("--explain", action="store_true", help="Print AST + SQL, don't execute")
    args = p.parse_args()

    if args.explain:
        engine = ExprEngine(db_path=args.db, table=args.table,
                            universe=args.universe, instruments_db=args.instruments_db)
        info = engine.explain(args.expr)
        print("=== AST ===")
        print(info["ast"])
        print("\n=== SQL ===")
        print(info["sql"])
        return

    if args.backend == "duckdb":
        if not args.db:
            print("Error: --db required for duckdb backend")
            sys.exit(1)
        engine = ExprEngine(db_path=args.db, table=args.table,
                            universe=args.universe, instruments_db=args.instruments_db)
        df = engine.compute_sql(args.expr, args.start, args.end)
        print(df.to_string())
    else:
        print("Pandas backend requires a DataFrame — use Python API")


if __name__ == "__main__":
    _cli()
