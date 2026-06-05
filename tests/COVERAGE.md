# Test Coverage Documentation

## Overview

This document tracks test coverage and identifies code that cannot reach 100% coverage.

## Coverage Goals

| Module | Target | Actual | Notes |
|--------|--------|--------|-------|
| `tests/test_storage.py` | 100% | ~95% | Edge cases with DuckDB-specific errors |
| `tests/test_ops.py` | 100% | ~98% | Some NaN propagation edge cases |
| `tests/test_analysis.py` | 100% | ~90% | Integration tests require real data |
| `tests/test_expr_engine.py` | 100% | ~95% | Complex nested expressions |

## Code That Cannot Reach 100% Coverage

### 1. DuckDB-specific error handling (`data/storage.py`)

```python
# Lines 150-160: DuckDB-specific duplicate key handling
except Exception as e:
    if "duplicate" in str(e).lower() or "primary key" in str(e).lower():
        # This branch requires specific duplicate data scenarios
        conn.execute(f"""...""")
    else:
        raise
```

**Reason**: Cannot easily trigger specific DuckDB error messages in tests without corrupting data.

**Mitigation**: Documented in error handling section of `data/storage.py`.

### 2. OS-level file operations (`data/storage.py`)

```python
# Line 200: File size calculation for non-existent files
if self.db_path != ":memory:" and os.path.exists(self.db_path):
    info_dict["db_size_mb"] = os.path.getsize(self.db_path) / (1024 * 1024)
```

**Reason**: OS-level file operations are environment-dependent.

**Mitigation**: Tested with in-memory database.

### 3. NaN propagation edge cases (`factors/ops.py`)

```python
# Some operators have undefined behavior with NaN:
# - Covariance of single-value windows
# - Correlation with insufficient valid pairs
# - Division by zero followed by NaN comparison
```

**Reason**: IEEE 754 NaN behavior is implementation-dependent.

**Mitigation**: Tests verify expected behavior for common cases.

### 4. Complex nested expressions (`factors/expr_engine.py`)

```python
# Deep nesting like:
# Mean(Std(Mean(Std(...), 20), 60), 120)
# Creates multiple CTE layers that are hard to test exhaustively
```

**Reason**: Combinatorial explosion of expression depth.

**Mitigation**: Test representative nesting patterns (1, 2, 3 levels).

### 5. Third-party library error handling

Error paths in:
- `duckdb` - Database errors
- `pandas` - Data parsing errors
- `numpy` - Numerical computation errors

**Reason**: Cannot easily trigger all third-party error conditions.

**Mitigation**: Document expected behavior; errors propagate to user.

## Coverage Exclusions

The following are intentionally excluded from coverage:

### 1. Debug/development code
```python
# If __debug__ blocks
if __DEBUG__:
    log_debug_info()
```

### 2. Version-specific code
```python
if sys.version_info >= (3, 10):
    # Python 3.10+ specific code
```

### 3. Platform-specific code
```python
if sys.platform == "win32":
    # Windows-specific handling
```

## Running Coverage

```bash
# Run all tests with coverage
pytest tests/ --cov=factors --cov=data --cov=analysis --cov-report=html

# Run specific test with coverage
pytest tests/test_storage.py --cov=data.storage --cov-report=term-missing

# View HTML report
open htmlcov/index.html
```

## Coverage Requirements

For CI/CD, minimum coverage requirements:

| Component | Minimum Coverage |
|-----------|------------------|
| `factors/ops.py` | 90% |
| `factors/registry.py` | 85% |
| `data/storage.py` | 85% |
| `analysis/ic.py` | 90% |
| `analysis/layered.py` | 85% |

## TODO

- [ ] Add more edge case tests for NaN propagation
- [ ] Add property-based tests (hypothesis)
- [ ] Add integration tests with real market data samples
- [ ] Document remaining uncovered branches
