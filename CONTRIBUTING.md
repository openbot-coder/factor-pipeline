# Contributing to Factor Pipeline

Thank you for your interest in contributing to Factor Pipeline!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/openbot-coder/factor-pipeline.git
cd factor-pipeline
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test file
pytest tests/test_ic.py -v
```

## Code Style

We use:
- **Ruff** for linting
- **Black** for code formatting

Format your code before committing:
```bash
ruff check . --fix
black .
```

## Adding New Factors

### Option 1: Simple Function (Recommended)

```python
from factors.registry import register_factor

@register_factor
def my_factor(data: dict) -> pd.Series:
    """Factor description here."""
    close = data["close"].iloc[:, 0]
    return close.pct_change(5).rename("my_factor")
```

### Option 2: FactorABC Class (For Complex Factors)

```python
from factors.base import FactorABC

class MyComplexFactor(FactorABC):
    name = "my_complex_factor"
    dependencies = ["close", "volume", "high", "low"]
    max_window = 20
    description = "Description of the factor"

    def compute(self) -> pd.DataFrame:
        c = self._col("close")
        # Your logic here
        return result.rename(columns={"close": "factor"})
```

## Project Structure

```
factor-pipeline/
├── factors/           # Factor implementations
│   ├── base.py       # FactorABC base class
│   ├── registry.py   # Factor registration
│   ├── gtja191.py    # GTJA 191 factors
│   └── technical.py  # Technical indicators
├── data/             # Data loading & preprocessing
├── analysis/         # IC, backtest, reports
├── config/           # Configuration
└── cli/              # CLI interface
```

## Submitting Changes

1. Create a new branch for your feature
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Reporting Issues

Please include:
- Python version
- Operating system
- Minimal reproducible example
- Expected vs actual behavior

## Questions?

Open an issue on GitHub or start a discussion.
