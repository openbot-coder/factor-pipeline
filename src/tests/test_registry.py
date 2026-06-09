"""Comprehensive tests for factors/registry.py — register_factor and FactorRegistry.

Covers:
- Bare decorator (@register_factor)
- Decorator with keyword name (@register_factor(name="xxx"))
- Decorator with positional string arg (@register_factor("xxx"))
- Empty parentheses (@register_factor())
- FactorRegistry.list(), .get(), .info(), .count(), .clear()
- Duplicate registration errors
- FactorRegistry.register() static method
- Dunder methods (__len__, __iter__, __contains__)
- Error handling for info() on missing factors
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from factor_pipeline.factors.registry import FactorRegistry, _REGISTRY, register_factor


# =============================================================================
# Bare Decorator Tests
# =============================================================================


class TestBareDecorator:
    """Tests for @register_factor (bare, no parentheses)."""

    def test_registers_using_func_name(self):
        """Positive: Bare decorator registers using function __name__."""

        @register_factor
        def my_bare_alpha(data):
            return data

        assert "my_bare_alpha" in FactorRegistry.list()
        assert FactorRegistry.get("my_bare_alpha") is my_bare_alpha

    def test_returns_original_function(self):
        """Positive: Bare decorator returns the original function unchanged."""

        @register_factor
        def my_original(data):
            """My docstring."""
            return data

        retrieved = FactorRegistry.get("my_original")
        assert retrieved.__name__ == "my_original"
        assert retrieved.__doc__ == "My docstring."

    def test_multiple_bare_decorators(self):
        """Positive: Multiple bare decorators each get their own name."""

        @register_factor
        def alpha_v1(data):
            return data

        @register_factor
        def alpha_v2(data):
            return data

        assert "alpha_v1" in FactorRegistry.list()
        assert "alpha_v2" in FactorRegistry.list()


# =============================================================================
# Decorator with name Keyword
# =============================================================================


class TestDecoratorWithName:
    """Tests for @register_factor(name="xxx")."""

    def test_uses_custom_name(self):
        """Positive: Decorator with name= keyword uses given name."""

        @register_factor(name="custom_vwap")
        def compute_vwap(data):
            return data

        assert "custom_vwap" in FactorRegistry.list()
        assert FactorRegistry.get("custom_vwap") is compute_vwap

    def test_returns_original_fn(self):
        """Positive: Returns original function, not a wrapper."""

        @register_factor(name="named_factor")
        def some_fn(data):
            return data

        assert FactorRegistry.get("named_factor").__name__ == "some_fn"

    def test_fn_name_differs_from_registry_name(self):
        """Boundary: Internal __name__ differs from registry name."""

        @register_factor(name="registry_name")
        def internal_name(data):
            return data

        # The registry uses 'registry_name' but the function's __name__ is 'internal_name'
        assert FactorRegistry.get("registry_name").__name__ == "internal_name"


# =============================================================================
# Empty Parentheses
# =============================================================================


class TestEmptyParens:
    """Tests for @register_factor() with no arguments."""

    def test_empty_parens_registers(self):
        """Positive: @register_factor() uses fn.__name__."""

        @register_factor()
        def auto_named(data):
            return data

        assert "auto_named" in FactorRegistry.list()

    def test_returns_original_fn(self):
        """Positive: Returns original function."""

        @register_factor()
        def some_fn(data):
            return data

        assert FactorRegistry.get("some_fn") is some_fn


# =============================================================================
# Positional String Argument
# =============================================================================


class TestPositionalArg:
    """Tests for @register_factor("positional_name")."""

    def test_positional_string(self):
        """Positive: @register_factor("custom_name") uses the string as name."""

        @register_factor("positional_name")
        def compute_something(data):
            return data

        assert "positional_name" in FactorRegistry.list()
        assert FactorRegistry.get("positional_name") is compute_something

    def test_positional_arg_returns_original(self):
        """Positive: Returns original function."""

        @register_factor("pos_name")
        def fn(data):
            return data

        assert FactorRegistry.get("pos_name").__name__ == "fn"


# =============================================================================
# Duplicate Registration Errors
# =============================================================================


class TestDuplicateRegistration:
    """Tests for ValueError on duplicate factor names."""

    def test_duplicate_bare_raises(self):
        """Negative: Registering same name twice with bare decorator raises ValueError."""

        @register_factor
        def unique_dup_alpha(data):
            return data

        with pytest.raises(ValueError, match="Duplicate factor name: unique_dup_alpha"):

            @register_factor
            def unique_dup_alpha(data):
                return data

    def test_duplicate_named_raises(self):
        """Negative: Registering same name twice with name= raises ValueError."""

        @register_factor(name="dup_named")
        def fn_a(data):
            return data

        with pytest.raises(ValueError, match="dup_named"):

            @register_factor(name="dup_named")
            def fn_b(data):
                return data

    def test_duplicate_positional_raises(self):
        """Negative: Registering same name with positional arg raises ValueError."""

        @register_factor("dup_positional")
        def fn1(data):
            return data

        with pytest.raises(ValueError, match="dup_positional"):

            @register_factor("dup_positional")
            def fn2(data):
                return data

    def test_different_names_no_conflict(self):
        """Positive: Different names don't conflict."""

        @register_factor
        def unique_alpha_ok_1(data):
            return data

        @register_factor
        def unique_alpha_ok_2(data):
            return data

        assert "unique_alpha_ok_1" in FactorRegistry.list()
        assert "unique_alpha_ok_2" in FactorRegistry.list()


# =============================================================================
# FactorRegistry.list() Tests
# =============================================================================


class TestRegistryList:
    """Tests for FactorRegistry.list()."""

    def test_returns_list_of_strings(self):
        """Positive: list() returns a list of strings."""
        result = FactorRegistry.list()
        assert isinstance(result, list)
        assert all(isinstance(n, str) for n in result)

    def test_includes_newly_registered(self):
        """Positive: newly registered factor appears in list."""

        @register_factor(name="list_test_alpha")
        def fn(data):
            return data

        assert "list_test_alpha" in FactorRegistry.list()

    def test_returns_copy(self):
        """Boundary: Modifying the returned list does not affect _REGISTRY."""
        before_len = len(FactorRegistry.list())
        lst = FactorRegistry.list()
        lst.append("fake_factor_xyz")
        assert len(FactorRegistry.list()) == before_len


# =============================================================================
# FactorRegistry.get() Tests
# =============================================================================


class TestRegistryGet:
    """Tests for FactorRegistry.get()."""

    def test_get_existing(self):
        """Positive: get() returns the callable for an existing factor."""

        @register_factor
        def get_test_fn(data):
            return data

        assert FactorRegistry.get("get_test_fn") is get_test_fn

    def test_get_nonexistent(self):
        """Negative: get() returns None for a missing name."""
        assert FactorRegistry.get("totally_nonexistent_xyz") is None

    def test_get_none_name(self):
        """Boundary: get(None) returns None without error."""
        assert FactorRegistry.get(None) is None

    def test_get_empty_string(self):
        """Boundary: get("") returns None for empty string."""
        assert FactorRegistry.get("") is None


# =============================================================================
# FactorRegistry.info() Tests
# =============================================================================


class TestRegistryInfo:
    """Tests for FactorRegistry.info()."""

    def test_info_existing_factor(self):
        """Positive: info() returns signature and docstring for existing factor."""

        @register_factor(name="info_test_fn")
        def documented_factor(data, threshold=0.5):
            """Compute something useful."""
            return data

        info = FactorRegistry.info("info_test_fn")
        assert info["found"] is True
        assert info["name"] == "info_test_fn"
        assert "signature" in info
        assert "doc" in info
        assert "Compute something useful" in info["doc"]

    def test_info_missing_factor(self):
        """Negative: info() returns found=False for missing factor."""
        info = FactorRegistry.info("nonexistent_info_xyz")
        assert info["found"] is False
        assert info["name"] == "nonexistent_info_xyz"
        assert "signature" not in info or info.get("signature") is None

    def test_info_has_correct_signature(self):
        """Positive: signature string matches the actual function signature."""

        @register_factor(name="info_sig_test")
        def fn_with_params(data, window=20, min_periods=None):
            return data

        info = FactorRegistry.info("info_sig_test")
        assert "window" in info["signature"]
        assert "min_periods" in info["signature"]

    def test_info_none_name(self):
        """Boundary: info(None) returns found=False."""
        info = FactorRegistry.info(None)
        assert info["found"] is False


# =============================================================================
# FactorRegistry.count() Tests
# =============================================================================


class TestRegistryCount:
    """Tests for FactorRegistry.count()."""

    def test_count_matches_list(self):
        """Positive: count() matches length of list()."""
        assert FactorRegistry.count() == len(FactorRegistry.list())

    def test_count_increases_after_register(self):
        """Positive: count increases after registration."""
        before = FactorRegistry.count()

        @register_factor
        def count_test_fn(data):
            return data

        assert FactorRegistry.count() == before + 1


# =============================================================================
# FactorRegistry.register() Static Method
# =============================================================================


class TestRegistryRegisterMethod:
    """Tests for FactorRegistry.register() static method."""

    def test_register_with_explicit_name(self):
        """Positive: register() with explicit name."""

        def my_factor(data):
            return data

        FactorRegistry.register(my_factor, name="manual_register")
        assert "manual_register" in FactorRegistry.list()
        assert FactorRegistry.get("manual_register") is my_factor

    def test_register_without_name(self):
        """Positive: register() without name uses fn.__name__."""

        def auto_register_fn(data):
            return data

        FactorRegistry.register(auto_register_fn)
        assert "auto_register_fn" in FactorRegistry.list()

    def test_register_class_with_name_attr(self):
        """Positive: register() with class that has 'name' attribute."""

        class MyClassFactor:
            name = "class_factor_1"

            def compute(self, data):
                return data

        FactorRegistry.register(MyClassFactor)
        assert "class_factor_1" in FactorRegistry.list()

    def test_register_duplicate_raises(self):
        """Negative: register() a duplicate name raises ValueError."""

        @register_factor(name="dup_register_test")
        def fn1(data):
            return data

        def fn2(data):
            return data

        with pytest.raises(ValueError, match="dup_register_test"):
            FactorRegistry.register(fn2, name="dup_register_test")


# =============================================================================
# Dunder Method Tests
# =============================================================================


class TestRegistryDunder:
    """Tests for __len__, __iter__, __contains__."""

    def test_len(self):
        """Positive: len(FactorRegistry()) matches count."""
        registry = FactorRegistry()
        assert len(registry) == FactorRegistry.count()

    def test_iter(self):
        """Positive: iterating FactorRegistry yields (name, fn) tuples."""

        @register_factor
        def iter_test_fn(data):
            return data

        registry = FactorRegistry()
        items = dict(registry)
        assert "iter_test_fn" in items
        assert items["iter_test_fn"] is iter_test_fn

    def test_contains_positive(self):
        """Positive: 'in' operator works for registered factor."""

        @register_factor
        def contain_test_fn(data):
            return data

        registry = FactorRegistry()
        assert "contain_test_fn" in registry

    def test_contains_negative(self):
        """Negative: 'in' operator returns False for unregistered factor."""
        registry = FactorRegistry()
        assert "not_in_registry_xyz" not in registry


# =============================================================================
# State Isolation Tests (relies on conftest autouse fixture)
# =============================================================================


class TestRegistryIsolation:
    """Tests for registry state isolation."""

    def test_registry_restores_after_test(self):
        """Boundary: Registry state is restored between tests by conftest fixture."""

        @register_factor
        def isolation_test_fn(data):
            return data

        assert "isolation_test_fn" in FactorRegistry.list()
        # conftest autouse fixture will restore after this test.

    def test_clear_then_repopulate(self):
        """Edge: Clear registry, add new, then check."""
        FactorRegistry.clear()
        assert FactorRegistry.count() == 0

        @register_factor
        def after_clear_fn(data):
            return data

        assert FactorRegistry.count() == 1
        assert "after_clear_fn" in FactorRegistry.list()

    def test_clear_removes_all(self):
        """Edge: clear() empties the entire registry."""
        # First add something
        @register_factor
        def to_be_cleared_fn(data):
            return data

        assert FactorRegistry.count() > 0
        FactorRegistry.clear()
        assert FactorRegistry.count() == 0
        assert FactorRegistry.list() == []
