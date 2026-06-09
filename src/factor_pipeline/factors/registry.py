"""Factor registry — map factor names to callables."""

from __future__ import annotations

import inspect
from collections import OrderedDict
from typing import Callable

# ---------------------------------------------------------------------------
# global registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable] = OrderedDict()


def register_factor(arg=None, *, name: str = None):
    """Decorator to register a factor function/class.

    Usage:
        @register_factor
        def alpha001(data): ...

        @register_factor(name="my_custom_vwap")
        def some_factor(data): ...

        @register_factor("custom_name")
        def some_factor(data): ...

    Supports both bare ``@register_factor`` and parameterised forms.
    """
    # Bare decorator: @register_factor (no parens) — arg is the function itself
    if callable(arg):
        fn = arg
        fn_name = name or fn.__name__
        if fn_name in _REGISTRY:
            raise ValueError(f"Duplicate factor name: {fn_name}")
        _REGISTRY[fn_name] = fn
        return fn

    # Parameterised: @register_factor(name=...) or @register_factor("name") or @register_factor()
    if isinstance(arg, str):
        name = arg

    def _decorator(fn: Callable):
        fn_name = name or fn.__name__
        if fn_name in _REGISTRY:
            raise ValueError(f"Duplicate factor name: {fn_name}")
        _REGISTRY[fn_name] = fn
        return fn

    return _decorator


class FactorRegistry:
    """Read/write access to the global factor registry."""

    @staticmethod
    def list() -> list[str]:
        return list(_REGISTRY.keys())

    @staticmethod
    def get(name: str) -> Callable | None:
        return _REGISTRY.get(name)

    @staticmethod
    def info(name: str) -> dict:
        fn = _REGISTRY.get(name)
        if fn is None:
            return {"name": name, "found": False}
        sig = inspect.signature(fn)
        doc = (fn.__doc__ or "").strip()
        return {
            "name": name,
            "found": True,
            "signature": str(sig),
            "doc": doc,
        }

    @staticmethod
    def register(fn: Callable, name: str = None):
        if name:
            register_factor(fn, name=name)
        else:
            # Check if the callable has a 'name' attribute (e.g. classes)
            cls_name = getattr(fn, 'name', None) or fn.__name__
            register_factor(fn, name=cls_name)

    @staticmethod
    def count() -> int:
        return len(_REGISTRY)

    @staticmethod
    def clear():
        """Remove all registered factors."""
        _REGISTRY.clear()

    def __len__(self):
        return len(_REGISTRY)

    def __iter__(self):
        return iter(_REGISTRY.items())

    def __contains__(self, name: str):
        return name in _REGISTRY
