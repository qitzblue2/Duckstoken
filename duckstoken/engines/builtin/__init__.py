"""Duckstoken's built-in inference engine: runs small GGUF models with no other program installed (it needs NumPy).

`from duckstoken.engines.builtin import BuiltinEngine` raises ImportError when NumPy is missing, which makes the engine list leave
it out. The submodules (for example `catalog`) can always be imported.
"""

__all__ = ["BuiltinEngine"]


def __getattr__(name: str):
    if name != "BuiltinEngine":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        import numpy  # noqa: F401
    except ImportError as e:
        raise ImportError("The built-in engine needs NumPy, which is not installed.") from e
    from .engine import BuiltinEngine
    return BuiltinEngine
