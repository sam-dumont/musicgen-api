"""Stub xformers.profiler module."""

from contextlib import contextmanager


class _Profiler:
    """Stub profiler class."""
    _CURRENT_PROFILER = None


class profile:
    """Stub profiler context manager."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# Make profiler accessible as a module with _Profiler attribute
class _ProfilerModule:
    _Profiler = _Profiler


profiler = _ProfilerModule()


@contextmanager
def profile_context(*args, **kwargs):
    """Stub profile context manager."""
    yield
