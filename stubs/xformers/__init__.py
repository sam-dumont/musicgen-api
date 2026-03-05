"""Stub xformers module for systems without CUDA.

This provides minimal compatibility for audiocraft on macOS/MPS
where xformers is not available.
"""

# Indicate this is a stub
__version__ = "0.0.0.stub"
_is_stub = True
