"""Centralized logging configuration for the pipeline.

Library modules obtain a logger with ``get_logger(__name__)`` and emit messages
through it; they never configure handlers or call ``print`` for diagnostics.
The CLI entry point (``run/main.py``) calls :func:`configure_logging` exactly
once to attach a console handler and choose the verbosity level.

This replaces the previous pattern of gating diagnostic ``print`` calls behind
``TQDM_AVAILABLE`` and globally suppressing warnings, which conflated progress
output with library availability and hid real problems.
"""

import logging

_DEFAULT_FORMAT = "%(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return the module logger for ``name`` (typically ``__name__``)."""
    return logging.getLogger(name)


def configure_logging(verbose: bool = True, *, fmt: str = _DEFAULT_FORMAT) -> None:
    """Attach a single console handler to the root logger.

    Safe to call more than once: a console handler is only added if one is not
    already present, so repeated calls just adjust the level.

    Args:
        verbose: When True, emit ``INFO`` and above; otherwise ``WARNING`` and above.
        fmt: Logging format string. Defaults to the bare message to preserve the
            pipeline's plain-text console output.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO if verbose else logging.WARNING)

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)
