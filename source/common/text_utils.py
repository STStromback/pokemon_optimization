"""Canonical text normalization for the Pokemon optimization pipeline.

A single normalization implementation is used for all join keys (Pokemon names,
locations, move names, types). Previously this logic was duplicated across many
modules with subtly different rules, which risked silent join mismatches.

Normalization rules (these match the project's test suite):
  - ``None`` / NaN  -> ``""``
  - lowercase
  - gender symbols: ``\u2640`` -> ``_f``, ``\u2642`` -> ``_m``
  - strip accents/diacritics (e.g. ``e`` with acute -> ``e``)
  - spaces -> underscore
  - hyphens and other punctuation are removed (e.g. ``Ho-Oh`` -> ``hooh``)
  - keep only ``[a-z0-9_/]``
  - collapse repeated underscores and strip leading/trailing underscores
"""

import unicodedata
from functools import lru_cache

import pandas as pd

_KEEP_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_/")


@lru_cache(maxsize=50000)
def normalize_text(value) -> str:
    """Normalize a single value to a canonical join-key string (cached)."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).lower()
    text = text.replace("\u2640", "_f").replace("\u2642", "_m")

    # Strip accents/diacritics (NFKD then drop combining marks).
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    text = text.replace(" ", "_")
    text = "".join(ch for ch in text if ch in _KEEP_CHARS)

    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with all text columns normalized.

    NaN values are preserved. Normalization is applied per unique value for
    efficiency on large frames.
    """
    df = df.copy()
    text_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    for col in text_cols:
        unique_vals = df[col].dropna().unique()
        mapping = {val: normalize_text(val) for val in unique_vals}
        df[col] = df[col].map(mapping)
    return df
