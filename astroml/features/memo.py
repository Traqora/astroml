"""Parse and extract features from transaction memos.

This module provides utilities for parsing Stellar transaction memos and
extracting features for machine learning. Memos can be of type 'text' (string),
'id' (64-bit unsigned integer), or 'hash' (32-byte hash). The module handles
malformed memos gracefully by flagging them and setting invalid values to None.

Features extracted:
- memo_type: 'text', 'id', 'hash', or 'none'
- memo_value: normalized value (string, int, or hex string)
- memo_length: length of text memos (0 for others)
- is_malformed: boolean indicating if the memo was malformed
"""
from typing import Dict, Any, Optional, Union
import pandas as pd


def parse_memo(memo: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse a single memo dict into structured features.

    Args:
        memo: Dict with 'type' and 'value' keys, or None.

    Returns:
        Dict with 'type', 'value', 'is_malformed' keys.

    Examples:
        >>> parse_memo({'type': 'text', 'value': 'hello'})
        {'type': 'text', 'value': 'hello', 'is_malformed': False}
        >>> parse_memo({'type': 'id', 'value': '123'})
        {'type': 'id', 'value': 123, 'is_malformed': False}
        >>> parse_memo({'type': 'invalid', 'value': 'x'})
        {'type': 'none', 'value': None, 'is_malformed': True}
    """
    if not isinstance(memo, dict) or 'type' not in memo or 'value' not in memo:
        return {'type': 'none', 'value': None, 'is_malformed': True}

    type_ = memo['type'].lower()
    value = memo['value']
    is_malformed = False

    if type_ == 'text':
        if not isinstance(value, str):
            is_malformed = True
            value = None
    elif type_ == 'id':
        try:
            value = int(value)
            if not (0 <= value <= 2**64 - 1):
                is_malformed = True
                value = None
        except (ValueError, TypeError):
            is_malformed = True
            value = None
    elif type_ == 'hash':
        if isinstance(value, bytes) and len(value) == 32:
            value = value.hex()
        elif isinstance(value, str):
            value = value.lower()
            if len(value) != 64 or not all(c in '0123456789abcdef' for c in value):
                is_malformed = True
                value = None
        else:
            is_malformed = True
            value = None
    else:
        is_malformed = True
        type_ = 'none'
        value = None

    return {'type': type_, 'value': value, 'is_malformed': is_malformed}


def extract_memo_features(
    df: pd.DataFrame,
    memo_col: str = 'memo',
    out_prefix: str = 'memo_'
) -> pd.DataFrame:
    """Extract memo features from a DataFrame with a memo column.

    Adds columns: memo_type, memo_value, memo_length, is_malformed.

    Args:
        df: DataFrame with memo column.
        memo_col: Name of the memo column.
        out_prefix: Prefix for output columns.

    Returns:
        DataFrame with added feature columns.

    Raises:
        KeyError: If memo_col not in df.
    """
    if memo_col not in df:
        raise KeyError(f"DataFrame must contain '{memo_col}' column")

    df_out = df.copy()
    parsed = df_out[memo_col].apply(parse_memo)

    df_out[f'{out_prefix}type'] = parsed.apply(lambda x: x['type'])
    df_out[f'{out_prefix}value'] = parsed.apply(lambda x: x['value'])
    df_out[f'{out_prefix}length'] = parsed.apply(
        lambda x: len(x['value']) if x['type'] == 'text' and x['value'] else 0
    )
    df_out[f'{out_prefix}is_malformed'] = parsed.apply(lambda x: x['is_malformed'])

    return df_out
