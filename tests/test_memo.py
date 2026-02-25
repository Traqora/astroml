import pandas as pd
import pytest

from astroml.features import memo


def test_parse_memo_text_valid():
    result = memo.parse_memo({'type': 'text', 'value': 'hello world'})
    assert result == {'type': 'text', 'value': 'hello world', 'is_malformed': False}


def test_parse_memo_text_invalid_value():
    result = memo.parse_memo({'type': 'text', 'value': 123})
    assert result == {'type': 'text', 'value': None, 'is_malformed': True}


def test_parse_memo_id_valid():
    result = memo.parse_memo({'type': 'id', 'value': '12345'})
    assert result == {'type': 'id', 'value': 12345, 'is_malformed': False}


def test_parse_memo_id_invalid_value():
    result = memo.parse_memo({'type': 'id', 'value': 'not_a_number'})
    assert result == {'type': 'id', 'value': None, 'is_malformed': True}


def test_parse_memo_id_out_of_range():
    result = memo.parse_memo({'type': 'id', 'value': str(2**64)})
    assert result == {'type': 'id', 'value': None, 'is_malformed': True}


def test_parse_memo_hash_valid_bytes():
    hash_bytes = b'\x00' * 32
    result = memo.parse_memo({'type': 'hash', 'value': hash_bytes})
    assert result == {'type': 'hash', 'value': '00' * 32, 'is_malformed': False}


def test_parse_memo_hash_valid_hex():
    hex_str = 'a' * 64
    result = memo.parse_memo({'type': 'hash', 'value': hex_str})
    assert result == {'type': 'hash', 'value': 'a' * 64, 'is_malformed': False}


def test_parse_memo_hash_invalid():
    result = memo.parse_memo({'type': 'hash', 'value': 'short'})
    assert result == {'type': 'hash', 'value': None, 'is_malformed': True}


def test_parse_memo_invalid_type():
    result = memo.parse_memo({'type': 'unknown', 'value': 'x'})
    assert result == {'type': 'none', 'value': None, 'is_malformed': True}


def test_parse_memo_none():
    result = memo.parse_memo(None)
    assert result == {'type': 'none', 'value': None, 'is_malformed': True}


def test_parse_memo_missing_keys():
    result = memo.parse_memo({'type': 'text'})
    assert result == {'type': 'none', 'value': None, 'is_malformed': True}


def test_extract_memo_features():
    df = pd.DataFrame({
        'memo': [
            {'type': 'text', 'value': 'hello'},
            {'type': 'id', 'value': '42'},
            {'type': 'hash', 'value': 'a' * 64},
            {'type': 'invalid', 'value': 'x'},
            None
        ]
    })
    result = memo.extract_memo_features(df)
    assert 'memo_type' in result.columns
    assert 'memo_value' in result.columns
    assert 'memo_length' in result.columns
    assert 'memo_is_malformed' in result.columns

    assert result['memo_type'].tolist() == ['text', 'id', 'hash', 'none', 'none']
    assert result['memo_value'].tolist() == ['hello', 42, 'a' * 64, None, None]
    assert result['memo_length'].tolist() == [5, 0, 0, 0, 0]
    assert result['memo_is_malformed'].tolist() == [False, False, False, True, True]


def test_extract_memo_features_missing_column():
    df = pd.DataFrame({'other': [1, 2]})
    with pytest.raises(KeyError):
        memo.extract_memo_features(df)


def test_extract_memo_features_custom_prefix():
    df = pd.DataFrame({'memo': [{'type': 'text', 'value': 'test'}]})
    result = memo.extract_memo_features(df, out_prefix='m_')
    assert 'm_type' in result.columns
