import numpy as np
import pandas as pd

from astroml.features import imbalance


def test_net_flow_ratio_scalars():
    r = imbalance.net_flow_ratio(100.0, 40.0)
    assert np.isclose(r, (100.0 - 40.0) / (100.0 + 40.0))


def test_net_flow_ratio_array():
    sent = [1.0, 0.0, 5.0]
    recv = [0.0, 1.0, 5.0]
    r = imbalance.net_flow_ratio(sent, recv)
    assert np.allclose(r, np.array([1.0, -1.0, 0.0]))


def test_log_scale_changes_result():
    sent = np.array([1.0, 1000.0])
    recv = np.array([1.0, 1.0])
    r_linear = imbalance.net_flow_ratio(sent, recv, log_scale=False)
    r_log = imbalance.net_flow_ratio(sent, recv, log_scale=True)
    # log compression should make the extreme case closer to 0 than linear
    assert abs(r_log[1]) < abs(r_linear[1])


def test_series_preserves_index():
    idx = pd.Index(["a", "b"])
    sent = pd.Series([10.0, 0.0], index=idx)
    recv = pd.Series([0.0, 5.0], index=idx)
    out = imbalance.net_flow_ratio(sent, recv)
    assert isinstance(out, pd.Series)
    assert list(out.index) == ["a", "b"]


def test_dataframe_helper():
    df = pd.DataFrame({"sent_amount": [10.0, 0.0], "received_amount": [0.0, 10.0]})
    out = imbalance.net_flow_ratio_from_transactions(df)
    assert "net_flow_ratio" in out.columns
    assert np.allclose(out["net_flow_ratio"].values, np.array([1.0, -1.0]))
