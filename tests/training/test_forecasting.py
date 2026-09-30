from astroml.training.time_series.arima_model import ARIMAModel
from astroml.training.time_series.ensemble import EnsembleForecaster
from astroml.training.time_series.lstm_model import LSTMModel
from astroml.training.time_series.prophet_model import ProphetModel


def test_forecasting() -> None:
    ARIMAModel().fit()
    ProphetModel().fit()
    LSTMModel().fit()
    assert EnsembleForecaster().forecast() == []
