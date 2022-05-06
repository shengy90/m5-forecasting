import pytest
import pandas as pd
import numpy as np

from m5_forecasting.src.evaluation import RMSSE, WRMSSE
from m5_forecasting.src.forecasters import naive_forecaster


HORIZON = 3
TRAIN_PERIOD = 10

@pytest.fixture
def dummy_pred():
    row1 = [0, 0, 1, 0, 0, 2, 3, 4, 5, 0, 1, 2, 0]
    row2 = [0, 1, 3, 1, 3, 2, 2, 2, 0, 0, 0, 0, 2]
    row3 = [1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 1, 2, 0]

    df = pd.DataFrame(
        np.array([row1, row2, row3]),
        columns = [f'd_{i+1}' for i in range(len(row1))]
    )

    df.index = ['A', 'B', 'C']
    return df


@pytest.fixture
def dummy_weights():
    row1 = [0.1]
    row2 = [0.4]
    row3 = [0.5]

    df = pd.DataFrame(
        np.array([row1, row2, row3]),
        columns = ['weights']
    )

    df.index = ['A', 'B', 'C']
    return df


@pytest.fixture
def dummy_rmsse(dummy_pred):
    evaluator = RMSSE(dummy_pred, dummy_pred, HORIZON, TRAIN_PERIOD)
    return evaluator


@pytest.fixture
def dummy_wrmsse(dummy_pred, dummy_weights):
    evaluator = WRMSSE(dummy_pred, dummy_pred, HORIZON, TRAIN_PERIOD, df_weights=dummy_weights)
    return evaluator


@pytest.fixture
def dummy_naive_forecaster(dummy_pred):
    forecaster = naive_forecaster.NaiveForecaster(dummy_pred, HORIZON, TRAIN_PERIOD)
    return forecaster


@pytest.fixture
def dummy_prices():
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    sell_price_dicts = {
        'A': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'B': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        'C': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    }
    period_ids = [f'd_{i+1}' for i in range(len(sell_price_dicts['A']))]

    for idx in sell_price_dicts:
        df1['sell_price'] = sell_price_dicts[idx]
        df1['item_id'] = idx
        df1['store_id'] = idx
        df1['period'] = period_ids
        df = pd.concat([df, df1])

    return df