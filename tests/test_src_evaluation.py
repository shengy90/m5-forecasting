import pytest
import pandas as pd
import numpy as np

from m5_forecasting.src.evaluation import WRMSSE

@pytest.fixture
def dummy_pred():
    row1 = [0, 0, 1, 0, 0, 2, 3, 4, 5, 0, 1, 2, 0]
    row2 = [0, 1, 1, 2, 2, 2, 3, 4, 0, 0, 0, 0, 1]
    row3 = [1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 1, 2, 0]

    df = pd.DataFrame(
        np.array([row1, row2, row2]),
        columns = [f'd_{i+1}' for i in range(len(row1))]
    )

    df.index = ['A', 'B', 'C']
    return df


@pytest.fixture
def dummy_wrmsse(dummy_pred):
    horizon = 3
    train_period = 10
    evaluator = WRMSSE(dummy_pred, dummy_pred, horizon, train_period)
    return evaluator


def test_dummy_eval(dummy_wrmsse):
    assert dummy_wrmsse.horizon == 3