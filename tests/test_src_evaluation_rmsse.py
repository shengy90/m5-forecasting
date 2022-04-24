import pytest
import pandas as pd
import numpy as np

from m5_forecasting.src.evaluation import RMSSE

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
def dummy_rmsse(dummy_pred):
    horizon = 3
    train_period = 10
    evaluator = RMSSE(dummy_pred, dummy_pred, horizon, train_period)
    return evaluator


def test_time_pred_cols(dummy_rmsse):
    """Test that train and prediction columns are populated correctly
    """
    time_cols = ['d_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8', 'd_9', 'd_10']
    pred_cols = ['d_11', 'd_12', 'd_13']
    assert dummy_rmsse.time_cols == time_cols
    assert dummy_rmsse.pred_cols == pred_cols


def test_nonzero_periods(dummy_rmsse):
    """Test that number of nonzero periods are calculated correctly
    """
    train_periods = dummy_rmsse.df_pred[dummy_rmsse.time_cols].copy().to_numpy()
    nonzero_periods_test = dummy_rmsse._get_non_zero_periods(train_periods)
    nonzero_periods_correct = np.array([8, 9, 10])
    assert (nonzero_periods_test == nonzero_periods_correct).all()


def test_denominator(dummy_rmsse):
    """Test that denominator is calculated correctly
    """
    denominator_correct = [4.25, 2, 1]
    denominator_test = dummy_rmsse.calculate_denominator()

    print(denominator_test)
    assert (denominator_correct == denominator_test).all()
