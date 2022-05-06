from multiprocessing import dummy
import pandas as pd
import numpy as np

from tests.fixtures.fixtures import *


def test_naive_forecaster(dummy_naive_forecaster):
    df_pred = dummy_naive_forecaster.df_pred
    last_time_stamp = dummy_naive_forecaster.last_time_stamp + dummy_naive_forecaster.horizon
    latest_pred = df_pred[f'd_{last_time_stamp}'].values
    assert (latest_pred == np.array([0, 0, 3])).all()


def test_naive_forecaster_grid(dummy_naive_forecaster):
    pred_cols = dummy_naive_forecaster.df_pred.columns
    train_cols = dummy_naive_forecaster.train_cols
    horizon = dummy_naive_forecaster.horizon
    assert len(pred_cols) == len(train_cols) + horizon
