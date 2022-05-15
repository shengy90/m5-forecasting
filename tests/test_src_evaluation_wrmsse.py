import pandas as pd
import numpy as np

from tests.fixtures.fixtures import *
from m5_forecasting.src.aggregation import calculate_sales

DUMMY_PRED_COLS = ['d_11', 'd_12', 'd_13']

def test_dummy_wrmsse(dummy_wrmsse, dummy_weights):
    weights = dummy_weights['weights'].values
    rmsse = np.array([0.485, 0.707, 1.])

    dummy_wrmsse.df_eval = dummy_wrmsse.df_pred + 1
    dummy_wrmsse.numer = dummy_wrmsse.calculate_numerator()
    dummy_wrmsse.denom = dummy_wrmsse.calculate_denominator()
    dummy_wrmsse.rmsse = np.round(dummy_wrmsse.calculate_rmsse(), 3)
    weighted_rmsse = np.round(dummy_wrmsse.calculate_wrmsse(), 3)
    correct_wrmsse = np.round(np.multiply(weights, rmsse), 3)
    assert (weighted_rmsse == correct_wrmsse).all()
