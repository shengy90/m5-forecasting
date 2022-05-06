import pytest
import pandas as pd
import numpy as np

from tests.fixtures.fixtures import *
from m5_forecasting.src.aggregation import calculate_sales


def test_calculate_sales(dummy_pred, dummy_prices):
    pred_cols = ['d_11', 'd_12', 'd_13']
    id_vals = ['A', 'B', 'C']
    dummy_pred['id'] = id_vals
    dummy_pred['item_id'] = id_vals
    dummy_pred['state_id'] = id_vals
    dummy_pred['store_id'] = id_vals

    df_sales = calculate_sales(dummy_prices, dummy_pred, pred_cols)
    sales = df_sales['sales'].values
    correct_sales = [3, 4, 9]
    assert (sales == correct_sales).all()