import pytest
import pandas as pd
import numpy as np

from tests.fixtures.fixtures import *
from m5_forecasting.src.aggregation import calculate_sales, calculate_hierarchy, calculate_weights
from m5_forecasting.definitions import level_dict, series_dict

PRED_COLS = ['d_1939','d_1940','d_1941']
DUMMY_PRED_COLS = ['d_11', 'd_12', 'd_13']


def test_calculate_sales(dummy_pred_with_ids, dummy_prices):
    df_sales = calculate_sales(dummy_prices, dummy_pred_with_ids, DUMMY_PRED_COLS)
    sales = df_sales['sales'].values
    correct_sales = [3, 4, 9]
    assert (sales == correct_sales).all()


def test_calculate_weights(dummy_pred_with_ids, dummy_prices, aggregation_level=3):
    # Get sales
    df_sales = calculate_sales(dummy_prices, dummy_pred_with_ids, DUMMY_PRED_COLS)

    # Get Weights
    groupby_cols = level_dict[aggregation_level]
    df_pred_weights = dummy_pred_with_ids.merge(df_sales[['id','sales']], on='id', how='left')
    df_weights = calculate_weights(df_pred_weights, groupby_cols, DUMMY_PRED_COLS)

    # Assert weights
    weights = np.round(df_weights['weights'].values, 3)
    correct_weights = np.round([3/(3+4+9), 4/(3+4+9), 9/(3+4+9)],3)
    assert (weights == correct_weights).all()


def test_aggregate_series_level3(dummy_pred_weights, aggregation_level=3):
    groupby_cols = level_dict[aggregation_level]
    num_series = series_dict[aggregation_level]
    df_out = calculate_hierarchy(dummy_pred_weights, groupby_cols, PRED_COLS)
    print(len(df_out))
    assert len(df_out) == num_series


def test_aggregate_series_level5(dummy_pred_weights, aggregation_level=5):
    groupby_cols = level_dict[aggregation_level]
    num_series = series_dict[aggregation_level]
    df_out = calculate_hierarchy(dummy_pred_weights, groupby_cols, PRED_COLS)
    print(len(df_out))
    assert len(df_out) == num_series
