import pytest
import pandas as pd
import numpy as np

from tests.bin.fixtures import *


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
    assert (dummy_rmsse.denom == denominator_correct).all()


def test_numerator(dummy_rmsse):
    """Test that numerator is calculated correctly
    """
    numerator_correct = [0., 0., 0.]
    numerator_test = dummy_rmsse.calculate_numerator()
    assert (dummy_rmsse.numer == numerator_correct).all()

    dummy_rmsse.df_eval = dummy_rmsse.df_pred + 1
    numerator_test2 = dummy_rmsse.calculate_numerator()
    assert (numerator_test2 == np.array([1., 1., 1.])).all()


def test_rmsse(dummy_rmsse):
    """Test that rmsse is calculated correctly
    """
    rmsse_correct = [0., 0., 0.]
    rmsse_test = dummy_rmsse.rmsse
    assert (rmsse_correct == rmsse_test).all()

    dummy_rmsse.df_eval = dummy_rmsse.df_pred + 1
    new_numer = dummy_rmsse.calculate_numerator()
    new_denom = dummy_rmsse.calculate_denominator()
    new_rmsse = np.round(dummy_rmsse.calculate_rmsse(new_numer, new_denom), 3)
    assert (new_rmsse == np.array([0.485, 0.707, 1.])).all()
