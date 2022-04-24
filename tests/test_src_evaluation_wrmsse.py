import pandas as pd
import numpy as np

from tests.fixtures.fixtures import *


def test_dummy_wrmsse(dummy_wrmsse):
    assert dummy_wrmsse.df_weights == 1
    assert (dummy_wrmsse.rmsse == np.array([0., 0., 0.])).all()
    assert (dummy_wrmsse.denom == np.array([4.25, 2, 1])).all()
