import pandas as pd
import numpy as np

from tests.bin.fixtures import *


def test_dummy_wrmsse(dummy_wrmsse):
    assert dummy_wrmsse.df_weights == 1