from abc import ABC
import pandas as pd
import numpy as np

class BaseEvaluator(ABC):
    def __init__(
        self, df_pred: pd.DataFrame, 
        df_eval: pd.DataFrame, 
        horizon: int,
        train_period: int
        ) -> None:
        """_summary_
        Args:
            df_pred (pd.DataFrame): Pandas Dataframe containing predicted values
            df_eval (pd.DataFrame): Pandas Dataframe containing real values
            horizon (int): forecast horizon window
            train_period (int): last time step of training period
        """
        self.df_pred = df_pred
        self.df_eval = df_eval
        self.horizon = horizon
        return None


class WRMSSE(BaseEvaluator):
    def __init__(self, df_pred, df_eval, horizon, train_period) -> None:
        super().__init__(df_pred, df_eval, horizon, train_period)

