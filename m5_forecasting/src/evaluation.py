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
        """Base Evaluator Class
        Args:
            df_pred (pd.DataFrame): Pandas Dataframe containing predicted values
            df_eval (pd.DataFrame): Pandas Dataframe containing real values
            horizon (int): forecast horizon window
            train_period (int): last time step of training period
        """
        self.df_pred = df_pred
        self.df_eval = df_eval
        self.horizon = horizon
        self.train_period = train_period
        self.time_cols = [f'd_{i+1}' for i in range(self.train_period)] # +1 because python is 0 indexed
        self.pred_cols = [f'd_{train_period + i + 1}' for i in range(horizon)] # +1 because python is 0 indexed
        return None


class RMSSE(BaseEvaluator):
    def __init__(self, df_pred, df_eval, horizon, train_period) -> None:
        super().__init__(df_pred, df_eval, horizon, train_period)
        self.numer = self.calculate_numerator()
        self.denom = self.calculate_denominator()
        self.rmsse = self.calculate_rmsse(self.numer, self.denom)


    def _get_non_zero_periods(self, train_periods: np.array) -> np.array:
        first_occurrence = (train_periods!=0).argmax(axis=1)
        return self.train_period - first_occurrence

    def calculate_denominator(self) -> np.array:
        train_periods = self.df_pred[self.time_cols].copy().to_numpy()
        nonzero_periods = self._get_non_zero_periods(train_periods)

        previous_values = np.roll(train_periods, 1)[:, 1:]
        current_values = train_periods[:, 1:]
        sq_err = np.square(current_values - previous_values)
        denominator = np.sum(sq_err, axis=1) / nonzero_periods
        return denominator

    def calculate_numerator(self) -> np.array:
        y_hat = self.df_pred[self.pred_cols].copy().to_numpy()
        y = self.df_eval[self.pred_cols].copy().to_numpy()
        sq_err = np.square(y_hat - y)
        numerator = np.sum(sq_err, axis=1) / self.horizon
        return numerator

    def calculate_rmsse(self, numer, denom) -> float:
        rmsse = np.sqrt(numer/denom)
        return rmsse


class WRMSSE(RMSSE):
    def __init__(self, df_pred, df_eval, horizon, train_period, df_weights):
        super().__init__(df_pred, df_eval, horizon, train_period)
        self.df_weights = df_weights
