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
            df_pred (pd.DataFrame): Pandas DataFrame containing predicted values
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
        """Root Mean Squared Scaled Error Evaluator

        Args:
            df_pred (pd.DataFrame): Pandas DataFrame containing predicted values
            df_eval (pd.DataFrame): Pandas Dataframe containing real values
            horizon (int): forecast horizon window
            train_period (int): last time step of the training period
        """
        super().__init__(df_pred, df_eval, horizon, train_period)
        self.numer = self.calculate_numerator()
        self.denom = self.calculate_denominator()
        self.rmsse = self.calculate_rmsse()


    def _get_non_zero_periods(self, train_periods: np.array) -> np.array:
        """Method to get index non-zero periods
        Args:
            train_periods (np.array): sales

        Returns:
            np.array: numpy array containing sales following the first non-zero demand observed
        """
        first_occurrence = (train_periods!=0).argmax(axis=1)
        return self.train_period - first_occurrence

    def calculate_denominator(self) -> np.array:
        """Method to calculate the denominator of RMSSE equation

        Returns:
            np.array: denominator of rmsse
        """
        train_periods = self.df_pred[self.time_cols].copy().to_numpy()
        nonzero_periods = self._get_non_zero_periods(train_periods)

        previous_values = np.roll(train_periods, 1)[:, 1:]
        current_values = train_periods[:, 1:]
        sq_err = np.square(current_values - previous_values)
        denominator = np.sum(sq_err, axis=1) / nonzero_periods
        return denominator

    def calculate_numerator(self) -> np.array:
        """Method to calculate the numerator of RMSSE equation

        Returns:
            np.array: numerator of rmsse
        """
        y_hat = self.df_pred[self.pred_cols].copy().to_numpy()
        y = self.df_eval[self.pred_cols].copy().to_numpy()
        sq_err = np.square(y_hat - y)
        numerator = np.sum(sq_err, axis=1) / self.horizon
        return numerator

    def calculate_rmsse(self) -> float:
        """Method to calculate RMSSE

        Args:
            numer (_type_): RMSSE numerator
            denom (_type_): RMSSE denominator

        Returns:
            float: RMSSE (numpy array) - 1 value for each time-series
        """
        rmsse = np.sqrt(self.numer/self.denom)
        return rmsse


class WRMSSE(RMSSE):
    def __init__(self, df_pred, df_eval, horizon, train_period, df_weights):
        """Weighted Room Mean Square Scaled Error Evaluator

        Args:
            df_pred (pd.DataFrame): Pandas DataFrame containing predicted values
            df_eval (pd.DataFrame): Pandas Dataframe containing real values
            horizon (int): forecast horizon window
            train_period (int): last time step of the training period
            df_weights (pd.DataFrame): DataFrame containing weights for each time-series
        """
        super().__init__(df_pred, df_eval, horizon, train_period)
        self.df_weights = df_weights
        self.wrmsse = self.calculate_wrmsse()

    def calculate_wrmsse(self):
        weights = self.df_weights['weights'].values
        assert weights.shape == self.rmsse.shape
        weigted_rmsse = np.multiply(weights, self.rmsse)
        return weigted_rmsse
