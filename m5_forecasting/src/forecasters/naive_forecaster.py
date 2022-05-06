import pandas as pd
import numpy as np


class NaiveForecaster():
    def __init__(self, df_train: pd.DataFrame, horizon:int, last_time_stamp: int):
        """Naive Forecaster: Y_t = Y_t-1

        Args:
            df_train (pd.DataFrame): Pandas DataFrame containing training data
            last_time_stamp (int): Last time stamp of training period
            horizon (int): Forecast time horizon
        """
        self.df_train = df_train
        self.last_time_stamp = last_time_stamp
        self.horizon = horizon

        self.train_cols = [f'd_{i+1}' for i in range(self.last_time_stamp)]
        self.pred_cols = [f'd_{self.last_time_stamp+1+i}' for i in range(self.horizon)]

        self.pred = self.fit()
        self.df_pred = self.predict()

    def fit(self):
        x = self.df_train[f'd_{self.last_time_stamp}']
        pred = np.tile(x, (self.horizon, 1)).T
        return pred

    def predict(self):
        df_pred = self.df_train.copy()
        df_pred[self.pred_cols] = self.pred
        print(len(df_pred.columns))
        print(len(self.df_train.columns))
        return df_pred
