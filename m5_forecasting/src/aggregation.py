from ast import List
from os import listdir
import pandas as pd
import numpy as np


def calculate_sales(
    df_prices: pd.DataFrame, 
    df_actuals: pd.DataFrame,
    pred_cols: list
    ) -> pd.DataFrame:
    """Function to calculate sales over the evaluation period

    Args:
        df_price_calendar (pd.DataFrame): Pandas DataFrame containing prices for each period for each item/ store/ state
        df_actuals (pd.DataFrame): Pandas DataFrame containing actual sale volumes for each period for each item/ store/ state
        pred_cols (list): Pandas DataFrame containing sale prices for each period for each item/ store/ state

    Returns:
        pd.DataFrame: Pandas DataFrame containing total sales for each item/ store/ state
    """
    id_cols = ['id', 'item_id', 'store_id', 'state_id']
    merge_cols = ['item_id', 'store_id', 'period']

    df_volume = df_actuals[id_cols+pred_cols].copy()
    df_volume = pd.wide_to_long(df_volume, stubnames='d_', i=id_cols, j='period').reset_index().rename(columns={'d_':'volume'})
    df_volume['period'] = 'd_' + df_volume['period'].astype(str)

    df_sales = df_volume.merge(df_prices[merge_cols+['sell_price']], on=merge_cols, how='left')
    df_sales = df_sales.fillna(0)
    df_sales['sales'] = df_sales['volume'] * df_sales['sell_price']
    df_sales = df_sales.groupby(id_cols)['sales'].sum().reset_index()
    return df_sales
