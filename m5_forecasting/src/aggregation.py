import pandas as pd
import numpy as np

from m5_forecasting.definitions import level_dict, series_dict


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


def calculate_hierarchy(df_pred: pd.DataFrame, groupby_cols: list, pred_cols: list):
    val_cols = pred_cols + ['sales']
    if groupby_cols is not None:
        df_out = df_pred.groupby(groupby_cols)[val_cols].sum().reset_index()
    else:
        df_out = pd.DataFrame(df_pred[val_cols].sum()).T
    return df_out


def calculate_weights(df_pred: pd.DataFrame, groupby_cols: list, pred_cols: list):
    df_hierarchy = calculate_hierarchy(df_pred, groupby_cols, pred_cols)
    total_sales = df_hierarchy['sales'].sum()
    df_hierarchy['weights'] = df_hierarchy['sales'] / total_sales
    return df_hierarchy