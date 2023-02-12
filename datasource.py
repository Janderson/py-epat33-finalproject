import pandas as pd
from abc import ABC
import abc


def load_and_cleanup_from_mt5(asset, prices_df):
    df = pd.read_csv(f"data/MT5D1_{asset}.csv", delimiter="\t")
    series_close = df[["<CLOSE>", "<DATE>"]]
    series_close.columns = [asset, "date"]
    series_close.date = pd.to_datetime(series_close.date.str.replace(".", "-"))
    if len(prices_df.columns)==0:
        prices_df = series_close
    else:
        prices_df = pd.merge(prices_df, series_close, how='outer', on='date')
    return prices_df


class DataSource(ABC):
    def __init__(self):
        self.prices_df = pd.DataFrame()

    def load(self):
        pass


class DataSourceMT5(DataSource):
    def __init__(self):
        self.prices_df = pd.DataFrame()

    def load(self, asset):
        self.prices_df = load_and_cleanup_from_mt5(asset, self.prices_df)
    
    def fix_index(self):
        self.prices_df.set_index(["date"], inplace=True)
    
    def get(self, asset):
        return self.prices_df[asset]


class DataSourcePyMT5(DataSource):
    pass


def select_datasource():
    return DataSourceMT5()
