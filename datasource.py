import pandas as pd
from abc import ABC
import abc


def load_mt5_file(asset):
    df = pd.read_csv(f"data/MT5D1_{asset}.csv", delimiter="\t")
    series_close = df[["<CLOSE>", "<DATE>"]]
    series_close.columns = [asset, "date"]
    series_close.date = pd.to_datetime(series_close.date.str.replace(".", "-"))
    return series_close


def load_and_cleanup_from_mt5(asset, prices_df):
    series_close = load_mt5_file(asset)
    if len(prices_df.columns)==0:
        prices_df = series_close
    else:
        prices_df = pd.merge(prices_df, series_close, how='outer', on='date')
    return prices_df


class DataSource(ABC):
    def __init__(self):
        self._prices_df = pd.DataFrame()
        self._filtered_prices_df = None

    @property
    def prices_df(self):
        if isinstance(self._filtered_prices_df, pd.DataFrame):
            return self._filtered_prices_df
        return self._prices_df

    def load(self):
        pass

    def merge_pricedf(self, dataframe):
        if len(self._prices_df.columns)==0:
            self._prices_df = dataframe
        else:
            self._prices_df = pd.merge(self._prices_df,
                                      dataframe,
                                      how='outer', on='date')

    def filter_by_year(self, start_year, end_year):
        self._filtered_prices_df = self.prices_df \
                                       .set_index(["date"]) \
                                       .loc[str(start_year):str(end_year)] \
                                       .reset_index()

    def get(self, asset):
        return self.prices_df[asset]


class DataSourceMT5(DataSource):
    def load(self, asset):
        self.merge_pricedf(load_mt5_file(asset))


class DataSourcePyMT5(DataSource):
    pass


def select_datasource():
    return DataSourceMT5()


class DataSourceFake(DataSource):
    def loadprices(self, asset, prices):
        df = pd.DataFrame(prices)
        df.columns = ["date", asset]
        self.merge_pricedf(df)
        return df
