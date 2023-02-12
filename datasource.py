import pandas as pd


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


class DataSourceMT5:
    def __init__(self):
        self.prices_df = pd.DataFrame()

    def load(self, asset):
        self.prices_df = load_and_cleanup_from_mt5(asset, self.prices_df)
