import pandas as pd
from datasource import load_and_cleanup_from_mt5, DataSourceMT5


def load_mt5_df(asset):
    return pd.read_csv(f"data/MT5D1_{asset}.csv")


def test_unit_temp_load_and_cleanup_from_mt5_1():
    stock_a = "BBAS3"
    df_first = load_mt5_df(stock_a)
    new_df = pd.DataFrame()
    new_df = load_and_cleanup_from_mt5(stock_a, new_df)
    assert df_first.shape == (2509, 1)
    assert new_df.shape == (df_first.shape[0], 2)
    assert stock_a in new_df.columns

def test_unit_temp_load_and_cleanup_from_mt5_2():
    stock_a = "BBAS3"
    stock_b = "ABEV3"
    dataframe_a = load_mt5_df(stock_a)
    dataframe_b = load_mt5_df(stock_b)
    dataframe_observed = pd.DataFrame()
    dataframe_observed = load_and_cleanup_from_mt5(stock_a, dataframe_observed)
    dataframe_observed = load_and_cleanup_from_mt5(stock_b, dataframe_observed)

    assert dataframe_a.shape == (2509, 1)
    assert dataframe_observed.shape == (dataframe_a.shape[0], 3)
    assert stock_a in dataframe_observed.columns
    assert stock_b in dataframe_observed.columns

def test_usage():
    # function import all list
    prices_df = pd.DataFrame()
    stock_list = ["BBAS3", "ABEV3"]
    for asset in stock_list:
        prices_df = load_and_cleanup_from_mt5(asset, prices_df)

    prices_df=prices_df.set_index(["date"])
    all_in_prices_df = [stock in prices_df.columns 
                        for stock in stock_list]
    assert all(all_in_prices_df)


def test_usage_datasource():
    stock_list = ["BBAS3", "ABEV3"]
    data_source = DataSourceMT5()

    for stock in stock_list:
        data_source.load(stock)

    all_in_prices_df = [stock in data_source.prices_df.columns 
                        for stock in stock_list]
    assert all(all_in_prices_df)
