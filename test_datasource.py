import pandas as pd
from datasource import load_and_cleanup_from_mt5, select_datasource, DataSourceFake
from datetime import datetime

def load_mt5_df(asset):
    return pd.read_csv(f"data/MT5D1_{asset}.csv")


def test_unit_temp_load_and_cleanup_from_mt5_1():
    stock_a = "BBAS3"
    df_first = load_mt5_df(stock_a)

    datasource = select_datasource()
    datasource.load(stock_a)

    assert df_first.shape == (2509, 1)
    assert datasource.prices_df.shape == (df_first.shape[0], 2)
    assert stock_a in datasource.prices_df.columns

def test_unit_temp_load_and_cleanup_from_mt5_2():
    stock_a = "BBAS3"
    stock_b = "ABEV3"
    dataframe_a = load_mt5_df(stock_a)
    dataframe_b = load_mt5_df(stock_b)

    datasource = select_datasource()
    datasource.load(stock_a)
    datasource.load(stock_b)

    dataframe_observed = datasource.prices_df
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
    data_source = select_datasource()

    for stock in stock_list:
        data_source.load(stock)

    all_in_prices_df = [stock in data_source.prices_df.columns 
                        for stock in stock_list]
    assert all(all_in_prices_df)


class TestFakeDataSource:
    def test_load_asset(self):
        datasource = DataSourceFake()
        stock = "fakestock"
        prices = [
            {"date": datetime(year=2022, month=10, day=5), "price": 10.5},
            {"date": datetime(year=2022, month=10, day=6), "price": 11},
            {"date": datetime(year=2022, month=10, day=7), "price": 12},
            {"date": datetime(year=2022, month=10, day=8), "price": 15},
        ]
        datasource.loadprices(stock, prices)

        assert datasource.prices_df.shape == (4, 2)
        assert stock in datasource.prices_df

    def test_load_asset_B(self):
        datasource = DataSourceFake()
        stock_a = "fakestock"
        prices_a = [
            {"date": datetime(year=2022, month=10, day=5), "price": 10.5},
            {"date": datetime(year=2022, month=10, day=6), "price": 11},
            {"date": datetime(year=2022, month=10, day=7), "price": 12},
            {"date": datetime(year=2022, month=10, day=8), "price": 15},
        ]
        stock_b = "fakestock_b"
        prices_b = [
            {"date": datetime(year=2022, month=10, day=5), "price": 100},
            {"date": datetime(year=2022, month=10, day=6), "price": 101},
            {"date": datetime(year=2022, month=10, day=7), "price": 102},
            {"date": datetime(year=2022, month=10, day=8), "price": 103},
            {"date": datetime(year=2022, month=10, day=9), "price": 104},
        ]

        datasource.loadprices(stock_b, prices_b)
        datasource.loadprices(stock_a, prices_a)

        assert datasource.prices_df.shape == (max(len(prices_a), len(prices_b)), 3)
        assert stock_a in datasource.prices_df
        assert stock_b in datasource.prices_df        