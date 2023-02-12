import click
from datasource import select_datasource
import os


@click.group
def cli():
    pass

@cli.command("test_load")
@click.argument("stocks")
def cmd_load(stocks):
    datasource = select_datasource()
    for stock in stocks.split(","):
        datasource.load(stock)


def year_setup(years_setup):
    return years_setup.split(":")[0], years_setup.split(":")[1]

@cli.command("plot_ratio")
@click.argument("stock_a")
@click.argument("stock_b")
@click.option("--years_setup", "years_setup", default="2013:2013", help="2011:2011")
@click.option("--loopback", "loopback", default="20", help="period of means")
@click.option("--devpad", "devpad", default="2", help="devpad")
def cmd_load(stock_a, stock_b, years_setup, loopback, devpad):
    import backtest_functions
    import matplotlib.pyplot as plt
    datasource = select_datasource()
    datasource.load(stock_a)
    datasource.load(stock_b)


    datasource.filter_by_year(str(year_setup(years_setup)[0]), str(year_setup(years_setup)[1]))
    prices_df = datasource.prices_df
    insample_df = prices_df.loc['2012':'2015']
    outsample_df = prices_df.loc['2016':'2016']    

    backtest_df = backtest_functions.back_build_df(
        prices_A=datasource.get(stock_a),
        prices_B=datasource.get(stock_b),
        index=datasource.prices_df.index
    )

    backtest_df = backtest_functions.back_buildratio(backtest_df, devpad=float(devpad), loopback_period=int(loopback))

    backtest_functions.plot_ratio(backtest_df)
    os.makedirs("results/", exist_ok=True)
    filename = f"results/ratio_{stock_a}_{stock_b}.png"
    plt.savefig(filename)
    backtest_df.to_csv(f"results/pair_{stock_a}_{stock_b}.csv")
    print(f"arquivo salvo: {filename}")


cli()