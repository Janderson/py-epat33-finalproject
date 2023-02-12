import pandas as pd                # greaty library to work on with data
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import warnings
from sklearn.model_selection import train_test_split

from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm

import itertools as it # used to optmization


import warnings
warnings.filterwarnings("ignore")


"""
#####################################################################################################
##                                                                                                 ##
##                 FUNCTION TO WORK WITH LISTS OF STOCKS                                           ##
##                                                                                                 ##
#####################################################################################################
"""


banking_stocks = ["ITUB4","ITUB3","BBDC4","BBDC3","BBAS3"]
gas_oil_stocks = ["VALE3","PETR4","PETR3","USIM5","UGPA3"]
services_stocks = ["ABEV3", "CIEL3", "LREN3"]
others_stocks = ["GGBR4", "GOAU4", "ITSA4", "B3SA3"]

# build a list of all stocks will data imported
stock_list = []
stock_list.extend(banking_stocks)
stock_list.extend(gas_oil_stocks)
stock_list.extend(services_stocks)
stock_list.extend(others_stocks)

# make a list of distinct objects
stock_list = list(set(stock_list))

same_company = [
    ('PETR3', 'PETR4'),
    ('ITUB3', 'ITUB4'),
    ('ITSA4', 'ITUB4'),
    ('GGBR4', 'GOAU4'),
]





"""
    function: load_and_cleanup_from_mt5
    Function to load and clean historical data
    
"""
def load_and_cleanup_from_mt5(asset, prices_df):
    df = pd.read_csv("data/mt5_d1/{}.csv".format(asset), delimiter="\t")
    series_close = df[["<CLOSE>", "<DATE>"]]
    series_close.columns= [asset, "date"]
    series_close.date = pd.to_datetime(series_close.date.str.replace(".", "-"))
    if len(prices_df.columns)==0:
        prices_df = series_close
    else:
        prices_df = pd.merge(prices_df, series_close, how='outer', on='date')
    #prices_df[asset] = series_close
    return prices_df


def load_data():
    prices_df = pd.DataFrame()

    # function import all list
    for asset in stock_list:
        prices_df = load_and_cleanup_from_mt5(asset, prices_df)

    prices_df=prices_df.set_index(["date"])
    # the dataframe will contain all prices of stocks analysed




def build_entire_list(list_random):
    end_list = []
    for key, stock_a in enumerate(list_random):
        for i in range(key, len(list_random)):
            stock_b = list_random[i]
            if stock_a != stock_b:
                end_list.append((stock_a, stock_b))
    return end_list



"""
    function: cointegration_to_df
        # convert a stock_list with dataframe price into a dataframe with correlations
        convert a stock_list with dataframe price into a dataframe with coin
        columns important:
            - dickey_pvalue: return the p-value of cointegration test using augmented dickey fuller
            - eagle_pvalue: return the p-value of cointegration test using eagle gangler 
"""
def cointegration_to_df(stock_list, df):
    res = []
    for i,j in build_entire_list(stock_list):
        prices_A, prices_B = df[i], df[j]
        X = sm.add_constant(prices_B)
        model = sm.OLS(prices_A, X, missing = 'drop').fit()
        beta = model.params[1]
        sprd = prices_A - beta * prices_B
        p_dickey = adfuller(sprd)[1]
        c_res = coint(prices_A, prices_B)
        res.append({
            "A": i, "B": j, 
            "t_value": c_res[0], 
            "p5":c_res[2][1], 
            "dickey_pvalue": p_dickey, 
            "eagle_pvalue": c_res[1], 
            "beta": beta,
        })
    return pd.DataFrame(res)


def correl_to_df(stock_list, df):
    res = []
    for i,j in build_entire_list(stock_list):
        res.append({
            "A": i, "B": j, "correlation": np.corrcoef(df[i], df[j])[0][1]
        })
    return pd.DataFrame(res)



"""
#####################################################################################################
##                                                                                                 ##
##                 FUNCTION TO MAKE A BACKTEST IN A PANDAS DATAFRAME                               ##
##                                                                                                 ##
#####################################################################################################
"""

"""
    function: back_build_df
        create a dataframe with price_A and price_B, to after that create a backtest on it.
        return: dataframe with "A" and "B" close price of 
"""
def back_build_df(prices_A, prices_B, index):
    backtest_df = pd.DataFrame(
        {"A":prices_A, 
         "B": prices_B}
    )
    backtest_df.set_index(index)
    return backtest_df


def back_buildratio(
    df, 
    devpad=2, 
    loopback_period=20, 
    use_log = False):

    # construction of pair ratio
    if use_log:
        df["ratio"] = np.log(df.A/df.B)
    else:
        df["ratio"] = np.log(df.A/df.B)
        

    # construction of zScore
    df["mean_ratio"] = df.ratio.rolling(loopback_period).mean()
    df["upperline"] = df.mean_ratio + (df.ratio.rolling(loopback_period).std() * devpad)
    df["lowerline"] = df.mean_ratio - (df.ratio.rolling(loopback_period).std() * devpad)
    df["zscore"] = (df.ratio - df.mean_ratio) / df.ratio.rolling(loopback_period).std()
    return df


"""
    function: back_filters
        this function create a column to put a value of test, like a 
        cointegration test using adf and a final column
        - adf_loopback_1 = 0 means adf test disable
        - corr_period_1 = 0 means correlation test disable
        
    return: put a column into dataframe called filter, when filter = True, 
        all filter choose by user are ok, and we can enter on the marketing
"""
def back_filters(backtest_df, **kwargs):
    
    # get parameters
    adf_loopback_1 = kwargs.get('adf_loopback_1', 25)
    adf_trigger = kwargs.get('adf_trigger', 0.05)
    corr_period_1 = kwargs.get('corr_period_1', 20)
    corr_trigger = kwargs.get('corr_trigger', .60)
    
    backtest_df["filter"] = True

    # function to help rolling apply build the filters
    def check_coint_d(idx, cuttoff=0.01):
        d1 = backtest_df[["A","B"]].iloc[idx]
        prices_A, prices_B = d1["A"], d1["B"]
        X = sm.add_constant(prices_B)
        model = sm.OLS(prices_A, X, missing="drop").fit()
        beta = model.params[1]
        sprd = prices_A - beta * prices_B
        p_dickey = adfuller(sprd)[1]
        return p_dickey

    def check_stati(rows, cuttoff=0.05):
        rows = rows[~np.isnan(rows)] # drops na's using np
        return (adfuller(rows,2)[1] < cuttoff)


    # filter to augmented dickey fuller
    if adf_loopback_1!=0:
        backtest_df["filter_adf_1"] =  pd.rolling_apply(
                                        np.arange(len(backtest_df)), adf_loopback_1, check_coint_d)
                
    else:
        backtest_df["filter_adf_1"] = adf_trigger

    # filter using correlation
    if corr_period_1!=0:
        backtest_df["corr_calc"] = pd.rolling_corr(
                                        backtest_df.A, backtest_df.B, corr_period_1)
        backtest_df["filter_corr_1"] = backtest_df.corr_calc > corr_trigger
    else:
        backtest_df["filter_corr_1"] = corr_trigger
    backtest_df["filter"] = (
                        backtest_df.filter_adf_1 <= adf_trigger) & (
                        backtest_df.filter_corr_1 >= corr_trigger)

    return backtest_df



def back_signals(df, z_entry_threshold = 1, z_exit_threshold = 0.5, z_stop_threshold= 2):
    sign_entry_none = 0
    sign_entry_buysell = 1
    sign_entry_sellbuy = -1
    sign_exit = 2
    sign_stop = -2
    df["signal"] = 0
    df["status"] = 0
    df["signal_exit"] = 0
    df["entry_priceA"] = 0.0
    df["entry_priceB"] = 0.0
    df["pnl"] = 0.0
    
    if not 'filter' in df.columns:
        df['filter'] = True
        
    # build entry signal
    df.signal = np.where( (df.zscore>z_entry_threshold) & df['filter'] , 1, 0)
    df.signal = np.where( (df.zscore<-z_entry_threshold) & df['filter'], -1, df.signal)

    # build exit/stop signal
    df.signal_exit = np.where(abs(df.zscore)<z_exit_threshold, 2, 0)
    df.signal_exit = np.where(abs(df.zscore)>z_stop_threshold, -2, df.signal_exit)



    status_vls = df.status.values
    prA_vls = df.entry_priceA.values
    prB_vls = df.entry_priceB.values

    def assign_vls(i, act=True):
        if act:
            prA_vls[i], prB_vls[i] = df.A.ix[i], df.B.ix[i]
        else:
            prA_vls[i], prB_vls[i] = prA_vls[i-1], prB_vls[i-1]
            
    for i, st in enumerate(status_vls):

        
        # variables to more convinience
        last_status = status_vls[i-1]
        sign_i = df.signal.ix[i]
        sign_exit_i = df.signal_exit.ix[i]

        # entry BUY pair
        if sign_i > 0 and sign_exit_i ==0 and last_status in [0,1]:
            status_vls[i] = 1
            if last_status==0: 
                assign_vls(i)
            else:
                assign_vls(i, False)
                
        # entry SELL pair
        elif sign_i<0 and sign_exit_i ==0 and last_status in [-1,0]:
            status_vls[i] = -1
            if last_status==0: 
                assign_vls(i)
            else:
                assign_vls(i, False)
                
        # exit buy pair
        elif df.signal_exit[i]!=0 and last_status==1:
            status_vls[i] = 2
            assign_vls(i, False)
        # exit sell pair
        elif df.signal_exit[i]!=0 and last_status==-1:
            status_vls[i] = -2
            assign_vls(i, False)
        elif status_vls[i-1] in [-1,1]:
            status_vls[i] = status_vls[i-1]
            assign_vls(i, False)



    df.entry_priceA = prA_vls
    df.entry_priceB = prB_vls
    df.status = status_vls

    # status=2: Exit buyPair signal=1
    # status=-2: Exit sellPair signal=-1

    conditions = [
        (df.status==2),
        (df.status==-2),
    ]
    choices = [
        (df.entry_priceA-df.A) + (df.B - df.entry_priceB),
        (df.A - df.entry_priceA) + (df.entry_priceB - df.B)
    ]
    df['pnl'] = np.select(conditions, choices, default=0)


    return df


#insample = back_complete(insample_df, "PETR3", "PETR4", z_entry=1.5, z_exit=0.5)
#outsample = back_complete(outsample_df, "PETR3", "PETR4", z_entry=1.5, z_exit=0.5)


"""
    function: opt_signals
        this function create a column to put a value of test, like a 
        cointegration test using adf and a final column
        - adf_loopback_1 = 0 means adf test disable
        - corr_period_1 = 0 means correlation test disable
    parameters:
        - backtest_df: dataframe with Price A and Price B
        - fake: if pass true none backtest will made.
    return: put a column into dataframe called filter, when filter = True, 
        all filter choose by user are ok, and we can enter on the marketing
"""
def opt_signals(backtest_df, fake=False, **kwargs):
    
    debug = kwargs.get('debug', False)
    
    if debug: print("debugging opt_signals: {} \r\n{}".format(backtest_df.tail(2), kwargs))

    # get parameters of kwargs
    z_entry_range = kwargs.get('z_entry_range', [1, 1.5, 2])
    z_exit_range = kwargs.get('z_exit_range', [0.5])
    z_stop_range = kwargs.get('z_stop_range', [2.5, 3])
    loopback_range = kwargs.get('loopback_range', [20]) 
    devpad_range = kwargs.get('devpad_range', [2]) 
    adf_range = kwargs.get('adf_range', [0, 50]) 
    corr_range = kwargs.get('corr_range', [.60]) 
    
    
    metrics = kwargs.get('metrics', ["return", "cagr", "sharpe_ratio", 
                                     "hit_ratio", "trades"])
    objective_function = metrics[0]
    
    results = []
    better_index = -1
    for comb in it.product(z_entry_range, z_exit_range, z_stop_range,
                           loopback_range, devpad_range, adf_range, corr_range):
        
        # get all variables of list optization
        z_entry, z_exit, z_stop, entry_loopback, entry_devpad, filter_adf, filter_corr = comb

        opt_itens = { 
            "z_entry":z_entry, 
            "z_exit":z_exit, 
            "z_stop":z_stop,
            "entry_loopback":entry_loopback, 
            "entry_devpad":entry_devpad,
            "filter_adf":filter_adf,
            "filter_corr":filter_corr
        }

        if debug: print(("backtest zentry:{} zexit:{} z_stop:{} m:{} d:{}").format(
                    z_entry, z_exit, z_stop, entry_loopback, entry_devpad)
        )

        if not fake:

            # we just copy the main dataframe, to avoid make unnecessary calculation
            opt_df = backtest_df.copy()

            # make an entire backtest 
            opt_df = back_buildratio(opt_df, loopback_period=entry_loopback, 
                                     devpad=entry_devpad)
            opt_df = back_filters(opt_df, adf_loopback_1=filter_adf,
                                corr_loopback_1=filter_corr)

            opt_df = back_signals(opt_df, z_entry, z_exit, z_stop)

            calc_metrics = back_get_stats(opt_df)

            # pass metric to optimization item of dict
            for m in metrics: opt_itens[m] = calc_metrics[m]
        # add itens to a result list
        results.append(opt_itens)
        results_df = pd.DataFrame(results)
        results_df.sort_values([objective_function], ascending=False, inplace=True)
        
    return better_index, results_df


"""
    function: back_complete
        This function package all function to make a complete backtest to one single pair
    inputs:
        backtest_df: dataframe with prices (usually prices_df) to make backtest
        
    returns:
        a dict with statistcs.
        Be careful not return the backtest_df
"""
def back_complete(backtest_df, stockA, stockB, **kwargs):

    debug = kwargs.get('debug', False)
    if debug: print("debugging back_complete: {}<-->{} \r\n{}".format(stockA, stockB, kwargs))

    # get parameters from kwargs
    entry_devpad = kwargs.get('entry_devpad', 2)
    entry_loopback = kwargs.get('entry_loopback', 20)
    z_entry = kwargs.get('z_entry', 1)
    z_exit = kwargs.get('z_exit', 0.5)
    z_stop = kwargs.get('z_stop', 3)
    adf_loopback_1 = kwargs.get('adf_loopback_1', 25)
    corr_period_1 = kwargs.get('corr_period_1', 20)
    
    if debug: print("dp:{} lb:{} z_en:{} z_ex:{} z_st:{} adf:{} corr:{}".format(
                                                entry_devpad, 
                                                entry_loopback, 
                                                z_entry, z_exit, z_stop,
                                                adf_loopback_1,corr_period_1))
    # make a entire backtest
    df = back_build_df(backtest_df[stockA], backtest_df[stockB], 
                       backtest_df.index)

    
    df = back_buildratio(df, devpad=entry_devpad, loopback_period=entry_loopback)
    df = back_filters(df, adf_loopback_1=adf_loopback_1, 
                      corr_period_1=corr_period_1)

    # now i will pass trougth the optimized parameters
    df = back_signals(df, z_entry_threshold=z_entry, 
                      z_exit_threshold=z_exit, z_stop_threshold=z_stop)
    stats = back_get_stats(df)
    stats["A"] = stockA
    stats["B"] = stockB
    return df, stats



"""
    function: back_inout_complete
        this function run a optmization in a insample part of data, 
        after that took the best group of parameters and run against the out-sample
        and collect the statistics
    parameters:
        - insample_df: dataframe with Price A and Price B
        - outsample_df: if pass true none backtest will made.
        - stockA:
        - stockB: 
    return: 
        - in_df: dataframe of backtest the best parameter in insample
        - out_df: dataframe of backtest the best parameter in outsample
        - insample_stats: statitics of backtest the best parameter in insample  
        - outsample_stats: statitics of backtest the best parameter in outsample  
"""

def back_inout_complete(insample_df, outsample_df, stockA, stockB, **kwargs) :
    
    debug = kwargs.get('debug', False)
    if debug: print("debugging back_inout_complete: {}<-->{} \r\n{}".format(stockA, stockB, kwargs))

    # optimizate the parameters into in-sample data
    in_df = back_build_df(insample_df[stockA], insample_df[stockB], insample_df.index)
    
    better , returns_df = opt_signals(in_df, **kwargs)

    # just the first of dataframe, because is ordered by objective_function
    best_parameters = returns_df.ix[0]
    if debug: print("better:",better)
    if debug: print("best choose parameters: {} ".format(best_parameters))

    
    # run the backtest into out-sample period to get some statistics

    out_df, outsample_stats = back_complete(outsample_df, stockA, stockB, 
                                          entry_devpad = best_parameters.get("entry_devpad"), 
                                          entry_loopback = int(best_parameters.get("entry_loopback")),
                                          z_entry = best_parameters.get("z_entry"), 
                                          z_exit = best_parameters.get("z_exit"), 
                                          z_stop = best_parameters.get("z_stop")
                                           )

    # run the back again into insample period to deliver plot function
    # the pnl curve
    in_df, insample_stats = back_complete(insample_df, stockA, stockB, 
                                          entry_devpad = best_parameters.get("entry_devpad"), 
                                          entry_loopback = int(best_parameters.get("entry_loopback")),
                                          z_entry = best_parameters.get("z_entry"), 
                                          z_exit = best_parameters.get("z_exit"), 
                                          z_stop = best_parameters.get("z_stop")
                                         )

    return in_df, out_df, insample_stats, outsample_stats, returns_df




"""
    function: back_complete
        This function package all function to make a complete backtest to one single pair
    inputs:
        backtest_df: dataframe with prices (usually prices_df) to make backtest
        
    returns:
        a dict with statistcs.
        Be careful not return the backtest_df
"""
import math
def back_get_stats(backtest_df, start_acc=1, N=252):
    
    cum_returns = float(backtest_df.pnl.cumsum()[-1:])
    stats = {
        "trades_win" : int(backtest_df[backtest_df.pnl>0].pnl.count()),
        "trades_loss" : int(backtest_df[backtest_df.pnl<0].pnl.count()),
        "final_acc" : start_acc+cum_returns,
        "start_acc" : start_acc
    }
    # CALC -  qtd of trades
    stats["trades"] = stats["trades_win"] + stats["trades_loss"]

    # CALC - win ratio
    if (stats["trades"])>0:
        stats["hit_ratio"] =  stats["trades_win"] / stats["trades"]
    else:
        stats["hit_ratio"] = 0

    # CALC - sharp ratio
    if backtest_df.pnl.std()>0:
        stats["sharpe_ratio"] = (backtest_df.pnl.mean()) / backtest_df.pnl.std()
    else:
        stats["sharpe_ratio"] = 0
    
    # CALC - CAGR
    final_acc = stats["final_acc"] 

    # num of years
    # https://stackoverflow.com/questions/36090917/calculating-the-number-of-years-in-a-pandas-dataframe
    nyear = int(len(np.unique(backtest_df.index.year)))
    
    if (stats["trades"])>0:
        stats["cagr"] = np.power( (final_acc/start_acc) , (1/nyear)) - 1
    else:
        stats["cagr"] = 0
    
    # CALC - % Return
    stats["return"] = (final_acc - start_acc) / start_acc
    return stats








#z_entry_threshold_range = [1.5,2,2.5], z_exit_threshold_range = [0.5, 1],z_stop_threshold_range = [2,3], entry_means_range = [20], entry_devpad_range = [2],objective_function=["cagr", "sharpe_ratio", "hit_ratio", "trades"], debug=False)




    
def back_inoutsample_from_list(list_stocks, insample_df, outsample_df):
    results = []
    for row in list_stocks.iterrows():
        stock_A, stock_B = row[1]["A"], row[1]["B"]
        in_df, out_df, in_stats, out_stats = back_inout_sample_complete(insample_df, outsample_df, stock_A, stock_B)
        results.append(out_stats)

    return results




"""
#####################################################################################################
##                                                                                                 ##
##                 FUNCTION TO WORK PLOT USING DATAFRAME                                           ##
##                                                                                                 ##
#####################################################################################################
"""
def printlog(*s):
    with open("l.log", "a") as f:
        f.write("{}\n".format(s))


def plot_optimization(results_df, param_a= "z_entry", param_b="z_exit", metric="return", format_str = "{:06.2f}", title_str="Return Heatmap"):
    df_r = results_df
    optsA, optsB = df_r[param_a].value_counts().keys(), df_r[param_b].value_counts().keys()
    data = np.zeros((len(optsA), len(optsB)))
    for key_a, vl_a in enumerate(optsA):
        for key_b, vl_b in enumerate(optsB):
            #print(key_a , key_b, "--->", vl_a, ":", vl_b)
            data[key_a][key_b] = list(df_r[(df_r[param_a] == vl_a)  & (df_r[param_b] == vl_b)][metric])[0]


    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Reds)
    row_labels = optsB
    column_labels = optsA
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            plt.text(x + 0.5, y + 0.5, '{}'.format(data[y, x]),
            horizontalalignment="center",
            verticalalignment="center",
    )
    plt.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_xticklabels("parameter: {}".format(row_labels), minor=False)
    ax.set_yticklabels("parameter: {}".format(column_labels), minor=False)
    plt.suptitle(title_str, fontsize=18)
    plt.xlabel(param_a, fontsize=14)
    plt.ylabel(param_b, fontsize=14)

def print_results(dict_stats):
    r = pd.DataFrame(dict_stats, index=[0])
    r=r[["A", "B","trades", "hit_ratio", "cagr", "sharpe_ratio", "return"]]
    r.columns=["Stock A", "Stock B", "Total Trades", "Hit Ratio", "CAGR", "Sharpe Ratio", "Return"]

    
def plot_ratio(backtest_df):
        # plot all lines in same chart
    #fig.
    backtest_df.ratio.plot(title="Pair Ratio")
    ax = backtest_df["mean_ratio"].plot()
    backtest_df.upperline.plot()
    backtest_df.lowerline.plot()
    dates = backtest_df[ ((backtest_df.zscore>2) & (backtest_df.zscore.shift(-1)<2)) |  ((backtest_df.zscore<-2) & (backtest_df.zscore.shift(-1)>-2)) ].index
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=dates, ymin=ymin, ymax=ymax, color='b', linestyle='--')

    
    
def plot_zscore(backtest_df):
    ax = backtest_df.zscore.plot()
    plt.axhline(backtest_df.zscore.mean(), color='black', linestyle='-')
    plt.axhline(2.0, color='red', linestyle='--')
    plt.axhline(-2.0, color='green', linestyle='--')
    plt.legend(['z-score', 'Mean', '+2', '-2'])



   
def plot_insample_outsample(insample_df, out_sample_df, stats_insample):
    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig = plt.figure(figsize=(20, 6))
    insample_df.pnl.cumsum().plot(ax=ax[0])
    out_sample_df.pnl.cumsum().plot(ax=ax[0], color="y")
    insample_df.pnl.plot(ax=ax[1])
    out_sample_df.pnl.plot(ax=ax[1], color="y")
    
    ax[0].set_title("{}<=>{} in-out sample backtest".format(stats_insample["A"], stats_insample["B"]))
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Cumulative Returns')
    ax[1].set_ylabel('Daily Returns')
    ax[0].plot([0], [0], 'o')
    dates = [out_sample_df.index[0]]
    ymin, ymax = ax[0].get_ylim()
    ax[0].vlines(x=dates, ymin=ymin, ymax=ymax+1, color='r')
    ax[1].vlines(x=dates, ymin=ymin, ymax=ymax+1, color='r')
    

def plot_insample_outsample_simple(insample_df, out_sample_df, stats_insample):
    ax = insample_df.pnl.cumsum().plot(title="{}<=>{} in-out sample backtest".format(stats_insample["A"], stats_insample["B"]))
    out_sample_df.pnl.cumsum().plot()
    dates = [out_sample_df.index[0]]
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=dates, ymin=ymin, ymax=ymax+1, color='r')
    
    

	
############
##		LIST 
############


if __name__ == "__main__":
    insample_df, outsample_df = train_test_split(prices_df, test_size=0.25, shuffle=False, stratify=None)
    #print(prices_df.head(1).index)
    df = back_build_df(insample_df.PETR3, insample_df.PETR4, insample_df.index)
    df = back_buildratio(df, 2, 20)
    df = back_filters(df, adf_loopback_1=80)
    df = back_signals(df, z_entry_threshold=2, z_exit_threshold=1, z_stop_threshold=3)

    stats = back_get_stats(df)
    print(pd.DataFrame(stats, index=[0]))
    df.to_csv("teste.csv", sep=";", decimal=",")
