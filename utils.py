import yfinance as yf
from iexfinance.stocks import Stock
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import linregress
from arch import arch_model


"""utility function"""

def download_data(stock, start, end):
    stock_data = {}
    ticker = yf.download(stock, start,end, progress=False)
    stock_data['price'] = ticker['Adj Close']
    return pd.DataFrame(stock_data)


#minute frequency data
def download_intraday_data(stock):
    stock_data = {}
    today = datetime.now()
    seven_days_ago = today - timedelta(days=7)

    ticker = yf.download(stock, seven_days_ago, today, progress=False, interval='1m')
    stock_data['price'] = ticker['Adj Close']
    return pd.DataFrame(stock_data)


def filter_non_null_rows(dfs:list, tickers:list):
    # Create a mask for rows that are not null in all DataFrames
    mask = pd.concat([df.notnull() for df in dfs], axis=1).all(axis=1)
    
    # Filter each DataFrame using the mask
    filtered_dfs = [df[mask] for df in dfs]
    concatenated_df = pd.concat(filtered_dfs, axis=1)
    concatenated_df.columns = tickers
    return concatenated_df


def perform_regression(data1, data2):
    slope, intercept, r_value, p_value, std_err = linregress(data1, data2)
    residuals = data2 - slope * data1 - intercept
    return residuals, slope, intercept, r_value, p_value, std_err


if __name__ == "__main__":
    data = download_intraday_data('SGD=X')
    print(data)
    # data = download_data('AAPL', '2020-01-01', '2020-01-10')
    # print(data)