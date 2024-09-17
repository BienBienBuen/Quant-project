import yfinance as yf
from iexfinance.stocks import Stock
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nasdaqdatalink as ndl
import datetime

"""using nasdaqdatalink"""
def data_to_csv(ticker, start, end):
    ndl.ApiConfig.api_key = "nsrdu2JfgqSVB4Czoca6"
    table = ndl.get_table('QDL/BITFINEX', date={'gte': start, 'lte': end} , code=ticker)
    print(type(table))
    print(table)



if __name__ == "__main__":
    print("Hello, World!")
    data_to_csv('BTCUSD', datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 31))


