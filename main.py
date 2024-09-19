import yfinance as yf
from iexfinance.stocks import Stock
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nasdaqdatalink as ndl
import datetime

"""using nasdaqdatalink"""
# def data_to_csv(ticker, start, end):
#     ndl.ApiConfig.api_key = "nsrdu2JfgqSVB4Czoca6"
#     table = ndl.get_table('QDL/BITFINEX', date={'gte': start, 'lte': end} , code=ticker)
#     print(type(table))
#     print(table)

"""using yfinance"""
def download_data(stock, start, end):
    stock_data = {}
    ticker = yf.download(stock, start,end, progress=False)
    stock_data['price'] = ticker['Adj Close']
    return pd.DataFrame(stock_data).fillna(method='ffill')

start_date = datetime.datetime(2012,12,1)
end_date = datetime.datetime(2012,12,31)
slope = 0.9131566687899764 
intercept = 0.22382541828609814
invested = 0

if __name__ == "__main__":
    AUD = download_data('AUD=X', start_date, end_date)
    NZD = download_data('NZD=X', start_date, end_date)
    print(np.shape(AUD), np.shape(NZD))
    residues = NZD - slope * AUD - intercept 
    plt.plot(residues)
    plt.show()
    print(AUD)

