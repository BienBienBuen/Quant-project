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
    return pd.DataFrame(stock_data)

start_date = datetime.datetime(2013,1,1)
end_date = datetime.datetime(2024,6,1)
slope = 1.1153621020607556
intercept = -0.6439426102796257

if __name__ == "__main__":
    SGD = download_data('SGD=X', start_date, end_date)
    EUR = download_data('EUR=X', start_date, end_date)
    residues = EUR - slope * SGD - intercept 
    plt.plot(residues)
    plt.show()

