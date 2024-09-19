import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
from utils import download_data, filter_non_null_rows, perform_regression


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

"""backtest"""
def backtest(start_date:datetime.datetime, end_date:datetime.datetime, ticker_list:list, strat, initial_capital=100000):
    capital_over_time = []
    sharpe_ratio = 0
    max_drawdown = 0

    df_list = [download_data(ticker, start_date, end_date) for ticker in ticker_list]
    filtered_df = filter_non_null_rows(df_list, ticker_list)
    result_df = strat(data=filtered_df, initial_capital=initial_capital)

    capital_over_time = result_df['capital'].values
    daily_returns = np.diff(capital_over_time) / capital_over_time[:-1]
    trading_days = len(filtered_df)

    # Calculate Sharpe Ratio
    risk_free_rate = 0.01  
    excess_returns = daily_returns - risk_free_rate / trading_days 
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(trading_days)  # Annualized

    # Calculate Maximum Drawdown
    peak = np.maximum.accumulate(capital_over_time)
    drawdown = (peak - capital_over_time) / peak
    max_drawdown = np.max(drawdown)

    return result_df, capital_over_time, sharpe_ratio, max_drawdown


"""strategies"""
def inversion_strategy(data:pd.DataFrame, initial_capital:str, ratio=0.5, inspection_window=300):


    column_names = data.columns.tolist()

    # Loop through the DataFrame starting from the inspection_window index
    for i in range(inspection_window, len(data)):
        # Define the window of data for regression
        window_data = data.iloc[i-inspection_window:i]
        # Perform regression to get slope and intercept
        residual, slope, intercept, r_value, p_value, std_err = perform_regression(window_data[column_names[0]], window_data[column_names[1]])

        currency1 = data.iloc[i][column_names[0]]
        currency2 = data.iloc[i][column_names[1]]
        residual_new = currency2 - (slope * currency1 + intercept)

        # Save the residual to the DataFrame
        data.at[data.index[i], 'residual'] = residual_new
    
    data['signal'] = data['residual'].apply(reversion_signal)

    # for i in range(inspection_window, len(data)):
    #     if data['signal'][i] == 1 and data['signal'][i-1] == 0:
    #         #enter long position
    #         portfolio_weight = [ratio, 1-ratio]
    #         weight_dict = {column_names[i]: portfolio_weight[i] for i in range(len(column_names))}
    #         shares = {ticker: (initial_capital * weight)/data[ticker][i] for ticker, weight in weight_dict.items()}
    #         data['capital'][i] = sum([shares[ticker] * data[ticker][i] for ticker in column_names])

    #     if data['signal'][i] == -1 and data['signal'][i-1] == 0:
    #         #enter short position
    #     if data['signal'][i] == 0:
    #         #exit position

    data['capital'] = 1
    return data

def buy_and_hold(data:pd.DataFrame, initial_capital:str):

    #a bit extra but follows the underlying logic
    column_names = data.columns.tolist()

    portfolio_weight = [1/len(column_names) for i in range(len(column_names))]
    weight_dict = {column_names[i]: portfolio_weight[i] for i in range(len(column_names))}
    shares = {ticker: (initial_capital * weight)/data[ticker][0] for ticker, weight in weight_dict.items()}

    data['capital'] = sum([shares[ticker] * data[ticker] for ticker in column_names])
    return data


def momentum(data:pd.DataFrame, initial_capital:str):
    pass

def reversion_signal(residual, threshold=0.025):
    if residual == 0:
        return 0
    elif residual > threshold:
        return 1
    elif residual < -threshold:
        return -1
    else:
        return 0.001

if __name__ == "__main__":
    # start_date = datetime.datetime(2013,1,1)
    # end_date = datetime.datetime(2024,6,1)
    # slope = 0.9131566687899764 
    # intercept = 0.22382541828609814
    # SGD = download_data('AUD=X', start_date, end_date)
    # EUR = download_data('NZD=X', start_date, end_date)
    # residues = EUR - slope * SGD - intercept 
    # plt.plot(residues)
    # plt.show()
    df, capital_over_time, sharpe_ratio, max_drawdown = backtest(datetime.datetime(2013,1,1), datetime.datetime(2024,6,1), ['AUD=X', 'NZD=X'], inversion_strategy)
    print(df['residual'])
    plt.plot(df['residual'])
    plt.plot(df['signal'])
    plt.show()
    print(AUD)

    # print(f'sharpe ratio: {sharpe_ratio}')
    # print(f'max drawdown: {max_drawdown}')

    # fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # currency_columns = df.columns[df.columns.str.contains('=')]  # Filter currency columns
    # for column in currency_columns:
    #     axs[0].plot(df.index, df[column], label=column)
    
    # axs[0].set_title('Currency Exchange Rates')
    # axs[0].set_xlabel('Date')
    # axs[0].set_ylabel('Exchange Rate')
    # axs[0].legend()
    # axs[0].grid()

    # # Plot capital
    # axs[1].plot(df.index, df['capital'], color='green', label='Capital')
    # axs[1].set_title('Capital Over Time')
    # axs[1].set_xlabel('Date')
    # axs[1].set_ylabel('Capital')
    # axs[1].legend()
    # axs[1].grid()

    # output_dir = 'plots'
    # os.makedirs(output_dir, exist_ok=True) 
    # plt.savefig(os.path.join(output_dir, 'buy_and_hold.png'))

    # plt.tight_layout()
    # plt.show()

