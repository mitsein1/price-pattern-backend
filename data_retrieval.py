import yfinance as yf
import pandas as pd

def get_historical_data(ticker, start_date, end_date):
    ticker_data = yf.Ticker(ticker)
    data = ticker_data.history(start=start_date, end=end_date)
    return data[['Close']].to_dict()  # Restituisce i dati di chiusura come dizionario
