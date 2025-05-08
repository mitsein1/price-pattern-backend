import yfinance as yf
import pandas as pd

def get_historical_data(ticker, start_date, end_date):
    # Scarica i dati di chiusura
    df = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
    # Porta l'indice Date in colonna
    df = df.reset_index()
    # Converte i Timestamp (datetimes) in stringhe “YYYY‑MM‑DD”
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    # Crea una lista di dizionari
    records = df.to_dict(orient='records')
    return records
