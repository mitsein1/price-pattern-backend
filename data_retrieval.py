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
    
def get_seasonal_window(ticker, start_md, end_md, years_back=None):
    """
    start_md, end_md = 'MM-DD' strings
    years_back = numero di anni da includere (oppure None = tutti)
    """
    df = yf.Ticker(ticker).history(period='max')[['Close']].reset_index()
    df['md'] = df['Date'].dt.strftime('%m-%d')
    # filtra le md tra start_md e end_md
    subset = df[(df['md'] >= start_md) & (df['md'] <= end_md)]
    if years_back:
        recent = subset['Date'].dt.year.max()
        subset = subset[subset['Date'].dt.year >= (recent - years_back + 1)]
    return subset
