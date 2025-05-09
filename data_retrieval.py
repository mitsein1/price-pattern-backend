import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional

# Carica dati di prezzo da CSV locale
# Il file CSV deve trovarsi in data/{symbol}.csv con colonna 'Date' e 'Close'
def get_data(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(f"data/{symbol}.csv", parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df = df[["Close"]]
    return df

# Scarica dati storici da yfinance e restituisce DataFrame con indice DatetimeIndex e colonna 'Close'
def get_historical_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start_date, end=end_date)[['Close']]
        df.index = pd.to_datetime(df.index)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Errore nel download dei dati storici per {ticker}: {e}")
        return pd.DataFrame(columns=['Close'], index=pd.DatetimeIndex([]))

# Wrapper per API: restituisce lista di record con 'Date' e 'Close'
def get_historical_records(ticker: str, start_date: str, end_date: str) -> List[Dict]:
    df = get_historical_data(ticker, start_date, end_date)
    records = df.reset_index()
    records['Date'] = records['Date'].dt.strftime('%Y-%m-%d')
    return records[['Date', 'Close']].to_dict(orient='records')

# Filtra DataFrame per finestra stagionale su base mese-giorno
# start_md, end_md nel formato 'MM-DD'
def filter_by_seasonal_window(df: pd.DataFrame, start_md: str, end_md: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        else:
            raise ValueError("DataFrame deve avere indice datetime o colonna 'Date'.")
    md = df.index.strftime('%m-%d')
    if start_md <= end_md:
        mask = (md >= start_md) & (md <= end_md)
    else:
        mask = (md >= start_md) | (md <= end_md)
    return df.loc[mask]

# Estrae la finestra stagionale per un ticker su serie completa e ritorna DataFrame con colonne 'Close' e 'md'
def get_seasonal_window(
    ticker: str,
    start_md: str,
    end_md: str,
    years_back: Optional[int] = None
) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period='max')[['Close']]
        df.index = pd.to_datetime(df.index)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(inplace=True)
        df['md'] = df.index.strftime('%m-%d')
        subset = df[(df['md'] >= start_md) & (df['md'] <= end_md)]
        if years_back is not None:
            max_year = subset.index.year.max()
            subset = subset[subset.index.year >= (max_year - years_back + 1)]
        return subset
    except Exception as e:
        print(f"Errore nel calcolo della finestra stagionale per {ticker}: {e}")
        return pd.DataFrame(columns=['Close', 'md'], index=pd.DatetimeIndex([]))
