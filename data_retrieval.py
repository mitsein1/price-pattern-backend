import yfinance as yf
import pandas as pd

def get_historical_data(ticker, start_date, end_date):
    """
    Scarica i dati di chiusura per un ticker tra due date.
    Restituisce una lista di dizionari con 'Date' (YYYY-MM-DD) e 'Close'.
    """
    try:
        df = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
        df = df.reset_index()
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        records = df.dropna().to_dict(orient='records')
        return records
    except Exception as e:
        print(f"Errore nel download dei dati storici per {ticker}: {e}")
        return []

def get_seasonal_window(ticker, start_md, end_md, years_back=None):
    """
    Estrae la finestra stagionale (es. 03-15 a 05-15) da tutti gli anni disponibili per il ticker.
    
    Args:
        ticker (str): simbolo del titolo (es. AAPL)
        start_md (str): data di inizio formato 'MM-DD'
        end_md (str): data di fine formato 'MM-DD'
        years_back (int, optional): numero di anni da includere (es. 10). Se None, include tutto.

    Returns:
        pd.DataFrame: con colonne 'Date', 'Close', 'md'
    """
    try:
        df = yf.Ticker(ticker).history(period='max')[['Close']].reset_index()
        df['md'] = df['Date'].dt.strftime('%m-%d')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna()

        # Finestra tra start_md e end_md
        subset = df[(df['md'] >= start_md) & (df['md'] <= end_md)]

        # Eventuale filtro per ultimi N anni
        if years_back:
            recent_year = subset['Date'].dt.year.max()
            subset = subset[subset['Date'].dt.year >= (recent_year - years_back + 1)]

        return subset

    except Exception as e:
        print(f"Errore nel calcolo della finestra stagionale per {ticker}: {e}")
        return pd.DataFrame()
