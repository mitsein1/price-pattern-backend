import yfinance as yf
import pandas as pd

# data_retrieval.py

def filter_by_seasonal_window(df, start_day, end_day):
    """
    Filtra le righe del DataFrame `df` che rientrano nella finestra stagionale tra `start_day` e `end_day`.

    - `df` deve avere un indice datetime o una colonna 'Date'.
    - `start_day` e `end_day` sono stringhe nel formato 'MM-DD', es: '03-10' o '05-05'.

    Restituisce un DataFrame filtrato.
    """
    import pandas as pd

    # Se l'indice non è datetime, prova a convertirlo
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        else:
            raise ValueError("DataFrame deve avere un indice datetime o una colonna 'Date'.")

    # Crea colonna MM-DD per ogni riga
    month_day = df.index.strftime('%m-%d')

    if start_day <= end_day:
        # Finestra lineare (es: 03-10 a 05-05)
        mask = (month_day >= start_day) & (month_day <= end_day)
    else:
        # Finestra che attraversa l'anno (es: 11-15 a 02-10)
        mask = (month_day >= start_day) | (month_day <= end_day)

    return df[mask]


def get_data(symbol):
    # Esempio: carica i dati da un CSV chiamato "AAPL.csv"
    df = pd.read_csv(f"data/{symbol}.csv", parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df = df[["Close"]]  # Tieni solo la colonna Close
    return df


# Carica dati di prezzo e restituisce un DataFrame con indice DatetimeIndex e colonna 'Close'
def get_historical_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Scarica i dati di chiusura per un ticker tra due date.
    Restituisce un oggetto pandas.DataFrame con:
      - DateTimeIndex (date)
      - colonna 'Close' (float)
    """
    try:
        # Scarica i dati con yfinance, mantiene solo la colonna Close
        df = yf.download(ticker, start=start_date, end=end_date)[['Close']]
        # Assicura che l'indice sia di tipo datetime
        df.index = pd.to_datetime(df.index)
        # Converte Close in numerico e rimuove eventuali NaN
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna()

        # Verifica i requisiti minimi
        assert isinstance(df.index, pd.DatetimeIndex), "L'indice non è DatetimeIndex"
        assert 'Close' in df.columns, "La colonna 'Close' non è presente"

        return df
    except Exception as e:
        print(f"Errore nel download dei dati storici per {ticker}: {e}")
        # Restituisce DataFrame vuoto con struttura corretta
        return pd.DataFrame(columns=['Close'], index=pd.DatetimeIndex([]))

# Wrapper che restituisce i dati in formato lista di dizionari
# utile se vuoi esportare in JSON o inviarli via API

def get_historical_records(ticker: str, start_date: str, end_date: str) -> list[dict]:
    """
    Scarica i dati di chiusura per un ticker e restituisce 
    una lista di dizionari con campi 'Date' (YYYY-MM-DD) e 'Close'.
    """
    df = get_historical_data(ticker, start_date, end_date)
    df_reset = df.reset_index()
    df_reset['Date'] = df_reset['Date'].dt.strftime('%Y-%m-%d')
    return df_reset.rename(columns={'index': 'Date'}).loc[:, ['Date', 'Close']].to_dict(orient='records')

# Estrae la finestra stagionale da tutti gli anni disponibili per il ticker
def get_seasonal_window(
    ticker: str,
    start_md: str,
    end_md: str,
    years_back: int | None = None
) -> pd.DataFrame:
    """
    Argomenti:
        ticker (str): simbolo del titolo (es. 'AAPL')
        start_md (str): data di inizio formato 'MM-DD'
        end_md (str): data di fine formato 'MM-DD'
        years_back (int, optional): numero di anni da includere.
            Se None, include tutti gli anni disponibili.
    Restituisce:
        pd.DataFrame con:
          - DateTimeIndex
          - colonna 'Close'
          - colonna 'md' (mese-giorno)
    """
    try:
        # Scarica l'intera serie storica
        df = yf.Ticker(ticker).history(period='max')[['Close']]
        # Assicura indice datetime e colonna Close
        df.index = pd.to_datetime(df.index)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna()

        # Aggiunge colonna 'md' per il confronto di mese-giorno
        df['md'] = df.index.strftime('%m-%d')

        # Filtra tra start_md e end_md
        subset = df[(df['md'] >= start_md) & (df['md'] <= end_md)]

        # Limita agli ultimi `years_back` anni, se richiesto
        if years_back:
            max_year = subset.index.year.max()
            subset = subset[subset.index.year >= (max_year - years_back + 1)]

        # Verifica i requisiti minimi
        assert isinstance(subset.index, pd.DatetimeIndex), "L'indice non è DatetimeIndex"
        assert 'Close' in subset.columns, "La colonna 'Close' non è presente"

        return subset
    except Exception as e:
        print(f"Errore nel calcolo della finestra stagionale per {ticker}: {e}")
        # Restituisce DataFrame vuoto con le colonne previste
        return pd.DataFrame(columns=['Close', 'md'], index=pd.DatetimeIndex([]))
