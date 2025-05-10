import pandas as pd
import numpy as np
import yfinance as yf
from data_retrieval import fetch_price_data, get_historical_data, align_to_trading_days
import datetime
#from statistics import get_price_series, calculate_cumulative_profit_per_year
from datetime import date, datetime


def calculate_average_annual_pattern(ticker: str, years_back: int | None = None) -> list[dict]:
    """
    Calcola l'andamento medio annuale di un asset, giorno per giorno, su base anni solari.
    Restituisce una lista di dizionari con 'day_of_year' e 'average_normalized_close'.
    """
    # Scarica dati storici completi
    df = yf.Ticker(ticker).history(period="max")[['Close']].reset_index()
    df['Date'] = pd.to_datetime(df['Date'])

    # Filtra anni recenti se richiesto
    if years_back:
        recent_year = df['Date'].dt.year.max()
        df = df[df['Date'].dt.year >= (recent_year - years_back + 1)]

    # Colonne di supporto
    df['Year'] = df['Date'].dt.year
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # Normalizza ogni anno rispetto al primo valore annuale
    df['Close_norm'] = df.groupby('Year')['Close'].transform(lambda x: x / x.iloc[0] if len(x) else np.nan)

    # Media dei valori normalizzati per ogni giorno dell'anno
    daily_avg = df.groupby('DayOfYear')['Close_norm'].mean().reset_index()
    daily_avg.columns = ['day_of_year', 'average_normalized_close']

    return daily_avg.to_dict(orient='records')


def calculate_performance_stats(data: list[dict]) -> dict:
    """Calcola drawdown massimo, picco max, rendimento totale, volatilità e Sharpe ratio."""
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    daily = df['Close'].pct_change().dropna()
    cum_ret = (1 + daily).cumprod()
    drawdown = (cum_ret / cum_ret.cummax() - 1).min()
    spike = daily.max()
    total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) if len(df) > 1 else 0
    volatility = daily.std()
    sharpe = daily.mean() / volatility if volatility and volatility > 0 else None

    return {
        'max_drawdown': round(float(drawdown), 6),
        'max_spike':    round(float(spike), 6),
        'total_return': round(float(total_return), 6),
        'volatility':   round(float(volatility), 6),
        'sharpe':       round(float(sharpe), 6) if sharpe is not None else None
    }


def calculate_statistics(data: list[dict]) -> dict:
    """Calcolo base: rendimento totale, Sharpe ratio, media e deviazione standard dei rendimenti giornalieri."""
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) if len(df) > 1 else 0
    daily_ret = df['Close'].pct_change().dropna()
    volatility = daily_ret.std()
    sharpe = daily_ret.mean() / volatility if volatility and volatility > 0 else None

    return {
        'pct_return_total':    round(float(total_return), 6),
        'sharpe_ratio':        round(float(sharpe), 6) if sharpe is not None else None,
        'mean_daily_return':   round(float(daily_ret.mean()), 6),
        'std_daily_return':    round(float(volatility), 6)
    }


def seasonal_stats(df_window: pd.DataFrame) -> dict:
    """Calcola statistiche stagionali anno per anno su DataFrame con colonna 'Close' e indice DateTimeIndex."""
    stats = []
    for year, group in df_window.groupby(df_window.index.year):
        prices = group['Close'].values
        if len(prices) < 2:
            continue
        daily = pd.Series(prices).pct_change().dropna()
        ret = (prices[-1] / prices[0] - 1)
        drawdown = ((1 + daily).cumprod() / (1 + daily).cumprod().cummax() - 1).min()
        spike = daily.max()
        cum_perf = (1 + daily).prod() - 1
        wins = int((daily > 0).sum())
        losses = int((daily < 0).sum())
        vol = daily.std()
        sharpe = daily.mean() / vol if vol and vol > 0 else None
        downside = daily[daily < 0].std()
        sortino = daily.mean() / downside if downside and downside > 0 else None

        stats.append({
            'year':       int(year),
            'pct_change': round(float(ret), 6),
            'drawdown':   round(float(drawdown), 6),
            'max_spike':  round(float(spike), 6),
            'cum_perf':   round(float(cum_perf), 6),
            'n_win':      wins,
            'n_loss':     losses,
            'sharpe':     round(float(sharpe), 6) if sharpe is not None else None,
            'sortino':    round(float(sortino), 6) if sortino is not None else None,
            'std_dev':    round(float(vol), 6)
        })

    total = len(stats)
    if total == 0:
        return {'years': [], 'pct_up_years': None, 'avg_profit': None, 'max_profit': None}

    up = [s for s in stats if s['pct_change'] > 0]
    pct_up = len(up) / total
    avg_profit = float(np.mean([s['pct_change'] for s in stats]))
    max_profit = float(max(s['pct_change'] for s in stats))

    return {
        'years':        stats,
        'pct_up_years': round(pct_up, 6),
        'avg_profit':   round(avg_profit, 6),
        'max_profit':   round(max_profit, 6)
    }


from datetime import datetime
import pandas as pd

def calculate_cumulative_profit_per_year(
    df: pd.DataFrame,
    start_md: str,
    end_md: str
) -> pd.DataFrame:
    """
    Calcola il profitto cumulativo percentuale per ciascun anno nella finestra month‑day:
      - df: DatetimeIndex, colonna 'close'
      - start_md/end_md: stringhe "MM-DD"
    Restituisce DataFrame con colonne ['year','cumulative_profit'].
    """

    # Helper: "MM-DD" → day-of-year per un dato anno
    def md_to_doy(md: str, year: int) -> int:
        dt = datetime.strptime(f"{year}-{md}", "%Y-%m-%d")
        return dt.timetuple().tm_yday

    results = []
    for year, group in df.groupby(df.index.year):
        sd = md_to_doy(start_md, year)
        ed = md_to_doy(end_md,   year)

        subset = group[
            (group.index.dayofyear >= sd) &
            (group.index.dayofyear <= ed)
        ]
        if len(subset) < 2:
            continue

        s = subset['close'].iloc[0]
        e = subset['close'].iloc[-1]
        profit = (e / s - 1) * 100
        results.append({
            'year':             int(year),
            'cumulative_profit': round(profit, 2)
        })

    return pd.DataFrame(results)



from datetime import datetime
import pandas as pd

def get_pattern_returns(
    df: pd.DataFrame,
    start_md: str,
    end_md: str
) -> pd.DataFrame:
    """
    Ritorni annuali percentuali per ciascun anno nella finestra month‑day:
      - df: DatetimeIndex, colonna 'close'
      - start_md/end_md: stringhe "MM-DD"
    Restituisce DataFrame con colonne ['year','return'].
    """

    # Helper: "MM-DD" → day-of-year per un dato anno
    def md_to_doy(md: str, year: int) -> int:
        dt = datetime.strptime(f"{year}-{md}", "%Y-%m-%d")
        return dt.timetuple().tm_yday

    results = []
    for year, group in df.groupby(df.index.year):
        sd = md_to_doy(start_md, year)
        ed = md_to_doy(end_md,   year)

        period = group[
            (group.index.dayofyear >= sd) &
            (group.index.dayofyear <= ed)
        ]
        if len(period) < 2:
            continue

        s = period['close'].iloc[0]
        e = period['close'].iloc[-1]
        pct = (e / s - 1) * 100
        results.append({'year': int(year), 'return': round(pct, 2)})

    return pd.DataFrame(results)




from data_retrieval import align_to_trading_days
import pandas as pd

def get_yearly_pattern_statistics(
    df: pd.DataFrame,
    start_md: str,
    end_md: str
) -> list[dict]:
    """
    Statistiche dettagliate per ogni anno su una finestra month‑day:
      - df: DatetimeIndex, colonna 'close' o 'Close'
      - start_md/end_md: stringhe "MM-DD"
    Restituisce lista di dict con year, start_price, end_price,
    profit, profit_pct, max_rise, max_drop.
    """

    # Helper: converte "MM-DD" in day-of-year per un anno
    def md_to_doy(md: str, year: int) -> int:
        dt = datetime.strptime(f"{year}-{md}", "%Y-%m-%d")
        return dt.timetuple().tm_yday

    stats = []

    for year, group in df.groupby(df.index.year):
        # 1) Calcola i day‑of‑year raw
        sd_raw = md_to_doy(start_md, year)
        ed_raw = md_to_doy(end_md,   year)

        # 2) Allinea ai giorni di trading validi
        aligned_start, aligned_end = align_to_trading_days(group, sd_raw, ed_raw)
        if aligned_start is None or aligned_end is None:
            continue

        # 3) Filtra la finestra allineata
        period = group[
            (group.index.dayofyear >= aligned_start) &
            (group.index.dayofyear <= aligned_end)
        ]
        if len(period) < 2:
            continue

        # 4) Estrai prezzi
        s = period['close' if 'close' in period.columns else 'Close'].iloc[0]
        e = period['close' if 'close' in period.columns else 'Close'].iloc[-1]

        # 5) Calcola statistiche
        max_rise   = round((period['Close'].max()  / s - 1) * 100, 2)
        max_drop   = round((period['Close'].min()  / s - 1) * 100, 2)
        profit     = round((e - s), 2)
        profit_pct = round((e / s - 1) * 100, 2)

        stats.append({
            'year':        int(year),
            'start_price': round(s, 2),
            'end_price':   round(e, 2),
            'profit':      profit,
            'profit_pct':  profit_pct,
            'max_rise':    max_rise,
            'max_drop':    max_drop
        })

    return stats



def get_profit_summary(
    df: pd.DataFrame,
    start_md: str,
    end_md: str
) -> dict:
    """
    Riepilogo dei profitti totali e medi del pattern tra start_md ed end_md:
      - df: DatetimeIndex, colonna 'close'
      - start_md/end_md: stringhe "MM-DD"
    """
    # Recupera i ritorni annuali percentuali
    returns_df = get_pattern_returns(df, start_md, end_md)

    total   = returns_df['return'].sum() if not returns_df.empty else 0
    average = returns_df['return'].mean() if not returns_df.empty else 0

    return {
        'total_profit':   round(total,   2),
        'average_profit': round(average, 2)
    }



def get_gains_losses(
    df: pd.DataFrame,
    start_md: str,
    end_md: str
) -> dict:
    """
    Conta guadagni e perdite sulla finestra month‑day per ogni anno:
      - df: DatetimeIndex, colonna 'close'
      - start_md/end_md: stringhe "MM-DD"
    """
    # Ottieni un DataFrame con colonne ['year','return']
    returns_df = get_pattern_returns(df, start_md, end_md)

    gains  = returns_df[returns_df['return'] >  0]['return']
    losses = returns_df[returns_df['return'] <  0]['return']

    return {
        'gains':     int(len(gains)),
        'losses':    int(len(losses)),
        'gain_pct':  round(gains.mean(), 2)  if not gains.empty else 0,
        'loss_pct':  round(losses.mean(), 2) if not losses.empty else 0,
        'max_gain':  round(gains.max(), 2)   if not gains.empty else 0,
        'max_loss':  round(losses.min(), 2)  if not losses.empty else 0
    }



def calculate_misc_metrics(
    df: pd.DataFrame,
    start_md: str,
    end_md: str,
    years_back: int,
    risk_free_rate: float = 0.0
) -> dict:
    """
    Calcola metriche avanzate su una finestra month-day per gli ultimi years_back:

      - df: indice DatetimeIndex, colonna 'close'
      - start_md/end_md: stringhe "MM-DD"
      - years_back: quanti anni indietro considerare

    Restituisce dict con:
      trades, calendar_days, std_dev, sharpe_ratio, sortino_ratio,
      volatility, current_streak, gains
    """
    # 1) Filtra ultimi N anni
    latest_year = df.index.year.max()
    cutoff = latest_year - years_back + 1
    df = df[df.index.year >= cutoff]

    # 2) Helper per MD → dayofyear
    def md_to_doy(md: str, year: int) -> int:
        dt = datetime.strptime(f"{year}-{md}", "%Y-%m-%d")
        return dt.timetuple().tm_yday

    all_returns = []
    trading_days = 0
    for year, group in df.groupby(df.index.year):
        sd = md_to_doy(start_md, year)
        ed = md_to_doy(end_md,   year)
        window = group[
            (group.index.dayofyear >= sd) &
            (group.index.dayofyear <= ed)
        ]
        # conta giorni di trading in finestra
        trading_days += len(window)
        rets = window['close'].pct_change().dropna().tolist()
        all_returns.extend(rets)

    returns = pd.Series(all_returns)
    count   = len(returns)
    std_dev = returns.std() if count > 1 else 0
    sharpe  = (returns.mean() - risk_free_rate) / std_dev if std_dev > 0 else 0
    downside = returns[returns < 0].std() if not returns[returns < 0].empty else 0
    sortino  = (returns.mean() - risk_free_rate) / downside if downside > 0 else 0

    # calcolo streak
    max_streak = streak = 0
    for val in returns:
        streak = streak + 1 if val > 0 else 0
        max_streak = max(max_streak, streak)

    return {
        'trades':         count,
        'calendar_days':  trading_days,
        'std_dev':        round(std_dev, 2),
        'sharpe_ratio':   round(sharpe, 2),
        'sortino_ratio':  round(sortino, 2),
        'volatility':     round(std_dev, 2),
        'current_streak': max_streak,
        'gains':          int((returns > 0).sum()),
    }
def get_price_series(asset: str, year: int) -> dict:
    """
    Ritorna le date e i prezzi di chiusura giornalieri di `asset` per l'anno `year`.
    Se `year` è l'anno corrente, include fino a oggi; altrimenti fino al 31/12.
    """
    today = datetime.date.today()
    start_date = f"{year}-01-01"
    end_date = (
        today.isoformat()
        if year == today.year
        else f"{year}-12-31"
    )
    df = fetch_price_data(asset, start_date, end_date)
    return {
        "dates":  [d.isoformat() for d in df["date"]],
        "prices": df["close"].tolist()
    }

from data_retrieval import get_historical_data
import pandas as pd
from datetime import date

def get_seasonality(asset: str, years_back: int, start_day: str = None, end_day: str = None) -> dict:
    """
    Restituisce per ogni giorno MM-DD della finestra:
      - dates: lista di "MM-DD"
      - average_prices: valore medio normalizzato (base=1.0 al primo giorno di ciascun anno)
    """
    # 1) Intervallo di anni completi
    today = date.today()
    end_year = today.year - 1
    start_year = end_year - years_back + 1

    # 2) Date di fetch
    sd = start_day or "01-01"
    ed = end_day   or "12-31"
    start_date = f"{start_year}-{sd}"
    end_date   = f"{end_year}-{ed}"

    # 3) Scarica dati (indice DatetimeIndex, colonna 'Close')
    df = get_historical_data(asset, start_date, end_date)
    df = df.rename(columns={'Close': 'close'})

    # --- DEBUG: verifica righe e primi indici ---
    print(f"[DEBUG] fetched {len(df)} rows from {start_date} to {end_date}")
    print(f"[DEBUG] first dates: {list(df.index[:5])}")

    # 3a) Assicuriamoci che l’indice sia datetime
    df.index = pd.to_datetime(df.index)

    # 3b) Normalizza per anno (base = primo close in finestra)
    df['Year'] = df.index.year
    df['close_norm'] = df['close'] / df.groupby('Year')['close'] \
                                        .transform(lambda s: s.iloc[0])

    # 3c) Raggruppa per MM-DD e calcola media normalizzata
    df['month_day'] = df.index.strftime('%m-%d')
    grouped = (
        df
        .groupby('month_day')['close_norm']
        .mean()
        .reset_index()
        .rename(columns={'close_norm': 'average_norm'})
    )

    # 4) Filtra la sotto-finestra scelta dall’utente
    mask = (grouped['month_day'] >= sd) & (grouped['month_day'] <= ed)
    season_df = grouped.loc[mask].sort_values('month_day')

    # 5) Risposta JSON
    return {
        'dates':          season_df['month_day'].tolist(),
        'average_prices': season_df['average_norm'].round(6).tolist()
    }






