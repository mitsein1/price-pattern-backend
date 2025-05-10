import pandas as pd
import numpy as np
import yfinance as yf
from data_retrieval import fetch_price_data
import datetime
from statistics import get_price_series, calculate_cumulative_profit_per_year



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


def calculate_cumulative_profit_per_year(df: pd.DataFrame, start_day: int, end_day: int) -> pd.DataFrame:
    """Calcola il profitto cumulativo percentuale per ciascun anno dal giorno start_day al giorno end_day."""
    results = []
    for year in df.index.year.unique():
        subset = df[df.index.year == year]
        try:
            s = subset[subset.index.dayofyear == start_day]['Close'].iloc[0]
            e = subset[subset.index.dayofyear == end_day]['Close'].iloc[0]
            profit = (e / s - 1) * 100
            results.append({'year': int(year), 'cumulative_profit': round(profit, 2)})
        except IndexError:
            continue
    return pd.DataFrame(results)


def get_pattern_returns(df: pd.DataFrame, start_day: int, end_day: int) -> pd.DataFrame:
    """Ritorni annuali percentuali per ciascun anno dal giorno start_day al giorno end_day."""
    results = []
    for year in df.index.year.unique():
        subset = df[df.index.year == year]
        try:
            s = subset[subset.index.dayofyear == start_day]['Close'].iloc[0]
            e = subset[subset.index.dayofyear == end_day]['Close'].iloc[0]
            results.append({'year': int(year), 'return': round((e / s - 1) * 100, 2)})
        except IndexError:
            continue
    return pd.DataFrame(results)


def get_yearly_pattern_statistics(df: pd.DataFrame, start_day: int, end_day: int) -> list[dict]:
    """Statistiche dettagliate per ogni anno: profitti, massimi rialzi e ribassi."""
    stats = []
    for year in df.index.year.unique():
        subset = df[df.index.year == year]
        period = subset[(subset.index.dayofyear >= start_day) & (subset.index.dayofyear <= end_day)]
        if len(period) < 2:
            continue
        s = period['Close'].iloc[0]
        e = period['Close'].iloc[-1]
        max_rise = round((period['Close'].max() / s - 1) * 100, 2)
        max_drop = round((period['Close'].min() / s - 1) * 100, 2)
        profit = round((e - s), 2)
        profit_pct = round((e / s - 1) * 100, 2)
        stats.append({
            'year':         int(year),
            'start_price':  s,
            'end_price':    e,
            'profit':       profit,
            'profit_pct':   profit_pct,
            'max_rise':     max_rise,
            'max_drop':     max_drop
        })
    return stats


def get_profit_summary(df: pd.DataFrame, start_day: int, end_day: int) -> dict:
    """Riepilogo dei profitti totali e medi del pattern tra start_day ed end_day."""
    returns = get_pattern_returns(df, start_day, end_day)
    total = returns['return'].sum() if not returns.empty else 0
    average = returns['return'].mean() if not returns.empty else 0
    return {'total_profit': round(total, 2), 'average_profit': round(average, 2)}


def get_gains_losses(df: pd.DataFrame, start_day: int, end_day: int) -> dict:
    """Conta guadagni e perdite, percentuali medie e massimi."""
    returns = get_pattern_returns(df, start_day, end_day)
    gains = returns[returns['return'] > 0]
    losses = returns[returns['return'] < 0]
    return {
        'gains':     int(len(gains)),
        'losses':    int(len(losses)),
        'gain_pct':  round(gains['return'].mean(), 2) if not gains.empty else 0,
        'loss_pct':  round(losses['return'].mean(), 2) if not losses.empty else 0,
        'max_gain':  round(gains['return'].max(), 2) if not gains.empty else 0,
        'max_loss':  round(losses['return'].min(), 2) if not losses.empty else 0
    }


def calculate_misc_metrics(df: pd.DataFrame, start_day: int, end_day: int, risk_free_rate: float = 0.0) -> dict:
    """Calcola metriche aggiuntive: Sharpe, Sortino, volatilità e streak."""
    returns = get_pattern_returns(df, start_day, end_day)['return']
    count = len(returns)
    std_dev = returns.std() if count > 1 else 0
    sharpe = (returns.mean() - risk_free_rate) / std_dev if std_dev > 0 else 0
    downside = returns[returns < 0].std() if not returns[returns < 0].empty else 0
    sortino = (returns.mean() - risk_free_rate) / downside if downside > 0 else 0
    max_streak = 0
    streak = 0
    for val in returns:
        streak = streak + 1 if val > 0 else 0
        max_streak = max(max_streak, streak)

    return {
        'trades':          count,
        'calendar_days':   end_day - start_day + 1,
        'std_dev':         round(std_dev, 2),
        'sharpe_ratio':    round(sharpe, 2),
        'sortino_ratio':   round(sortino, 2),
        'volatility':      round(std_dev, 2),
        'current_streak':  max_streak,
        'gains':           int((returns > 0).sum()),
    }

def get_seasonality(
    asset: str,
    years_back: int,
    start_day: str = None,
    end_day: str = None
) -> dict:
    """
    Calcola la media del prezzo di chiusura giornaliero di `asset` sul periodo stagionale
    da `start_day` a `end_day` (formato 'MM-DD') negli ultimi `years_back` anni completi.
    Restituisce un dict con:
      - 'dates': list di 'MM-DD'
      - 'average_prices': list di float (arrotondati a 6 decimali)
    """
    from datetime import date

    # Determina l’intervallo di anni pieni
    today = date.today()
    end_year = today.year - 1
    start_year = end_year - years_back + 1

    sd = start_day or '01-01'
    ed = end_day   or '12-31'

    start_date = f"{start_year}-{sd}"
    end_date   = f"{end_year}-{ed}"

    # Prendi i prezzi
    df = fetch_price_data(asset, start_date, end_date)

    # Raggruppa per giorno-mese
    df['month_day'] = df['date'].apply(lambda d: d.strftime('%m-%d'))
    grouped = df.groupby('month_day')['close'].mean().reset_index()

    # Seleziona l’intervallo stagionale
    mask = (grouped['month_day'] >= sd) & (grouped['month_day'] <= ed)
    season_df = grouped.loc[mask].sort_values('month_day')

    return {
        'dates':          season_df['month_day'].tolist(),
        'average_prices': season_df['close'].round(6).tolist()
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

