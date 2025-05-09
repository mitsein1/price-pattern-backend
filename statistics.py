import pandas as pd
import numpy as np
import yfinance as yf

def calculate_average_annual_pattern(ticker, years_back=None):
    """
    Calcola l'andamento medio annuale di un asset, giorno per giorno, su base anni solari.
    """
    # Scarica dati storici completi
    df = yf.Ticker(ticker).history(period="max")[['Close']].reset_index()
    df['Date'] = pd.to_datetime(df['Date'])

    # Filtro anni recenti se richiesto
    if years_back:
        recent_year = df['Date'].dt.year.max()
        df = df[df['Date'].dt.year >= (recent_year - years_back + 1)]

    # Aggiungi colonne ausiliarie
    df['Year'] = df['Date'].dt.year
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # Normalizza ogni anno con il primo valore dell’anno (opzionale)
    df['Close_norm'] = df.groupby('Year')['Close'].transform(lambda x: x / x.iloc[0])

    # Calcola la media dei valori normalizzati per ogni giorno dell’anno
    daily_avg = df.groupby('DayOfYear')['Close_norm'].mean().reset_index()

    # Rinomina colonne per chiarezza
    daily_avg.columns = ['day_of_year', 'average_normalized_close']

    return daily_avg.to_dict(orient='records')

def calculate_performance_stats(data):
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Close'] = pd.to_numeric(df['Close'])

    daily = df['Close'].pct_change().dropna()
    cum_ret = (1 + daily).cumprod()
    drawdown = (cum_ret / cum_ret.cummax() - 1).min()
    max_spike = daily.max()

    sharpe = daily.mean() / daily.std() if daily.std() > 0 else None

    return {
        'max_drawdown': round(float(drawdown), 6),
        'max_spike': round(float(max_spike), 6),
        'total_return': round(float((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]), 6),
        'volatility': round(float(daily.std()), 6),
        'sharpe': round(float(sharpe), 6) if sharpe is not None else None
    }

def calculate_statistics(data):
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Close'] = pd.to_numeric(df['Close'])

    ret = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
    daily_ret = df['Close'].pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() if daily_ret.std() > 0 else None

    return {
        'pct_return_total': round(float(ret), 6),
        'sharpe_ratio': round(float(sharpe), 6) if sharpe is not None else None,
        'mean_daily_return': round(float(daily_ret.mean()), 6),
        'std_daily_return': round(float(daily_ret.std()), 6)
    }

def seasonal_stats(df_window):
    stats_per_year = []
    for year, g in df_window.groupby(df_window['Date'].dt.year):
        prices = g['Close'].values
        if len(prices) < 2:
            continue  # skip anni incompleti
        ret = (prices[-1] - prices[0]) / prices[0]
        daily = pd.Series(prices).pct_change().dropna()
        drawdown = ((1 + daily).cumprod() / (1 + daily).cumprod().cummax() - 1).min()
        spike = daily.max()
        cum_perf = (1 + daily).prod() - 1
        wins = (daily > 0).sum()
        losses = (daily < 0).sum()
        sharpe = daily.mean() / daily.std() if daily.std() > 0 else None
        downside = daily[daily < 0].std()
        sortino = daily.mean() / downside if downside and downside > 0 else None

        stats_per_year.append({
            'year': int(year),
            'pct_change': round(float(ret), 6),
            'drawdown': round(float(drawdown), 6),
            'max_spike': round(float(spike), 6),
            'cum_perf': round(float(cum_perf), 6),
            'n_win': int(wins),
            'n_loss': int(losses),
            'sharpe': round(float(sharpe), 6) if sharpe is not None else None,
            'sortino': round(float(sortino), 6) if sortino is not None else None,
            'std_dev': round(float(daily.std()), 6)
        })

    total_years = len(stats_per_year)
    if total_years == 0:
        return {
            'years': [],
            'pct_up_years': None,
            'avg_profit': None,
            'max_profit': None
        }

    up_years = [s for s in stats_per_year if s['pct_change'] > 0]
    pct_up = len(up_years) / total_years
    avg_profit = np.mean([s['pct_change'] for s in stats_per_year])
    max_profit = max([s['pct_change'] for s in stats_per_year])

    return {
        'years': stats_per_year,
        'pct_up_years': round(float(pct_up), 6),
        'avg_profit': round(float(avg_profit), 6),
        'max_profit': round(float(max_profit), 6)
    }
