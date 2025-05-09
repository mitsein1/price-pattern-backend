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

# Calcola il profitto cumulativo per ciascun anno in un pattern specifico
def calculate_cumulative_profit_per_year(df, start_day, end_day):
    results = []
    for year in df.index.year.unique():
        year_data = df[df.index.year == year]
        try:
            start = year_data[(year_data.index.dayofyear == start_day)].iloc[0]
            end = year_data[(year_data.index.dayofyear == end_day)].iloc[0]
            profit = (end['Close'] - start['Close']) / start['Close'] * 100
            results.append({'year': year, 'cumulative_profit': profit})
        except:
            continue
    return pd.DataFrame(results)

# Ritorni annuali del pattern (dal giorno start al giorno end) sotto forma di valori % per anno
def get_pattern_returns(df, start_day, end_day):
    results = []
    for year in df.index.year.unique():
        year_data = df[df.index.year == year]
        try:
            start = year_data[(year_data.index.dayofyear == start_day)].iloc[0]
            end = year_data[(year_data.index.dayofyear == end_day)].iloc[0]
            profit = (end['Close'] - start['Close']) / start['Close'] * 100
            results.append({'year': year, 'return': profit})
        except:
            continue
    return pd.DataFrame(results)

# Statistiche annuali dettagliate per il pattern: prezzo iniziale/finale, profitti, max rise/drop
def get_yearly_pattern_statistics(df, start_day, end_day):
    stats = []
    for year in df.index.year.unique():
        year_data = df[df.index.year == year]
        try:
            period_data = year_data[(year_data.index.dayofyear >= start_day) & (year_data.index.dayofyear <= end_day)]
            start = period_data.iloc[0]
            end = period_data.iloc[-1]
            max_rise = (period_data['Close'].max() - start['Close']) / start['Close'] * 100
            max_drop = (period_data['Close'].min() - start['Close']) / start['Close'] * 100
            profit = end['Close'] - start['Close']
            profit_pct = profit / start['Close'] * 100
            stats.append({
                'year': year,
                'start_date': start.name.strftime('%d %b %Y'),
                'end_date': end.name.strftime('%d %b %Y'),
                'start_price': round(start['Close'], 2),
                'end_price': round(end['Close'], 2),
                'profit': round(profit, 2),
                'profit_pct': round(profit_pct, 2),
                'max_rise': round(max_rise, 2),
                'max_drop': round(max_drop, 2),
            })
        except:
            continue
    return stats

# Riepilogo profitti totali e medi
def get_profit_summary(df, start_day, end_day):
    returns = get_pattern_returns(df, start_day, end_day)
    total = returns['return'].sum()
    average = returns['return'].mean()
    return {'total_profit': round(total, 2), 'average_profit': round(average, 2)}

# Conta gains/losses, percentuali e massimi
def get_gains_losses(df, start_day, end_day):
    returns = get_pattern_returns(df, start_day, end_day)
    gains = returns[returns['return'] > 0]
    losses = returns[returns['return'] < 0]
    return {
        'gains': len(gains),
        'losses': len(losses),
        'gain_pct': round(gains['return'].mean(), 2) if not gains.empty else 0,
        'loss_pct': round(losses['return'].mean(), 2) if not losses.empty else 0,
        'max_profit': round(gains['return'].max(), 2) if not gains.empty else 0,
        'max_loss': round(losses['return'].min(), 2) if not losses.empty else 0,
    }

# Statistiche varie: sharpe, sortino, dev std, volatilità, streak
def calculate_misc_metrics(df, start_day, end_day, risk_free_rate=0.0):
    returns = get_pattern_returns(df, start_day, end_day)
    values = returns['return']
    std_dev = values.std()
    sharpe = (values.mean() - risk_free_rate) / std_dev if std_dev != 0 else 0
    downside_std = values[values < 0].std()
    sortino = (values.mean() - risk_free_rate) / downside_std if downside_std != 0 else 0
    volatility = std_dev
    streak = 0
    max_streak = 0
    last_was_gain = None
    for val in values:
        if val > 0:
            if last_was_gain is True:
                streak += 1
            else:
                streak = 1
                last_was_gain = True
        else:
            last_was_gain = False
            streak = 0
        max_streak = max(max_streak, streak)

    return {
        'trades': len(values),
        'trading_days': end_day - start_day + 1,
        'std_dev': round(std_dev, 2),
        'sharpe_ratio': round(sharpe, 2),
        'sortino_ratio': round(sortino, 2),
        'volatility': round(volatility, 2),
        'calendar_days': end_day - start_day + 1,
        'current_streak': max_streak,
        'gains': (values > 0).sum(),
    }
