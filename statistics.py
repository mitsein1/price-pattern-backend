import pandas as pd
import numpy as np

def seasonal_stats(df_window):
    """
    df_window contiene i record di tutti gli anni ma solo nella finestra selezionata.
    Vogliamo svezzare per anno:
      - pct_up: % di anni con prezzo finale > iniziale
      - years_up: lista anni positivi
      - profit_per_year: lista di profit % per anno
      - drawdown_per_year, spike_per_year
      - performance cumulata, n_win/n_loss
      - sharpe, sortino, std
    """
    stats_per_year = []
    for year, g in df_window.groupby(df_window['Date'].dt.year):
        prices = g['Close'].values
        ret = (prices[-1] - prices[0]) / prices[0]
        daily = pd.Series(prices).pct_change().dropna()
        drawdown = ( (1+daily).cumprod() / (1+daily).cumprod().cummax() -1 ).min()
        spike     = daily.max()
        cum_perf  = (1+daily).prod() -1
        wins      = (daily>0).sum()
        losses    = (daily<0).sum()
        sharpe    = daily.mean()/daily.std() if daily.std()>0 else None
        downside  = daily[daily<0].std()
        sortino   = daily.mean()/downside if downside>0 else None

        stats_per_year.append({
          'year':          int(year),
          'pct_change':    float(ret),
          'drawdown':      float(drawdown),
          'max_spike':     float(spike),
          'cum_perf':      float(cum_perf),
          'n_win':         int(wins),
          'n_loss':        int(losses),
          'sharpe':        sharpe,
          'sortino':       sortino,
          'std_dev':       float(daily.std())
        })

    # Aggregati
    total_years = len(stats_per_year)
    up_years   = [s for s in stats_per_year if s['pct_change']>0]
    pct_up     = len(up_years)/total_years if total_years>0 else None
    avg_profit = np.mean([s['pct_change'] for s in stats_per_year]) if total_years>0 else None
    max_profit = max([s['pct_change'] for s in stats_per_year]) if total_years>0 else None

    return {
      'years':          stats_per_year,
      'pct_up_years':   pct_up,
      'avg_profit':     avg_profit,
      'max_profit':     max_profit
    }

