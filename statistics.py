import pandas as pd

def calculate_statistics(data):
    df = pd.DataFrame(data)
    # Calcoliamo il profitto/perdita medio
    df['Daily Return'] = df['Close'].pct_change()
    mean_return = df['Daily Return'].mean()

    # Calcoliamo il drawdown massimo
    df['Cumulative Return'] = (1 + df['Daily Return']).cumprod()
    df['Drawdown'] = df['Cumulative Return'] / df['Cumulative Return'].cummax() - 1
    max_drawdown = df['Drawdown'].min()

    # Calcoliamo il picco massimo (max spike)
    max_spike = df['Daily Return'].max()

    return {
        "mean_return": mean_return,
        "max_drawdown": max_drawdown,
        "max_spike": max_spike
    }
