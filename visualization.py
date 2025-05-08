import matplotlib.pyplot as plt
import pandas as pd
import os

# Assicurati che la cartella static esista
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

def plot_chart(records):
    df = pd.DataFrame(records)
    df['Date'] = pd.to_datetime(df['Date'])
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Close'], label='Close')
    plt.title('Prezzo di chiusura')
    plt.xlabel('Data')
    plt.ylabel('Prezzo')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Salva il grafico in static/
    chart_path = os.path.join(STATIC_DIR, 'price_chart.png')
    plt.savefig(chart_path)
    plt.close()
    # Restituisci l’URL relativo
    return '/static/price_chart.png'
