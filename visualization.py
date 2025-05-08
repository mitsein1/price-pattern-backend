import matplotlib.pyplot as plt

def plot_chart(data):
    df = pd.DataFrame(data)
    plt.plot(df['Close'])
    plt.title('Price Chart')
    plt.xlabel('Date')
    plt.ylabel('Price')
    
    # Salviamo il grafico come immagine
    chart_path = '/tmp/price_chart.png'
    plt.savefig(chart_path)
    plt.close()
    
    # Restituiamo il percorso dell'immagine salvata
    return chart_path
