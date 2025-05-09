from flask import Flask, jsonify, request
from flask_cors import CORS
import data_retrieval
import statistics
import visualization

app = Flask(__name__)
CORS(app)  # Consente richieste cross-origin

# Rotta di base
@app.route('/api')
def api_index():
    return jsonify({"message": "API Backend pronta!"})

# 1) Recupera i prezzi storici
@app.route('/api/price/<ticker>', methods=['GET'])
def get_price(ticker):
    start_date = request.args.get('start_date', '2021-01-01')
    end_date   = request.args.get('end_date',   '2025-01-01')
    records = data_retrieval.get_historical_data(ticker, start_date, end_date)
    return jsonify(records)

# 2) Analisi statistica sui prezzi in un intervallo
@app.route('/api/analysis/<ticker>', methods=['GET'])
def analyze_price(ticker):
    start_date = request.args.get('start_date', '2021-01-01')
    end_date   = request.args.get('end_date',   '2025-01-01')
    
    # Recupera i dati storici per il ticker
    data = data_retrieval.get_historical_data(ticker, start_date, end_date)
    
    # Verifica se i dati sono stati recuperati
    if not data:
        return jsonify({"error": "Nessun dato trovato per il ticker specificato"}), 404
    
    # Calcola le statistiche sui dati storici
    stats = statistics.calculate_statistics(data)
    
    # Restituisci le statistiche come risposta
    return jsonify(stats)

# 3) Generazione grafico e restituzione URL
@app.route('/api/visualize/<ticker>', methods=['GET'])
def visualize_price(ticker):
    start_date = request.args.get('start_date', '2021-01-01')
    end_date   = request.args.get('end_date',   '2025-01-01')
    data = data_retrieval.get_historical_data(ticker, start_date, end_date)
    chart_url = visualization.plot_chart(data)
    return jsonify({"chart_url": chart_url})

# 4) Analisi stagionale per finestra fissa anno su anni passati
@app.route('/api/seasonal/<ticker>', methods=['GET'])
def seasonal(ticker):
    # MM-DD start and end
    start_md   = request.args.get('start_md', '01-01')
    end_md     = request.args.get('end_md',   '01-18')
    years_back = request.args.get('years_back', None, type=int)

    # Ottieni dati filtrati per la finestra stagionale
    df_window = data_retrieval.get_seasonal_window(ticker, start_md, end_md, years_back)

    # Calcola le statistiche stagionali su quella finestra
    stats = statistics.seasonal_stats(df_window)
    return jsonify(stats)

@app.route('/api/performance/<ticker>', methods=['GET'])
def get_performance(ticker):
    start_date = request.args.get('start_date', '2021-01-01')
    end_date   = request.args.get('end_date',   '2025-01-01')
    
    # Recupera i dati storici per il ticker
    data = data_retrieval.get_historical_data(ticker, start_date, end_date)
    
    # Calcola le statistiche di performance come drawdown, max spike, etc.
    performance_stats = statistics.calculate_performance_stats(data)
    
    return jsonify(performance_stats)

@app.route('/api/seasonal_analysis/<ticker>', methods=['GET'])
def seasonal_analysis(ticker):
    start_md   = request.args.get('start_md', '01-01')  # Start Month-Day
    end_md     = request.args.get('end_md', '01-18')    # End Month-Day
    years_back = request.args.get('years_back', 5, type=int)  # Numero di anni da analizzare
    
    # Recupera i dati storici per il ticker
    data = data_retrieval.get_historical_data(ticker, '2000-01-01', '2025-01-01')
    
    # Filtro dei dati per la finestra temporale (01-01 al 18-01 per ogni anno)
    seasonal_data = data_retrieval.filter_by_seasonal_window(data, start_md, end_md)
    
    # Calcola le statistiche stagionali
    seasonal_stats = statistics.seasonal_analysis(seasonal_data, years_back)
    
    return jsonify(seasonal_stats)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
