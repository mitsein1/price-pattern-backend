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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
