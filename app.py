from flask import Flask, jsonify, request
import data_retrieval
import statistics
import visualization

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({"message": "Backend pronto!"})

@app.route('/price/<ticker>', methods=['GET'])
def get_price(ticker):
    start_date = request.args.get('start_date', '2021-01-01')
    end_date = request.args.get('end_date', '2025-01-01')
    data = data_retrieval.get_historical_data(ticker, start_date, end_date)
    return jsonify(data)

@app.route('/analysis/<ticker>', methods=['GET'])
def analyze_price(ticker):
    start_date = request.args.get('start_date', '2021-01-01')
    end_date = request.args.get('end_date', '2025-01-01')
    data = data_retrieval.get_historical_data(ticker, start_date, end_date)
    stats = statistics.calculate_statistics(data)
    return jsonify(stats)

@app.route('/visualize/<ticker>', methods=['GET'])
def visualize_price(ticker):
    start_date = request.args.get('start_date', '2021-01-01')
    end_date = request.args.get('end_date', '2025-01-01')
    data = data_retrieval.get_historical_data(ticker, start_date, end_date)
    chart = visualization.plot_chart(data)
    return jsonify({"chart_url": chart})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
