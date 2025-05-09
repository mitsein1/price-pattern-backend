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
    end_date   = request.args.get('end_date',   '2025-01-01')
    records = data_retrieval.get_historical_data(ticker, start_date, end_date)
    return jsonify(records)

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
    chart_url = visualization.plot_chart(data)
    return jsonify({"chart_url": chart_url})

@app.route('/seasonal/<ticker>', methods=['GET'])
def seasonal(ticker):
    start_md   = request.args.get('start_md','01-01')
    end_md     = request.args.get('end_md','01-18')
    years_back = request.args.get('years_back', None, type=int)
    df_window  = data_retrieval.get_seasonal_window(ticker, start_md, end_md, years_back)
    stats      = statistics.seasonal_stats(df_window)
    return jsonify(stats)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
