from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import datetime
# Importiamo il modulo locale 'statistics' rinominandolo per non confliggere
import statistics as stats_mod
import visualization
import data_retrieval

# Import esplicito delle funzioni aggiuntive dal modulo locale statistics
from statistics import (
    calculate_average_annual_pattern,
    calculate_cumulative_profit_per_year,
    get_pattern_returns,
    get_yearly_pattern_statistics,
    get_profit_summary,
    get_gains_losses,
    calculate_misc_metrics,
    get_seasonality, get_price_series,
    calculate_cumulative_profit_per_year,
)

# Assicuriamoci che get_data fornisca DataFrame con DateTimeIndex e colonna 'Close'
from data_retrieval import get_historical_data, get_seasonal_window, filter_by_seasonal_window, get_data
from fastapi import FastAPI, HTTPException, Query

app = Flask(__name__)
CORS(app)  # Consente richieste cross-origin

# Rotta di base
@app.route('/api')
def api_index():
    return jsonify({"message": "API Backend pronta!"})

# 1) Cumulative profit per year (pattern specifico)
@app.route("/api/cumulative-profit", methods=["GET"])
def cumulative_profit():
    symbol = request.args.get("symbol")
    start_day = int(request.args.get("start_day"))
    end_day = int(request.args.get("end_day"))
    df = get_data(symbol)
    result = calculate_cumulative_profit_per_year(df, start_day, end_day)
    return jsonify(result.to_dict(orient="records"))

# 2) Pattern returns
@app.route("/api/pattern-returns", methods=["GET"])
def pattern_returns():
    symbol = request.args.get("symbol")
    start_day = int(request.args.get("start_day"))
    end_day = int(request.args.get("end_day"))
    df = get_data(symbol)
    result = get_pattern_returns(df, start_day, end_day)
    return jsonify(result.to_dict(orient="records"))

# 3) Yearly pattern statistics
@app.route("/api/yearly-statistics", methods=["GET"])
def yearly_statistics():
    symbol = request.args.get("symbol")
    start_day = int(request.args.get("start_day"))
    end_day = int(request.args.get("end_day"))
    df = get_data(symbol)
    result = get_yearly_pattern_statistics(df, start_day, end_day)
    return jsonify(result)

# 4) Profit summary
@app.route("/api/profit-summary", methods=["GET"])
def profit_summary():
    symbol = request.args.get("symbol")
    start_day = int(request.args.get("start_day"))
    end_day = int(request.args.get("end_day"))
    df = get_data(symbol)
    result = get_profit_summary(df, start_day, end_day)
    return jsonify(result)

# 5) Gains and losses count
@app.route("/api/gains-losses", methods=["GET"])
def gains_losses():
    symbol = request.args.get("symbol")
    start_day = int(request.args.get("start_day"))
    end_day = int(request.args.get("end_day"))
    df = get_data(symbol)
    result = get_gains_losses(df, start_day, end_day)
    return jsonify(result)

# 6) Misc metrics (Sharpe, volatility, etc.)
@app.route("/api/misc-metrics", methods=["GET"])
def misc_metrics():
    symbol = request.args.get("symbol")
    start_day = int(request.args.get("start_day"))
    end_day = int(request.args.get("end_day"))
    df = get_data(symbol)
    result = calculate_misc_metrics(df, start_day, end_day)
    return jsonify(result)

# 7) Pattern stats endpoint aggregato
@app.route('/api/pattern_stats/<ticker>', methods=['GET'])
def get_pattern_statistics(ticker):
    start_md = request.args.get('start_md')  # es: "01-01"
    end_md = request.args.get('end_md')      # es: "01-18"
    if not start_md or not end_md:
        return jsonify({'error': 'start_md and end_md are required in format MM-DD'}), 400
    try:
        df = get_data(ticker)
        result = {
            'cumulative_profit': calculate_cumulative_profit_per_year(df, start_md, end_md),
            'pattern_returns': get_pattern_returns(df, start_md, end_md),
            'yearly_stats': get_yearly_pattern_statistics(df, start_md, end_md),
            'summary': get_profit_summary(df, start_md, end_md),
            'gains_losses': get_gains_losses(df, start_md, end_md)
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 8) Average annual pattern
@app.route("/api/average_annual/<ticker>", methods=['GET'])
def average_annual(ticker):
    years_back = request.args.get("years_back", default=None, type=int)
    try:
        data = calculate_average_annual_pattern(ticker, years_back)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 9) Historical price data
@app.route('/api/price/<ticker>', methods=['GET'])
def get_price(ticker):
    start_date = request.args.get('start_date', '2021-01-01')
    end_date = request.args.get('end_date', '2025-01-01')
    records = get_historical_data(ticker, start_date, end_date)
    return jsonify(records)

# 10) Basic statistics on history
@app.route('/api/analysis/<ticker>', methods=['GET'])
def analyze_price(ticker):
    start_date = request.args.get('start_date', '2021-01-01')
    end_date   = request.args.get('end_date',   '2025-01-01')
    data = get_historical_data(ticker, start_date, end_date)
    # Evita l’errore su DataFrame: controlla None o empty
    if data is None or data.empty:
        return jsonify({"error": "Nessun dato trovato per il ticker specificato"}), 404

    stats = stats_mod.calculate_statistics(data)
    return jsonify(stats)


# 11) Chart generation
@app.route('/api/visualize/<ticker>', methods=['GET'])
def visualize_price(ticker):
    start_date = request.args.get('start_date', '2021-01-01')
    end_date = request.args.get('end_date', '2025-01-01')
    data = get_historical_data(ticker, start_date, end_date)
    chart_url = visualization.plot_chart(data)
    return jsonify({"chart_url": chart_url})

# 12) Seasonal window stats
@app.route('/api/seasonal/<ticker>', methods=['GET'])
def seasonal(ticker):
    start_md = request.args.get('start_md', '01-01')
    end_md = request.args.get('end_md', '01-18')
    years_back = request.args.get('years_back', None, type=int)
    df_window = get_seasonal_window(ticker, start_md, end_md, years_back)
    stats = stats_mod.seasonal_stats(df_window)
    return jsonify(stats)

# 13) Performance stats (drawdown etc.)
@app.route('/api/performance/<ticker>', methods=['GET'])
def get_performance(ticker):
    start_date = request.args.get('start_date', '2021-01-01')
    end_date = request.args.get('end_date', '2025-01-01')
    data = get_historical_data(ticker, start_date, end_date)
    performance_stats = stats_mod.calculate_performance_stats(data)
    return jsonify(performance_stats)

# 14) Seasonal analysis endpoint
@app.route('/api/seasonal_analysis/<ticker>', methods=['GET'])
def seasonal_analysis(ticker):
    start_md = request.args.get('start_md', '01-01')
    end_md = request.args.get('end_md', '01-18')
    years_back = request.args.get('years_back', 5, type=int)
    data = get_historical_data(ticker, '2000-01-01', '2025-01-01')
    seasonal_data = filter_by_seasonal_window(data, start_md, end_md)
    seasonal_stats = stats_mod.seasonal_analysis(seasonal_data, years_back)
    return jsonify(seasonal_stats)

@app.route("/api/seasonality", methods=["GET"])
def seasonality():
    asset      = request.args.get("asset", type=str)
    years_back = request.args.get("years_back", type=int)
    start_day  = request.args.get("start_day", default=None, type=str)
    end_day    = request.args.get("end_day", default=None, type=str)

    # Validazione minima
    if not asset or not years_back:
        return jsonify({"error": "asset e years_back sono obbligatori"}), 400
    if start_day and end_day and start_day > end_day:
        return jsonify({"error": "start_day deve precedere o essere uguale a end_day"}), 400

    try:
        result = get_seasonality(asset, years_back, start_day, end_day)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
# PROVVISORIO PER TEST
@app.route("/api/debug-historical", methods=["GET"])
def debug_historical():
    from flask import request, jsonify
    ticker     = request.args.get("asset")
    start_date = request.args.get("start_date", "2020-01-01")
    end_date   = request.args.get("end_date",   "2024-12-31")
    from data_retrieval import get_historical_data
    df = get_historical_data(ticker, start_date, end_date)
    return jsonify({
        "rows":        len(df),
        "columns":     df.columns.tolist(),
        "first_dates": df.index[:3].strftime("%Y-%m-%d").tolist(),
        "last_dates":  df.index[-3:].strftime("%Y-%m-%d").tolist() if len(df)>=3 else []
    })

@app.route("/api/price-series", methods=["GET"])
def price_series():
    asset = request.args.get("asset")
    year  = request.args.get("year", type=int)
    if not asset or not year:
        return jsonify({"error": "asset e year sono obbligatori"}), 400

    data = get_price_series(asset, year)
    return jsonify(data)


@app.route("/api/cumulative-profit", methods=["GET"])
def cumulative_profit():
    asset    = request.args.get("asset")
    start_dd = request.args.get("start_day", type=int)
    end_dd   = request.args.get("end_day",   type=int)
    if not asset or start_dd is None or end_dd is None:
        return jsonify({"error": "asset, start_day e end_day obbligatori"}), 400

    # Scarica dati completi dal 2000 ad oggi
    df = get_historical_data(asset, "2000-01-01", datetime.date.today().isoformat())
    result_df = calculate_cumulative_profit_per_year(df, start_dd, end_dd)
    return jsonify(result_df.to_dict(orient="records"))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
