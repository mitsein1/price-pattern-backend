from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import datetime
# Importiamo il modulo locale 'statistics' rinominandolo per non confliggere
import statistics as stats_mod
import visualization
import data_retrieval
import logging
logger = logging.getLogger(__name__)
from datetime import date
import io
import matplotlib.pyplot as plt
from flask import send_file


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



# 2) Pattern returns
@app.route("/api/pattern-returns", methods=["GET"])
def pattern_returns():
    asset    = request.args.get("asset",     type=str)
    start_md = request.args.get("start_day", type=str)
    end_md   = request.args.get("end_day",   type=str)
    if not asset or not start_md or not end_md:
        return jsonify({"error":"asset, start_day e end_day obbligatori"}),400

    df = get_historical_data(asset, "2000-01-01", date.today().isoformat())
    df = df.rename(columns={'Close':'close'})

    result_df = get_pattern_returns(df, start_md, end_md)
    return jsonify(result_df.to_dict(orient="records"))

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
    asset    = request.args.get("asset",     type=str)
    start_md = request.args.get("start_day", type=str)
    end_md   = request.args.get("end_day",   type=str)
    if not asset or not start_md or not end_md:
        return jsonify({"error":"asset, start_day e end_day obbligatori"}),400

    # 1) Fetch storico completo
    df = get_historical_data(asset, "2000-01-01", date.today().isoformat())
    df = df.rename(columns={'Close':'close'})

    # 2) Calcola summary con month-day strings
    result = get_profit_summary(df, start_md, end_md)

    return jsonify(result)

# 5) Gains and losses count
@app.route("/api/cumulative-profit", methods=["GET"])
def cumulative_profit():
    asset    = request.args.get("asset",     type=str)
    start_md = request.args.get("start_day", type=str)
    end_md   = request.args.get("end_day",   type=str)
    if not asset or not start_md or not end_md:
        return jsonify({"error":"asset, start_day e end_day obbligatori"}),400

    # Scarica dati storici dal 2000 fino a oggi
    df = get_historical_data(asset, "2000-01-01", date.today().isoformat())
    # Standardizza sempre la colonna a 'close'
    df = df.rename(columns={'Close':'close'})

    # Calcola il profitto cumulativo per ciascun anno nella finestra MM-DD
    result_df = calculate_cumulative_profit_per_year(df, start_md, end_md)

    # Restituisci JSON
    return jsonify(result_df.to_dict(orient="records"))

# 6) Misc metrics (Sharpe, volatility, etc.)
@app.route("/api/misc-metrics", methods=["GET"])
def misc_metrics():
    asset      = request.args.get("asset",      type=str)
    years_back = request.args.get("years_back", type=int)
    start_md   = request.args.get("start_day",  type=str)
    end_md     = request.args.get("end_day",    type=str)

    # Validazione
    if not asset or not years_back or not start_md or not end_md:
        return jsonify({"error":"asset, years_back, start_day e end_day obbligatori"}),400

    # 1) Scarica dati storici fino a oggi
    df = get_historical_data(asset, "2000-01-01", date.today().isoformat())
    df = df.rename(columns={'Close':'close'})

    # 2) Calcola metriche tecniche sui últimos years_back
    result = calculate_misc_metrics(df, start_md, end_md, years_back)

    # 3) Aggiungi calendar_days dinamico
    sd = md_to_doy(start_md, date.today().year)
    ed = md_to_doy(end_md,   date.today().year)
    result['calendar_days'] = ed - sd + 1

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

from flask import request, jsonify
from statistics import get_seasonality

@app.route("/api/seasonality", methods=["GET"])
def seasonality():
    # Debug: conferma entrata nella route
    logger.info("Entered /api/seasonality handler")
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
        # Debug: parametri passati a get_seasonality
        print(f"[DEBUG] Params - asset: {asset}, years_back: {years_back}, start_day: {start_day}, end_day: {end_day}")
        
        result = get_seasonality(asset, years_back, start_day, end_day)
        
        # Debug: dimensione del risultato
        print(f"[DEBUG] Seasonality result - dates: {len(result['dates'])}, prices: {len(result['average_prices'])}")
        
        return jsonify(result)
    except Exception as e:
        print(f"[DEBUG] Seasonality error: {e}")
        return jsonify({"error": str(e)}), 500
@app.route("/api/seasonality/plot", methods=["GET"])
def seasonality_plot():
    asset      = request.args.get("asset",      type=str)
    years_back = request.args.get("years_back", type=int)
    start_day  = request.args.get("start_day",  default=None, type=str)
    end_day    = request.args.get("end_day",    default=None, type=str)

    data = get_seasonality(asset, years_back, start_day, end_day)
    dates = data["dates"]
    vals  = data["average_prices"]

    # Se non ci sono dati
    if not dates or not vals:
        return jsonify({"error": "Nessun dato seasonality da plottare"}), 400

    # Crea il plot usando indici numerici
    import io
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8,4))
    x = list(range(len(dates)))
    ax.plot(x, vals, marker="o", linewidth=2, label="Seasonality")

    # Etichette personalizzate sull'asse x: mostrane solo alcune per non sovraccaricare
    step = max(1, len(dates)//10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([dates[i] for i in x[::step]], rotation=45, fontsize=8)

    ax.set_title(f"Seasonality {asset} ({start_day}→{end_day}, {years_back} yrs)")
    ax.set_xlabel("Date (MM-DD)")
    ax.set_ylabel("Normalized Price")
    ax.grid(True)
    ax.legend()

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return send_file(buf, mimetype="image/png", download_name=f"seasonality_{asset}.png")

        
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



@app.route("/api/pattern-statistics", methods=["GET"])
def pattern_statistics():
    # 1. Estrai e valida i parametri
    asset     = request.args.get("asset",     type=str)
    start_day = request.args.get("start_day", type=int)
    end_day   = request.args.get("end_day",   type=int)
    if not asset or start_day is None or end_day is None:
        return jsonify({"error": "asset, start_day e end_day sono obbligatori"}), 400

    # 2. Scarica dati storici dal 1° gennaio dell'anno più vecchio
    #    fino a oggi
    today = datetime.date.today().isoformat()
    df    = get_historical_data(asset, "2000-01-01", today)

    # 3. Calcola le statistiche
    stats_list = get_yearly_pattern_statistics(df, start_day, end_day)

    # 4. Restituisci la lista JSON
    return jsonify(stats_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
