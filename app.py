from flask import Flask, jsonify
import yfinance as yf

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({"message": "Backend pronto!"})

@app.route('/price/<ticker>')
def price(ticker):
    data = yf.Ticker(ticker).history(period="1mo")
    prices = data['Close'].tolist()
    return jsonify({ticker: prices})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
