from flask import Flask, render_template, request, send_file, jsonify
import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.stats import t
from isyatirimhisse import StockData
import matplotlib.pyplot as plt
import io
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

plt.style.use('fivethirtyeight')

app = Flask(__name__)

def generate_buy_sell_signals(symbol, start_date='01-01-2003', frequency='1d', exchange='0', alpha=0.2, buy_threshold=40, sell_threshold=60, buffer_percentage=0, RSI_length=14):
    stock_data = StockData()
    data = stock_data.get_data(symbols=symbol, start_date=start_date, frequency=frequency, exchange=exchange)

    data = data[['DATE', 'CLOSING_TL']]
    data['CLOSING_TL'] = np.log(data['CLOSING_TL'])
    data = data.rename(columns={'CLOSING_TL': symbol})
    data['DATE'] = pd.to_datetime(data['DATE'])
    data = data.set_index('DATE')

    data['RSI'] = ta.rsi(close=data[symbol], length=RSI_length)
    data['Time'] = np.arange(1, len(data) + 1)
    linreg_result = ta.linear_regression(data['Time'], data[symbol])
    data['Fitted'] = linreg_result['line']
    data = data.dropna()

    degrees_of_freedom = len(data) - 2
    t_crit = t.ppf(1 - alpha / 2, degrees_of_freedom)
    beta0 = linreg_result['a']
    beta1 = linreg_result['b']
    data['Lower'] = beta0 + beta1 * data['Time'] - t_crit * np.sqrt(np.mean((data[symbol] - data['Fitted']) ** 2) * (1 + 1 / len(data) + (data['Time'] - data['Time'].mean()) ** 2 / ((data['Time'] - data['Time'].mean()) ** 2).sum()))
    data['Upper'] = beta0 + beta1 * data['Time'] + t_crit * np.sqrt(np.mean((data[symbol] - data['Fitted']) ** 2) * (1 + 1 / len(data) + (data['Time'] - data['Time'].mean()) ** 2 / ((data['Time'] - data['Time'].mean()) ** 2).sum()))

    buy_signals = []
    sell_signals = []
    last_signal = None

    for i in range(len(data)):
        if data['RSI'].iloc[i] > sell_threshold:
            if last_signal != 'sell' and last_signal is not None:
                value_upper = data['Upper'].iloc[i] * (1 + buffer_percentage / 100) if data['Upper'].iloc[i] >= 0 else data['Upper'].iloc[i] * (1 - buffer_percentage / 100)
                if data[symbol].iloc[i] > value_upper:
                    sell_signals.append(data[symbol].iloc[i])
                    buy_signals.append(np.nan)
                    last_signal = 'sell'
                else:
                    sell_signals.append(np.nan)
                    buy_signals.append(np.nan)
            else:
                sell_signals.append(np.nan)
                buy_signals.append(np.nan)
        elif data['RSI'].iloc[i] < buy_threshold:
            if last_signal != 'buy':
                value_lower = data['Lower'].iloc[i] * (1 - buffer_percentage / 100) if data['Lower'].iloc[i] >= 0 else data['Lower'].iloc[i] * (1 + buffer_percentage / 100)
                if data[symbol].iloc[i] < value_lower:
                    buy_signals.append(data[symbol].iloc[i])
                    sell_signals.append(np.nan)
                    last_signal = 'buy'
                else:
                    buy_signals.append(np.nan)
                    sell_signals.append(np.nan)
            else:
                buy_signals.append(np.nan)
                sell_signals.append(np.nan)
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)

    data['Buy_Signal_Price'] = buy_signals
    data['Sell_Signal_Price'] = sell_signals

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))

    ax1.set_title(f'{symbol} - for Educational Purposes Only', fontsize=16)

    ax1.plot(data.index, data[symbol], color='gray', alpha=.5)
    ax1.set_ylabel("Log Price")

    ax1.plot(data.index, data['Fitted'], label='Fitted', linestyle='--', color='blue')
    ax1.plot(data.index, data['Lower'], label='Lower Band', linestyle='--', color='orange')
    ax1.plot(data.index, data['Upper'], label='Upper Band', linestyle='--', color='purple')

    buy_signal_indices = data.index[data['Buy_Signal_Price'].notnull()]
    ax1.plot(buy_signal_indices, data.loc[buy_signal_indices, symbol], '^', markersize=12, color='green', label='Buy Signal')
    sell_signal_indices = data.index[data['Sell_Signal_Price'].notnull()]
    ax1.plot(sell_signal_indices, data.loc[sell_signal_indices, symbol], 'v', markersize=12, color='red', label='Sell Signal')

    ax1.legend()

    ax2.plot(data.index, data['RSI'], color='red')
    ax2.axhline(y=buy_threshold, color='black', linestyle='--')
    ax2.axhline(y=sell_threshold, color='black', linestyle='--')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    plt.close()

    return img

def fetch_news(stock_symbol=None):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get("https://www.kap.org.tr/tr/")
    time.sleep(5)
    xhr_requests = driver.execute_script("return window.performance.getEntriesByType('resource')")
    api = [request["name"] for request in xhr_requests if request["initiatorType"] == "xmlhttprequest"]
    driver.quit()

    r = requests.get(api[2]).json()
    news_list = []
    basic = [item["basic"] for item in r]
    for item in basic:
        disclosure_index = item["disclosureIndex"]
        company_code = item["stockCodes"]
        main_title = item["mainTitle"]  # Change 'mainTitle' to the actual key for the main title
        pdf_url = f"https://www.kap.org.tr/tr/BildirimPdf/{disclosure_index}"
        news_list.append({"company_code": company_code, "title": main_title, "pdf_url": pdf_url})
        if len(news_list) >= 20:  # Limit to the first 20 news items
            break
    return news_list

@app.route('/', methods=['GET', 'POST'])
def home():
    stock_code = None
    if request.method == 'POST':
        stock_code = request.form['stock']
    return render_template('index.html', stock_code=stock_code)

@app.route('/plot_image/<symbol>')
def plot_image(symbol):
    img = generate_buy_sell_signals(symbol)
    return send_file(img, mimetype='image/png')

@app.route('/fetch_news')
def fetch_news_endpoint():
    stock_symbol = request.args.get('symbol')
    news = fetch_news(stock_symbol)
    return jsonify(news=news)

if __name__ == '__main__':
    app.run(debug=True)
