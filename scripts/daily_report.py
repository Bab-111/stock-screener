import yfinance as yf
import os

# Make sure output folder exists
os.makedirs("output", exist_ok=True)

# Small universe of tickers
symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

html = "<h1>Latest Prices</h1><ul>"

for symbol in symbols:
    try:
        data = yf.download(symbol, period="1d", interval="1d")
        latest_close = float(data["Close"].iloc[-1])
        html += f"<li>{symbol}: {latest_close}</li>"
    except Exception as e:
        html += f"<li>{symbol}: Error ({e})</li>"

html += "</ul>"

with open("output/index.html", "w") as f:
    f.write(html)

print("Generated report with", len(symbols), "stocks")
