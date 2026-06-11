import json
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# Load config
with open("config/config.json") as f:
    config = json.load(f)

# Load universe
symbols = pd.read_csv(f"config/{config['universe']}")["Symbol"].tolist()

def check_breakout(latest, breakout_threshold):
    return latest['Close'] > latest['Open'] * breakout_threshold

def generate_sector_chart():
    # Placeholder chart (replace with real sector data later)
    sectors = ["Tech", "Healthcare", "Industrials"]
    perf = [1.8, 1.2, 0.9]
    plt.bar(sectors, perf, color=["#4caf50", "#2196f3", "#ff9800"])
    plt.title("Sector Rotation (Past Week)")
    plt.savefig("output/sector_rotation.png")

def main():
    results = []
    for symbol in symbols:
        try:
            data = yf.download(symbol, period="6mo", interval="1d")
            avg_vol = data['Volume'].mean()
            latest = data.iloc[-1]
            ma200 = data['Close'].rolling(config["ma_period"]).mean().iloc[-1]

            volume_spike = latest['Volume'] >= config["volume_spike_threshold"] * avg_vol
            breakout = check_breakout(latest, config["breakout_threshold"])
            above_ma200 = latest['Close'] > ma200

            score = sum([volume_spike, breakout, above_ma200])  # simple scoring

            results.append((symbol, score, volume_spike, breakout, above_ma200))
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # Generate HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Daily Stock Screener</title></head>
    <body>
      <h1>Top Conviction Picks – {datetime.now().strftime("%Y-%m-%d")}</h1>
      <ol>
    """
    results.sort(key=lambda x: x[1], reverse=True)
    for symbol, score, vol, brk, ma in results[:10]:
        html += f"<li>{symbol} – Score {score}/3 (Vol={vol}, Breakout={brk}, MA200={ma})</li>"
    html += """
      </ol>
      <h2>Sector Rotation (Past Week)</h2>
      <img src="sector_rotation.png" width="600">
    </body>
    </html>
    """

    with open("output/index.html", "w") as f:
        f.write(html)

    generate_sector_chart()

if __name__ == "__main__":
    main()

