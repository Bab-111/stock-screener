# 📈 Stock Breakout Screener

A fully automated, cloud-based stock screener that runs in **GitHub Actions** and publishes a live HTML dashboard to **GitHub Pages** — accessible from any phone, tablet, or laptop, for free.

---

## 🚀 What It Does

Every 30 minutes during market hours (9:30 AM → 9:00 PM ET), the automation:

1. Scans a large universe of US stocks (configurable via CSV)
2. Applies breakout filters:
   - Volume spike ≥ 2× average
   - Large breakout candle (close > open × 1.03)
   - Price above 200-day moving average
   - Market cap ≥ $5 billion
3. Adds institutional signals:
   - Institutional ownership %
   - Money Flow Index (MFI)
   - Sector rotation alignment
   - Implied volatility from options chain
   - Historical breakout outcome (did it go up last time?)
4. Scores each stock 0–17 using a weighted conviction model
5. Publishes a colour-coded HTML dashboard to GitHub Pages

---

## 📂 Repo Structure

```
stock-screener/
├── scripts/
│   └── daily_report.py        ← main screener logic
├── config/
│   ├── config.json            ← all parameters (edit here)
│   └── universe.csv           ← list of tickers to scan
├── output/                    ← auto-generated report (do not edit)
├── requirements.txt           ← Python packages
└── .github/
    └── workflows/
        └── daily-report.yml   ← GitHub Actions schedule
```

---

## ⚙️ Setup (One-Time)

### Step 1 — Fork or create the repo
Create a **public** GitHub repository named `stock-screener`.

### Step 2 — Copy all files
Upload all files exactly as structured above.

### Step 3 — Enable Actions permissions
Go to **Settings → Actions → General → Workflow permissions**
→ Select **"Read and write permissions"**
→ Check **"Allow GitHub Actions to create and approve pull requests"**
→ Save

### Step 4 — Run manually first
Go to **Actions → Stock Breakout Screener → Run workflow**
Wait for the green ✅ — this creates the `gh-pages` branch.

### Step 5 — Enable GitHub Pages
Go to **Settings → Pages**
→ Branch: `gh-pages` / folder: `/`
→ Save

Your dashboard is now live at:
```
https://<your-username>.github.io/stock-screener
```

---

## 🔧 How to Customize

All tunable parameters live in **`config/config.json`** — no Python editing needed.

| Parameter | Default | Meaning |
|---|---|---|
| `volume_spike_threshold` | 2.0 | Min volume multiplier vs average |
| `breakout_threshold` | 1.03 | Close must be > Open × this value |
| `ma_period` | 200 | Moving average period (days) |
| `market_cap_min` | 5000000000 | Min market cap ($5B) |
| `top_picks` | 5 | How many stocks to highlight |
| `mfi_threshold` | 50 | Min Money Flow Index |
| `iv_high_threshold` | 25 | IV % considered "high" |
| `inst_ownership_high` | 60 | Institutional ownership % = strong |
| `alpha_vantage_api_key` | `YOUR_ALPHA_VANTAGE_KEY` | Optional — for institutional data |

To change the stock universe, edit **`config/universe.csv`** (one ticker per row under the `Symbol` header).

---

## 📊 Conviction Scoring (Total: 17 points)

| Factor | Weight | Strong (green) |
|---|---|---|
| Institutional Ownership | 3 | > 60% |
| Sector Alignment | 3 | Top-ranked sector today |
| Volume Spike | 2 | ≥ 2× average |
| Breakout Candle | 2 | Close > Open × 1.03 |
| MA200 | 2 | Price above 200-day MA |
| MFI (Money Flow Index) | 2 | > 60 |
| Historical Breakout | 2 | Last similar breakout → +5%+ |
| Implied Volatility | 1 | IV > 25% (market expects move) |

Traffic lights:
- 🟢 Strong conviction ≥ 75% of max score
- 🟡 Moderate ≥ 55%
- 🔴 Weak < 55%

---

## 🔑 Alpha Vantage API (Optional but Recommended)

For institutional ownership data and real MFI:
1. Get a free key at [alphavantage.co](https://www.alphavantage.co)
2. Add it to `config/config.json` → `"alpha_vantage_api_key": "YOUR_KEY"`

Without the key, the screener uses yfinance fallbacks and still works.

---

## ⚠️ Disclaimer

This tool is for **informational purposes only** and does not constitute financial advice. Always do your own research before making investment decisions.
