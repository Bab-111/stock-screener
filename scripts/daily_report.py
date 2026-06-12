"""
Stock Breakout Screener
Screens stocks for breakout signals with institutional confirmation.
Runs via GitHub Actions and publishes HTML report to GitHub Pages.
"""

import json
import os
import sys
import requests
import warnings
import traceback
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "config.json")
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Config ────────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

UNIVERSE_FILE   = os.path.join(ROOT, "config", CFG["universe"])
VOL_THRESHOLD   = float(CFG["volume_spike_threshold"])
BRK_THRESHOLD   = float(CFG["breakout_threshold"])
MA_PERIOD       = int(CFG["ma_period"])
MCAP_MIN        = float(CFG["market_cap_min"])
TOP_N           = int(CFG["top_picks"])
MFI_PERIOD      = int(CFG["mfi_period"])
MFI_THRESHOLD   = float(CFG["mfi_threshold"])
IV_HIGH         = float(CFG["iv_high_threshold"])
IV_MOD          = float(CFG["iv_moderate_threshold"])
INST_HIGH       = float(CFG["inst_ownership_high"])
INST_MOD        = float(CFG["inst_ownership_moderate"])
HIST_DAYS       = int(CFG["history_days"])
FWD_DAYS        = int(CFG["forward_return_days"])
HIST_RET_THRESH = float(CFG["history_return_threshold"])
AV_KEY          = CFG.get("alpha_vantage_api_key", "")

# Scoring weights (total = 17)
WEIGHTS = {
    "inst_own":    3,
    "sector":      3,
    "volume":      2,
    "breakout":    2,
    "ma200":       2,
    "mfi":         2,
    "history":     2,
    "iv":          1,
}
MAX_SCORE = sum(WEIGHTS.values())

# ── Helpers ────────────────────────────────────────────────────────────────────

def classify(value, high_thresh, mod_thresh):
    """Return 'green', 'yellow', or 'red' based on thresholds."""
    if value is None:
        return "red"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "red"
    if v >= high_thresh:
        return "green"
    if v >= mod_thresh:
        return "yellow"
    return "red"


def traffic_light(score):
    """Return emoji traffic light based on total score."""
    pct = score / MAX_SCORE
    if pct >= 0.75:
        return "🟢"
    if pct >= 0.55:
        return "🟡"
    return "🔴"


def safe_float(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ── Data Fetchers ──────────────────────────────────────────────────────────────

def fetch_price_data(symbol):
    """Download 6-month daily OHLCV. Returns DataFrame or None."""
    try:
        data = yf.download(symbol, period="6mo", interval="1d",
                           progress=False, auto_adjust=True)
        if data is None or len(data) < MA_PERIOD // 2:
            return None
        return data
    except Exception:
        return None


def fetch_market_cap(symbol):
    """Return market cap in dollars using yfinance info."""
    try:
        info = yf.Ticker(symbol).info
        return safe_float(info.get("marketCap"))
    except Exception:
        return None


def fetch_implied_volatility(symbol):
    """Return average IV of nearest-expiry call options (as %)."""
    try:
        ticker = yf.Ticker(symbol)
        exps = ticker.options
        if not exps:
            return None
        chain = ticker.option_chain(exps[0])
        avg_iv = chain.calls["impliedVolatility"].dropna().mean()
        return round(float(avg_iv) * 100, 2)
    except Exception:
        return None


def fetch_institutional_data(symbol):
    """
    Fetch institutional ownership % and MFI from Alpha Vantage.
    Falls back to yfinance heldPercentInstitutions if no AV key.
    Returns (inst_pct, mfi_value).
    """
    inst_pct = None
    mfi_val  = None

    # -- Alpha Vantage path --
    if AV_KEY and AV_KEY != "YOUR_ALPHA_VANTAGE_KEY":
        try:
            url = (f"https://www.alphavantage.co/query"
                   f"?function=OVERVIEW&symbol={symbol}&apikey={AV_KEY}")
            r = requests.get(url, timeout=10).json()
            raw = r.get("InstitutionalOwnership") or r.get("PercentInstitutions")
            if raw:
                inst_pct = round(float(str(raw).replace("%", "")) * 100, 1)
        except Exception:
            pass

        try:
            url = (f"https://www.alphavantage.co/query"
                   f"?function=MFI&symbol={symbol}&interval=daily"
                   f"&time_period={MFI_PERIOD}&apikey={AV_KEY}")
            r = requests.get(url, timeout=10).json()
            series = r.get("Technical Analysis: MFI", {})
            if series:
                latest_date = sorted(series.keys())[-1]
                mfi_val = round(float(series[latest_date]["MFI"]), 1)
        except Exception:
            pass

    # -- yfinance fallback --
    if inst_pct is None:
        try:
            info = yf.Ticker(symbol).info
            raw = info.get("heldPercentInstitutions")
            if raw is not None:
                inst_pct = round(float(raw) * 100, 1)
        except Exception:
            pass

    return inst_pct, mfi_val


def fetch_sector_rotation():
    """
    Return dict of {sector_name: daily_pct_change} from Alpha Vantage,
    or a yfinance-based approximation using sector ETFs.
    """
    sector_etfs = {
        "Technology":    "XLK",
        "Healthcare":    "XLV",
        "Financials":    "XLF",
        "Industrials":   "XLI",
        "Energy":        "XLE",
        "Consumer Disc": "XLY",
        "Consumer Stpl": "XLP",
        "Materials":     "XLB",
        "Real Estate":   "XLRE",
        "Utilities":     "XLU",
        "Comm. Services":"XLC",
    }

    if AV_KEY and AV_KEY != "YOUR_ALPHA_VANTAGE_KEY":
        try:
            url = (f"https://www.alphavantage.co/query"
                   f"?function=SECTOR&apikey={AV_KEY}")
            r = requests.get(url, timeout=10).json()
            daily = r.get("Rank A: Real-Time Performance", {})
            if daily:
                return {k: float(v.replace("%", ""))
                        for k, v in daily.items()}
        except Exception:
            pass

    # fallback: compute 1-day % change from sector ETFs
    result = {}
    for sector, etf in sector_etfs.items():
        try:
            data = yf.download(etf, period="5d", interval="1d",
                               progress=False, auto_adjust=True)
            if len(data) >= 2:
                pct = float((data["Close"].iloc[-1] - data["Close"].iloc[-2])
                            / data["Close"].iloc[-2] * 100)
                result[sector] = round(pct, 2)
        except Exception:
            pass
    return result


def get_stock_sector(symbol):
    """Return sector name for a symbol using yfinance."""
    try:
        return yf.Ticker(symbol).info.get("sector", "Unknown")
    except Exception:
        return "Unknown"


def last_breakout_history(data):
    """
    Find the most recent prior high-volume breakout (excluding today)
    and return (date_str, forward_return_pct) or (None, None).
    """
    try:
        closes   = data["Close"].values
        volumes  = data["Volume"].values
        avg_vol  = float(np.mean(volumes))

        if len(closes) < MA_PERIOD:
            ma_vals = np.full(len(closes), np.nan)
        else:
            ma_vals = np.array(
                [np.mean(closes[max(0, i - MA_PERIOD):i])
                 for i in range(len(closes))]
            )

        # Search backwards, skip last row (today)
        for i in range(len(data) - 2, 0, -1):
            vol_spike   = volumes[i] >= VOL_THRESHOLD * avg_vol
            breakout_ok = closes[i] > data["Open"].values[i] * BRK_THRESHOLD
            above_ma    = (not np.isnan(ma_vals[i])) and closes[i] > ma_vals[i]

            if vol_spike and breakout_ok and above_ma:
                date_str = data.index[i].strftime("%b %d, %Y")
                fwd_idx  = min(i + FWD_DAYS, len(closes) - 1)
                fwd_ret  = round(
                    (closes[fwd_idx] - closes[i]) / closes[i] * 100, 1
                )
                return date_str, fwd_ret

    except Exception:
        pass
    return None, None


# ── MFI Calculation (local fallback) ──────────────────────────────────────────

def calc_mfi(data, period=14):
    """Calculate Money Flow Index from OHLCV data."""
    try:
        tp   = (data["High"] + data["Low"] + data["Close"]) / 3
        rmf  = tp * data["Volume"]
        pos  = rmf.where(tp > tp.shift(1), 0)
        neg  = rmf.where(tp < tp.shift(1), 0)
        pos_sum = pos.rolling(period).sum()
        neg_sum = neg.rolling(period).sum()
        mfr  = pos_sum / neg_sum.replace(0, 1e-9)
        mfi  = 100 - (100 / (1 + mfr))
        val  = mfi.iloc[-1]
        return round(float(val), 1) if not np.isnan(val) else None
    except Exception:
        return None


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_stock(vol_ratio, breakout, above_ma, inst_pct, mfi_val,
                sector_rank, iv_val, hist_ret):
    """
    Compute weighted conviction score and per-factor classification.
    Returns (total_score, factors_dict).
    """
    factors = {}
    score   = 0

    # Volume (weight 2)
    vol_cls = classify(vol_ratio, VOL_THRESHOLD, 1.5)
    factors["volume"] = vol_cls
    score += WEIGHTS["volume"] if vol_cls == "green" else (
             WEIGHTS["volume"] // 2 if vol_cls == "yellow" else 0)

    # Breakout (weight 2)
    factors["breakout"] = "green" if breakout else "red"
    score += WEIGHTS["breakout"] if breakout else 0

    # MA200 (weight 2)
    factors["ma200"] = "green" if above_ma else "red"
    score += WEIGHTS["ma200"] if above_ma else 0

    # Institutional ownership (weight 3)
    inst_cls = classify(inst_pct, INST_HIGH, INST_MOD)
    factors["inst_own"] = inst_cls
    score += WEIGHTS["inst_own"] if inst_cls == "green" else (
             WEIGHTS["inst_own"] // 2 if inst_cls == "yellow" else 0)

    # MFI (weight 2)
    mfi_cls = classify(mfi_val, MFI_THRESHOLD + 10, MFI_THRESHOLD)
    factors["mfi"] = mfi_cls
    score += WEIGHTS["mfi"] if mfi_cls == "green" else (
             WEIGHTS["mfi"] // 2 if mfi_cls == "yellow" else 0)

    # Sector alignment (weight 3)
    if sector_rank == 1:
        factors["sector"] = "green"
        score += WEIGHTS["sector"]
    elif sector_rank <= 3:
        factors["sector"] = "yellow"
        score += WEIGHTS["sector"] // 2
    else:
        factors["sector"] = "red"

    # Implied volatility (weight 1) — elevated IV = options expect move
    iv_cls = classify(iv_val, IV_HIGH, IV_MOD)
    factors["iv"] = iv_cls
    score += WEIGHTS["iv"] if iv_cls == "green" else (
             WEIGHTS["iv"] // 2 if iv_cls == "yellow" else 0)

    # Historical breakout success (weight 2)
    if hist_ret is not None and hist_ret >= HIST_RET_THRESH:
        factors["history"] = "green"
        score += WEIGHTS["history"]
    elif hist_ret is not None and hist_ret > 0:
        factors["history"] = "yellow"
        score += WEIGHTS["history"] // 2
    else:
        factors["history"] = "red"

    return score, factors


# ── Charts ─────────────────────────────────────────────────────────────────────

def generate_sector_chart(sector_data):
    """Save a bar chart of sector performance to output/sector_rotation.png."""
    if not sector_data:
        return

    sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1], reverse=True)
    names  = [s[0] for s in sorted_sectors]
    values = [s[1] for s in sorted_sectors]
    colors = ["#4caf50" if v > 0 else "#f44336" for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Daily % Change")
    ax.set_title("Sector Rotation – Today's Performance", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}%", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sector_rotation.png"), dpi=120)
    plt.close()


def generate_score_chart(results):
    """Save a stacked horizontal bar chart of factor contributions."""
    if not results:
        return

    factor_order = ["inst_own", "sector", "volume", "breakout",
                    "ma200", "mfi", "history", "iv"]
    factor_labels = {
        "inst_own":  "Inst. Own",
        "sector":    "Sector",
        "volume":    "Volume",
        "breakout":  "Breakout",
        "ma200":     "MA200",
        "mfi":       "MFI",
        "history":   "History",
        "iv":        "IV",
    }
    color_map = {"green": "#4caf50", "yellow": "#ffeb3b", "red": "#ef9a9a"}

    symbols = [r["symbol"] for r in results]
    fig, ax = plt.subplots(figsize=(10, max(4, len(symbols) * 0.9)))

    for i, res in enumerate(results):
        left = 0
        for fkey in factor_order:
            cls = res["factors"].get(fkey, "red")
            w   = WEIGHTS.get(fkey, 1)
            pts = w if cls == "green" else (w // 2 if cls == "yellow" else 0)
            if pts > 0:
                ax.barh(i, pts, left=left,
                        color=color_map[cls], edgecolor="white", linewidth=0.5)
                ax.text(left + pts / 2, i, factor_labels[fkey],
                        ha="center", va="center", fontsize=7, color="black")
                left += pts

    ax.set_yticks(range(len(symbols)))
    ax.set_yticklabels([f"{r['tl']} {r['symbol']}" for r in symbols], fontsize=10)
    ax.set_xlabel("Conviction Points")
    ax.set_xlim(0, MAX_SCORE + 1)
    ax.set_title("Factor Contribution by Stock", fontsize=12, fontweight="bold")

    green_p  = mpatches.Patch(color="#4caf50", label="Strong")
    yellow_p = mpatches.Patch(color="#ffeb3b", label="Moderate")
    red_p    = mpatches.Patch(color="#ef9a9a", label="Weak")
    ax.legend(handles=[green_p, yellow_p, red_p], loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "score_chart.png"), dpi=120)
    plt.close()


# ── HTML Report ────────────────────────────────────────────────────────────────

COLOR_MAP = {
    "green":  "#c8e6c9",
    "yellow": "#fff9c4",
    "red":    "#ffcdd2",
}

def cls_badge(cls):
    colors = {"green": "#2e7d32", "yellow": "#f57f17", "red": "#b71c1c"}
    bg     = {"green": "#e8f5e9", "yellow": "#fffde7", "red": "#ffebee"}
    icons  = {"green": "✅", "yellow": "⚠️", "red": "❌"}
    c = cls if cls in colors else "red"
    return (f'<span style="background:{bg[c]};color:{colors[c]};'
            f'padding:2px 6px;border-radius:4px;font-size:11px;">'
            f'{icons[c]}</span>')


def build_html(results, sector_data, run_time, green_pct, yellow_pct, red_pct):
    sentiment_label = (
        "🐂 Bullish" if green_pct >= 50 else
        "😐 Neutral" if green_pct >= 30 else
        "🐻 Bearish"
    )

    rows = ""
    for r in results:
        f = r["factors"]
        vol_txt  = f"{r['vol_ratio']:.1f}× avg"
        brk_txt  = f"Close {r['close']:.2f} vs Open {r['open']:.2f}"
        ma_txt   = f"{r['close']:.2f} vs MA {r['ma200']:.2f}"
        inst_txt = f"{r['inst_pct']}%" if r['inst_pct'] else "N/A"
        mfi_txt  = str(r['mfi_val']) if r['mfi_val'] else "N/A"
        iv_txt   = f"{r['iv_val']}%" if r['iv_val'] else "N/A"
        hist_txt = (f"{r['hist_date']} → {r['hist_ret']:+.1f}%"
                    if r['hist_date'] else "No recent breakout")
        sec_txt  = r['sector']

        rows += f"""
        <tr>
          <td style="font-weight:600;font-size:15px;">{r['tl']} {r['symbol']}</td>
          <td style="background:{COLOR_MAP[f['volume']]};">{vol_txt}</td>
          <td style="background:{COLOR_MAP[f['breakout']]};">{brk_txt}</td>
          <td style="background:{COLOR_MAP[f['ma200']]};">{ma_txt}</td>
          <td style="background:{COLOR_MAP[f['inst_own']]};">{inst_txt}</td>
          <td style="background:{COLOR_MAP[f['mfi']]};">{mfi_txt}</td>
          <td style="background:{COLOR_MAP[f['sector']]};">{sec_txt}</td>
          <td style="background:{COLOR_MAP[f['iv']]};">{iv_txt}</td>
          <td style="background:{COLOR_MAP[f['history']]};">{hist_txt}</td>
          <td style="font-weight:700;font-size:15px;">{r['score']}/{MAX_SCORE}</td>
        </tr>"""

    top5_rows = ""
    for i, r in enumerate(results, 1):
        f = r["factors"]
        bullets = [
            f"Volume: {r['vol_ratio']:.1f}× average daily volume {cls_badge(f['volume'])}",
            f"Breakout candle: Close {r['close']:.2f} above {BRK_THRESHOLD:.0%} of Open {cls_badge(f['breakout'])}",
            f"Price vs 200-Day MA: {r['close']:.2f} vs {r['ma200']:.2f} {cls_badge(f['ma200'])}",
            f"Institutional Ownership: {r['inst_pct']}% | MFI: {r['mfi_val']} {cls_badge(f['inst_own'])}",
        ]
        bullet_html = "".join(f"<li style='margin:4px 0;'>{b}</li>" for b in bullets)
        mcap_b = f"{r['mcap']/1e9:.1f}B" if r['mcap'] else "N/A"
        top5_rows += f"""
        <div style="background:#fff;border:1px solid #e0e0e0;border-radius:8px;
                    padding:14px 18px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,0.07);">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
            <span style="font-size:22px;">{r['tl']}</span>
            <span style="font-size:18px;font-weight:700;">{i}. {r['symbol']}</span>
            <span style="font-size:13px;color:#666;">({r['sector']})</span>
            <span style="margin-left:auto;background:#1a237e;color:#fff;
                         padding:3px 10px;border-radius:12px;font-size:13px;">
              Score: {r['score']}/{MAX_SCORE}
            </span>
            <span style="font-size:12px;color:#555;">Market Cap: ${mcap_b}</span>
          </div>
          <ul style="margin:0;padding-left:18px;font-size:13px;color:#333;">
            {bullet_html}
          </ul>
        </div>"""

    sector_rows = ""
    if sector_data:
        sorted_s = sorted(sector_data.items(), key=lambda x: x[1], reverse=True)
        for rank, (sec, pct) in enumerate(sorted_s, 1):
            clr = "#4caf50" if pct > 0 else "#f44336"
            sector_rows += f"""
            <tr>
              <td style="font-weight:600;">#{rank} {sec}</td>
              <td style="color:{clr};font-weight:700;">{pct:+.2f}%</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Stock Breakout Screener – {run_time}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f6fa;
           margin: 0; padding: 16px; color: #222; }}
    h1   {{ font-size: 22px; margin: 0 0 4px; color: #1a237e; }}
    h2   {{ font-size: 16px; margin: 24px 0 10px; color: #283593; border-bottom: 2px solid #3949ab;
           padding-bottom: 4px; }}
    .header {{ background: #1a237e; color: #fff; padding: 14px 20px;
              border-radius: 8px; margin-bottom: 18px; }}
    .run-time {{ font-size: 12px; color: #b0bec5; margin-top: 4px; }}
    .sentiment-bar {{ background: #fff; border-radius: 8px; padding: 14px 18px;
                     margin-bottom: 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.07); }}
    .bar-wrap {{ display: flex; height: 22px; border-radius: 6px; overflow: hidden;
                margin: 8px 0; }}
    .bar-seg  {{ height: 100%; transition: width 0.3s; }}
    .legend   {{ font-size: 12px; color: #555; margin-top: 6px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff;
            border-radius: 8px; overflow: hidden;
            box-shadow: 0 1px 4px rgba(0,0,0,0.07); font-size: 12px; }}
    th {{ background: #1a237e; color: #fff; padding: 9px 8px;
         text-align: center; font-size: 11px; }}
    td {{ padding: 8px; text-align: center; border-bottom: 1px solid #eeeeee; }}
    tr:last-child td {{ border-bottom: none; }}
    img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,0.1); }}
    .two-col {{ display: flex; gap: 18px; flex-wrap: wrap; }}
    .two-col > div {{ flex: 1; min-width: 260px; }}
    .sector-table {{ background: #fff; border-radius: 8px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.07); }}
    .sector-table th {{ background: #283593; }}
    .disclaimer {{ font-size: 11px; color: #888; margin-top: 20px;
                  border-top: 1px solid #ddd; padding-top: 10px; }}
  </style>
</head>
<body>

<div class="header">
  <h1>📈 Stock Breakout Screener Dashboard</h1>
  <div class="run-time">Last updated: {run_time} UTC &nbsp;|&nbsp;
    Top {TOP_N} picks from {len(results)} qualifying stocks &nbsp;|&nbsp;
    Universe: large-cap ≥ $5B market cap</div>
</div>

<!-- Sentiment Bar -->
<div class="sentiment-bar">
  <strong>Overall Market Sentiment: {sentiment_label}</strong>
  <div class="bar-wrap">
    <div class="bar-seg" style="width:{green_pct:.0f}%;background:#4caf50;"></div>
    <div class="bar-seg" style="width:{yellow_pct:.0f}%;background:#ffeb3b;"></div>
    <div class="bar-seg" style="width:{red_pct:.0f}%;background:#f44336;"></div>
  </div>
  <div class="legend">
    🟢 Strong {green_pct:.0f}% &nbsp;&nbsp;
    🟡 Moderate {yellow_pct:.0f}% &nbsp;&nbsp;
    🔴 Weak {red_pct:.0f}%
  </div>
</div>

<!-- Top 5 Picks -->
<h2>🏆 Top {TOP_N} Conviction Picks</h2>
{top5_rows}

<!-- Detailed Table -->
<h2>📊 Detailed Conviction Dashboard</h2>
<div style="overflow-x:auto;">
<table>
  <tr>
    <th>Stock</th>
    <th>Volume (w:{WEIGHTS['volume']})</th>
    <th>Breakout (w:{WEIGHTS['breakout']})</th>
    <th>MA200 (w:{WEIGHTS['ma200']})</th>
    <th>Inst. Own (w:{WEIGHTS['inst_own']})</th>
    <th>MFI (w:{WEIGHTS['mfi']})</th>
    <th>Sector (w:{WEIGHTS['sector']})</th>
    <th>IV (w:{WEIGHTS['iv']})</th>
    <th>History (w:{WEIGHTS['history']})</th>
    <th>Score /{MAX_SCORE}</th>
  </tr>
  {rows}
</table>
</div>

<!-- Factor Chart -->
<h2>🎯 Factor Contribution Chart</h2>
<img src="score_chart.png" alt="Score breakdown by factor">

<!-- Sector + Rotation -->
<div class="two-col" style="margin-top:24px;">
  <div>
    <h2>🌀 Sector Rotation</h2>
    <img src="sector_rotation.png" alt="Sector rotation chart">
  </div>
  <div>
    <h2>📋 Sector Ranking Today</h2>
    <table class="sector-table">
      <tr><th>Sector</th><th>Daily Change</th></tr>
      {sector_rows}
    </table>
  </div>
</div>

<div class="disclaimer">
  ⚠️ This report is for informational purposes only and does not constitute financial advice.
  Always do your own research before making investment decisions.
  Scoring weights: Institutional Ownership (3) · Sector Alignment (3) ·
  Volume Spike (2) · Breakout Candle (2) · MA200 (2) · MFI (2) · History (2) · IV (1).
</div>

</body>
</html>"""

    return html


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"[{datetime.utcnow():%Y-%m-%d %H:%M}] Starting stock screener...")

    # Load universe
    symbols = pd.read_csv(UNIVERSE_FILE)["Symbol"].dropna().tolist()
    print(f"  Universe: {len(symbols)} tickers")

    # Sector rotation
    print("  Fetching sector rotation...")
    sector_data = fetch_sector_rotation()
    sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1], reverse=True)
    sector_rank_map = {s: i + 1 for i, (s, _) in enumerate(sorted_sectors)}

    results = []

    for symbol in symbols:
        try:
            print(f"  Screening {symbol}...", end=" ")

            # 1. Price data
            data = fetch_price_data(symbol)
            if data is None or len(data) < 30:
                print("skip (no data)")
                continue

            latest = data.iloc[-1]
            close  = float(latest["Close"])
            open_  = float(latest["Open"])
            vol    = float(latest["Volume"])
            avg_vol = float(data["Volume"].mean())

            # 2. Basic filters
            vol_ratio  = round(vol / avg_vol, 2) if avg_vol > 0 else 0
            breakout   = close > open_ * BRK_THRESHOLD
            ma_series  = data["Close"].rolling(MA_PERIOD).mean()
            ma200_val  = float(ma_series.iloc[-1]) if not pd.isna(ma_series.iloc[-1]) else None
            above_ma   = (ma200_val is not None) and (close > ma200_val)

            # Skip if doesn't pass core filters
            if not (vol_ratio >= 1.5 and breakout):
                print("skip (core filter)")
                continue

            # 3. Market cap filter
            mcap = fetch_market_cap(symbol)
            if mcap is not None and mcap < MCAP_MIN:
                print(f"skip (mcap {mcap/1e9:.1f}B < 5B)")
                continue

            # 4. Institutional data & MFI
            inst_pct, mfi_val = fetch_institutional_data(symbol)

            # Local MFI fallback
            if mfi_val is None:
                mfi_val = calc_mfi(data, MFI_PERIOD)

            # 5. Sector
            sector   = get_stock_sector(symbol)
            sec_rank = sector_rank_map.get(sector, 99)

            # 6. Implied volatility
            iv_val = fetch_implied_volatility(symbol)

            # 7. Historical breakout
            hist_date, hist_ret = last_breakout_history(data)

            # 8. Score
            score, factors = score_stock(
                vol_ratio, breakout, above_ma,
                inst_pct, mfi_val, sec_rank, iv_val, hist_ret
            )

            results.append({
                "symbol":    symbol,
                "score":     score,
                "tl":        traffic_light(score),
                "factors":   factors,
                "close":     close,
                "open":      open_,
                "vol_ratio": vol_ratio,
                "ma200":     ma200_val if ma200_val else 0,
                "mcap":      mcap,
                "inst_pct":  inst_pct,
                "mfi_val":   mfi_val,
                "sector":    sector,
                "iv_val":    iv_val,
                "hist_date": hist_date,
                "hist_ret":  hist_ret,
            })
            print(f"✓ score={score}/{MAX_SCORE}")

        except Exception as e:
            print(f"error: {e}")
            traceback.print_exc()

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:TOP_N]

    # Sentiment %
    total = len(results) if results else 1
    green_pct  = sum(1 for r in results if r["tl"] == "🟢") / total * 100
    yellow_pct = sum(1 for r in results if r["tl"] == "🟡") / total * 100
    red_pct    = sum(1 for r in results if r["tl"] == "🔴") / total * 100

    # Generate charts
    print("  Generating charts...")
    generate_sector_chart(sector_data)
    generate_score_chart(top_results)

    # Generate HTML
    run_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    html = build_html(top_results, sector_data, run_time, green_pct, yellow_pct, red_pct)
    out_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(out_path, "w") as f:
        f.write(html)

    print(f"  Report written: {out_path}")
    print(f"  Qualified stocks: {len(results)} | Top {TOP_N} shown")
    print("Done ✓")


if __name__ == "__main__":
    main()
