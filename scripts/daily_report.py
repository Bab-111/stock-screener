"""
Stock Breakout Screener
Screens stocks for breakout signals with institutional confirmation.
Runs via GitHub Actions and publishes HTML report to GitHub Pages.

Strategy: Score ALL stocks in universe, show top 5 by conviction.
Hard filters only remove stocks with truly missing data or sub-$5B cap.
"""

import json
import os
import sys
import requests
import warnings
import traceback
from datetime import datetime

import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "config.json")
OUTPUT_DIR  = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

UNIVERSE_FILE   = os.path.join(ROOT, "config", CFG["universe"])
VOL_THRESHOLD   = float(CFG["volume_spike_threshold"])      # 2.0
BRK_THRESHOLD   = float(CFG["breakout_threshold"])          # 1.03
MA_PERIOD       = int(CFG["ma_period"])                     # 200
MCAP_MIN        = float(CFG["market_cap_min"])              # 5B
TOP_N           = int(CFG["top_picks"])                     # 5
MFI_PERIOD      = int(CFG["mfi_period"])                    # 14
MFI_THRESHOLD   = float(CFG["mfi_threshold"])               # 50
IV_HIGH         = float(CFG["iv_high_threshold"])           # 25
IV_MOD          = float(CFG["iv_moderate_threshold"])       # 15
INST_HIGH       = float(CFG["inst_ownership_high"])         # 60
INST_MOD        = float(CFG["inst_ownership_moderate"])     # 40
HIST_RET_THRESH = float(CFG["history_return_threshold"])    # 5
FWD_DAYS        = int(CFG["forward_return_days"])           # 10
AV_KEY          = CFG.get("alpha_vantage_api_key", "")

# Scoring weights (max = 17)
WEIGHTS = {
    "inst_own": 3,
    "sector":   3,
    "volume":   2,
    "breakout": 2,
    "ma200":    2,
    "mfi":      2,
    "history":  2,
    "iv":       1,
}
MAX_SCORE = sum(WEIGHTS.values())  # 17

# ── Helpers ──────────────────────────────────────────────────────────────────

def safe_float(val, default=None):
    try:
        return float(val)
    except Exception:
        return default

def classify(value, high_thresh, mod_thresh):
    v = safe_float(value)
    if v is None:
        return "red"
    if v >= high_thresh:
        return "green"
    if v >= mod_thresh:
        return "yellow"
    return "red"

def traffic_light(score):
    pct = score / MAX_SCORE
    if pct >= 0.70:
        return "🟢"
    if pct >= 0.45:
        return "🟡"
    return "🔴"

# ── Data Fetchers ─────────────────────────────────────────────────────────────

def fetch_price_data(symbol):
    try:
        data = yf.download(symbol, period="12mo", interval="1d",
                           progress=False, auto_adjust=True)
        if data is None or len(data) < 20:
            return None
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception:
        return None

def fetch_market_cap(symbol):
    try:
        info = yf.Ticker(symbol).fast_info
        return safe_float(getattr(info, "market_cap", None))
    except Exception:
        return None

def fetch_implied_volatility(symbol):
    try:
        ticker = yf.Ticker(symbol)
        exps = ticker.options
        if not exps:
            return None
        chain = ticker.option_chain(exps[0])
        avg_iv = chain.calls["impliedVolatility"].dropna().mean()
        return round(float(avg_iv) * 100, 1)
    except Exception:
        return None

def fetch_institutional_ownership(symbol):
    """Return institutional ownership % (0-100). yfinance fallback."""
    # Alpha Vantage path
    if AV_KEY and AV_KEY not in ("YOUR_ALPHA_VANTAGE_KEY", ""):
        try:
            url = (f"https://www.alphavantage.co/query"
                   f"?function=OVERVIEW&symbol={symbol}&apikey={AV_KEY}")
            r = requests.get(url, timeout=8).json()
            raw = r.get("PercentInstitutions") or r.get("InstitutionalOwnership")
            if raw:
                v = safe_float(str(raw).replace("%", ""))
                if v and v <= 1.0:
                    v = round(v * 100, 1)
                return v
        except Exception:
            pass
    # yfinance fallback
    try:
        info = yf.Ticker(symbol).info
        raw = info.get("heldPercentInstitutions")
        if raw is not None:
            return round(float(raw) * 100, 1)
    except Exception:
        pass
    return None

def calc_mfi(data, period=14):
    """Money Flow Index — calculated from local OHLCV."""
    try:
        hi  = data["High"].squeeze()
        lo  = data["Low"].squeeze()
        cl  = data["Close"].squeeze()
        vol = data["Volume"].squeeze()
        tp  = (hi + lo + cl) / 3
        rmf = tp * vol
        pos = rmf.where(tp > tp.shift(1), 0.0)
        neg = rmf.where(tp < tp.shift(1), 0.0)
        pos_sum = pos.rolling(period).sum()
        neg_sum = neg.rolling(period).sum()
        mfr = pos_sum / neg_sum.replace(0, 1e-9)
        mfi = 100 - (100 / (1 + mfr))
        val = mfi.iloc[-1]
        return round(float(val), 1) if not np.isnan(float(val)) else None
    except Exception:
        return None

def fetch_sector_rotation():
    """Sector daily % change via sector ETFs (always free)."""
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
    result = {}
    for sector, etf in sector_etfs.items():
        try:
            d = yf.download(etf, period="5d", interval="1d",
                            progress=False, auto_adjust=True)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            if len(d) >= 2:
                cl = d["Close"].squeeze()
                pct = float((cl.iloc[-1] - cl.iloc[-2]) / cl.iloc[-2] * 100)
                result[sector] = round(pct, 2)
        except Exception:
            pass
    return result

def get_stock_sector(symbol):
    try:
        return yf.Ticker(symbol).info.get("sector", "Unknown") or "Unknown"
    except Exception:
        return "Unknown"

def last_breakout_history(data):
    """Find last high-volume breakout before today and its forward return."""
    try:
        closes  = data["Close"].squeeze().values.astype(float)
        opens   = data["Open"].squeeze().values.astype(float)
        volumes = data["Volume"].squeeze().values.astype(float)
        avg_vol = float(np.nanmean(volumes))

        # rolling MA200
        ma_vals = np.full(len(closes), np.nan)
        for i in range(MA_PERIOD, len(closes)):
            ma_vals[i] = np.mean(closes[i - MA_PERIOD:i])

        for i in range(len(data) - 2, max(0, len(data) - 120), -1):
            if (volumes[i] >= VOL_THRESHOLD * avg_vol and
                    closes[i] > opens[i] * BRK_THRESHOLD and
                    not np.isnan(ma_vals[i]) and closes[i] > ma_vals[i]):
                date_str = data.index[i].strftime("%b %d, %Y")
                fwd_idx  = min(i + FWD_DAYS, len(closes) - 1)
                fwd_ret  = round((closes[fwd_idx] - closes[i]) / closes[i] * 100, 1)
                return date_str, fwd_ret
    except Exception:
        pass
    return None, None

# ── Scoring ───────────────────────────────────────────────────────────────────

def score_stock(vol_ratio, breakout, above_ma, inst_pct,
                mfi_val, sector_rank, iv_val, hist_ret):
    factors = {}
    score   = 0

    # Volume (2pts) — partial credit for 1.2x+
    if vol_ratio >= VOL_THRESHOLD:
        factors["volume"] = "green"; score += WEIGHTS["volume"]
    elif vol_ratio >= 1.2:
        factors["volume"] = "yellow"; score += 1
    else:
        factors["volume"] = "red"

    # Breakout candle (2pts)
    if breakout:
        factors["breakout"] = "green"; score += WEIGHTS["breakout"]
    elif vol_ratio >= 1.5:   # strong volume even without huge candle
        factors["breakout"] = "yellow"; score += 1
    else:
        factors["breakout"] = "red"

    # MA200 (2pts)
    factors["ma200"] = "green" if above_ma else "red"
    score += WEIGHTS["ma200"] if above_ma else 0

    # Institutional ownership (3pts)
    inst_cls = classify(inst_pct, INST_HIGH, INST_MOD)
    factors["inst_own"] = inst_cls
    score += WEIGHTS["inst_own"] if inst_cls == "green" else (1 if inst_cls == "yellow" else 0)

    # MFI (2pts)
    mfi_cls = classify(mfi_val, MFI_THRESHOLD + 10, MFI_THRESHOLD)
    factors["mfi"] = mfi_cls
    score += WEIGHTS["mfi"] if mfi_cls == "green" else (1 if mfi_cls == "yellow" else 0)

    # Sector alignment (3pts)
    if sector_rank == 1:
        factors["sector"] = "green"; score += WEIGHTS["sector"]
    elif sector_rank <= 3:
        factors["sector"] = "yellow"; score += 1
    else:
        factors["sector"] = "red"

    # IV (1pt) — elevated IV = options market expects a move
    iv_cls = classify(iv_val, IV_HIGH, IV_MOD)
    factors["iv"] = iv_cls
    score += WEIGHTS["iv"] if iv_cls == "green" else 0

    # Historical breakout success (2pts)
    if hist_ret is not None and hist_ret >= HIST_RET_THRESH:
        factors["history"] = "green"; score += WEIGHTS["history"]
    elif hist_ret is not None and hist_ret > 0:
        factors["history"] = "yellow"; score += 1
    else:
        factors["history"] = "red"

    return score, factors

# ── Charts ────────────────────────────────────────────────────────────────────

def generate_sector_chart(sector_data):
    if not sector_data:
        return
    sorted_s = sorted(sector_data.items(), key=lambda x: x[1], reverse=True)
    names  = [s[0] for s in sorted_s]
    values = [s[1] for s in sorted_s]
    colors = ["#43a047" if v > 0 else "#e53935" for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("Daily % Change", fontsize=10)
    ax.set_title("Sector Rotation — Today's Performance", fontsize=12, fontweight="bold")
    for i, (name, val) in enumerate(zip(names[::-1], values[::-1])):
        ax.text(val + (0.03 if val >= 0 else -0.03),
                i, f"{val:+.2f}%", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sector_rotation.png"), dpi=120)
    plt.close()

def generate_score_chart(results):
    if not results:
        return
    factor_order  = ["inst_own","sector","volume","breakout","ma200","mfi","history","iv"]
    factor_labels = {"inst_own":"Inst.Own","sector":"Sector","volume":"Volume",
                     "breakout":"Breakout","ma200":"MA200","mfi":"MFI",
                     "history":"History","iv":"IV"}
    color_map = {"green":"#43a047","yellow":"#fdd835","red":"#ef9a9a"}

    fig, ax = plt.subplots(figsize=(10, max(3.5, len(results) * 1.0)))
    for i, res in enumerate(results):
        left = 0
        for fkey in factor_order:
            cls = res["factors"].get(fkey, "red")
            w   = WEIGHTS.get(fkey, 1)
            pts = w if cls == "green" else (1 if cls == "yellow" else 0)
            if pts > 0:
                ax.barh(i, pts, left=left, color=color_map[cls],
                        edgecolor="white", linewidth=0.6)
                ax.text(left + pts / 2, i, factor_labels[fkey],
                        ha="center", va="center", fontsize=7.5, color="#222")
                left += pts

    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([f"{r['tl']} {r['symbol']}" for r in results], fontsize=11)
    ax.set_xlabel("Conviction Points", fontsize=10)
    ax.set_xlim(0, MAX_SCORE + 0.5)
    ax.set_title("Factor Contribution by Stock", fontsize=12, fontweight="bold")
    green_p  = mpatches.Patch(color="#43a047", label="Strong")
    yellow_p = mpatches.Patch(color="#fdd835", label="Moderate")
    red_p    = mpatches.Patch(color="#ef9a9a", label="Weak/Missing")
    ax.legend(handles=[green_p, yellow_p, red_p], loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "score_chart.png"), dpi=120)
    plt.close()

# ── HTML ──────────────────────────────────────────────────────────────────────

COLOR_BG = {"green":"#c8e6c9","yellow":"#fff9c4","red":"#ffcdd2"}

def badge(cls):
    icons = {"green":"✅","yellow":"⚠️","red":"❌"}
    bg    = {"green":"#e8f5e9","yellow":"#fffde7","red":"#ffebee"}
    fg    = {"green":"#2e7d32","yellow":"#f57f17","red":"#b71c1c"}
    c = cls if cls in icons else "red"
    return (f'<span style="background:{bg[c]};color:{fg[c]};'
            f'padding:2px 7px;border-radius:4px;font-size:11px;">{icons[c]}</span>')

def build_html(results, sector_data, run_time, universe_size,
               green_pct, yellow_pct, red_pct):
    sentiment = ("🐂 Bullish" if green_pct >= 50 else
                 "😐 Neutral" if green_pct >= 25 else "🐻 Bearish")

    # ── Top picks cards ──
    top_cards = ""
    for i, r in enumerate(results, 1):
        f = r["factors"]
        mcap_str = f"${r['mcap']/1e9:.1f}B" if r["mcap"] else "N/A"
        iv_str   = f"{r['iv_val']}%" if r["iv_val"] else "N/A"
        inst_str = f"{r['inst_pct']}%" if r["inst_pct"] is not None else "N/A"
        mfi_str  = str(r["mfi_val"]) if r["mfi_val"] is not None else "N/A"
        hist_str = (f"{r['hist_date']} → {r['hist_ret']:+.1f}%"
                    if r["hist_date"] else "No prior breakout found")

        bullets = [
            (f["volume"],   f"Volume: <b>{r['vol_ratio']:.1f}×</b> average — "
                            f"{'strong institutional interest' if r['vol_ratio']>=2 else 'elevated activity'}"),
            (f["breakout"], f"Candle: Close <b>${r['close']:.2f}</b> vs Open <b>${r['open']:.2f}</b> "
                            f"(+{(r['close']/r['open']-1)*100:.1f}%)"),
            (f["ma200"],    f"200-Day MA: Price <b>${r['close']:.2f}</b> vs MA <b>${r['ma200']:.2f}</b> — "
                            f"{'above' if r['close'] > r['ma200'] else 'below'} long-term trend"),
            (f["inst_own"], f"Institutional ownership <b>{inst_str}</b> · "
                            f"MFI <b>{mfi_str}</b> · Market Cap <b>{mcap_str}</b>"),
            (f["history"],  f"Last breakout: {hist_str}"),
        ]
        bullet_html = "".join(
            f'<li style="margin:5px 0;">{badge(cls)} {txt}</li>'
            for cls, txt in bullets
        )

        top_cards += f"""
        <div style="background:#fff;border:1px solid #ddd;border-radius:10px;
                    padding:16px 20px;margin-bottom:14px;
                    box-shadow:0 2px 6px rgba(0,0,0,0.07);">
          <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:10px;">
            <span style="font-size:24px;">{r['tl']}</span>
            <span style="font-size:20px;font-weight:700;">{i}. {r['symbol']}</span>
            <span style="font-size:13px;color:#666;">{r['sector']}</span>
            <span style="margin-left:auto;background:#1a237e;color:#fff;
                         padding:4px 12px;border-radius:14px;font-size:13px;font-weight:600;">
              Score: {r['score']}/{MAX_SCORE}
            </span>
          </div>
          <ul style="margin:0;padding-left:20px;font-size:13px;color:#333;line-height:1.7;">
            {bullet_html}
          </ul>
        </div>"""

    # ── Detail table rows ──
    table_rows = ""
    for r in results:
        f = r["factors"]
        table_rows += f"""
        <tr>
          <td style="font-weight:700;text-align:left;padding-left:10px;">
            {r['tl']} {r['symbol']}</td>
          <td style="background:{COLOR_BG[f['volume']]};">{r['vol_ratio']:.1f}×</td>
          <td style="background:{COLOR_BG[f['breakout']]};">
            {'+' if r['close']>=r['open'] else ''}{(r['close']/r['open']-1)*100:.1f}%</td>
          <td style="background:{COLOR_BG[f['ma200']]};">
            {'▲ Above' if r['close']>r['ma200'] else '▼ Below'}</td>
          <td style="background:{COLOR_BG[f['inst_own']]};">
            {r['inst_pct']}%</td>
          <td style="background:{COLOR_BG[f['mfi']]};">{r['mfi_val'] or 'N/A'}</td>
          <td style="background:{COLOR_BG[f['sector']]};">{r['sector']}</td>
          <td style="background:{COLOR_BG[f['iv']]};">{r['iv_val'] or 'N/A'}%</td>
          <td style="background:{COLOR_BG[f['history']]};">
            {f"{r['hist_ret']:+.1f}%" if r['hist_ret'] is not None else 'N/A'}</td>
          <td style="font-weight:800;font-size:15px;">{r['score']}/{MAX_SCORE}</td>
        </tr>"""

    # ── Sector table ──
    sector_rows = ""
    for rank, (sec, pct) in enumerate(
            sorted(sector_data.items(), key=lambda x: x[1], reverse=True), 1):
        clr = "#2e7d32" if pct > 0 else "#c62828"
        bg  = "#e8f5e9" if pct > 0 else "#ffebee"
        sector_rows += f"""
        <tr>
          <td style="text-align:left;padding-left:10px;font-weight:600;">#{rank} {sec}</td>
          <td style="background:{bg};color:{clr};font-weight:700;">{pct:+.2f}%</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Stock Breakout Screener</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f5;padding:14px;color:#222;}}
    h2{{font-size:15px;font-weight:700;color:#283593;border-bottom:2px solid #3949ab;
        padding-bottom:5px;margin:22px 0 12px;}}
    .hdr{{background:linear-gradient(135deg,#1a237e,#283593);color:#fff;
          padding:16px 20px;border-radius:10px;margin-bottom:16px;}}
    .hdr h1{{font-size:20px;margin-bottom:4px;}}
    .hdr p{{font-size:12px;color:#b0bec5;}}
    .card{{background:#fff;border-radius:10px;padding:14px 18px;margin-bottom:14px;
           box-shadow:0 1px 5px rgba(0,0,0,0.08);}}
    .bar-wrap{{display:flex;height:24px;border-radius:6px;overflow:hidden;margin:8px 0;}}
    table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;
           overflow:hidden;box-shadow:0 1px 5px rgba(0,0,0,0.08);font-size:12px;}}
    th{{background:#1a237e;color:#fff;padding:9px 8px;text-align:center;font-size:11px;}}
    td{{padding:7px 8px;text-align:center;border-bottom:1px solid #f0f0f0;}}
    tr:last-child td{{border-bottom:none;}}
    img{{max-width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);}}
    .two-col{{display:flex;gap:16px;flex-wrap:wrap;}}
    .two-col>div{{flex:1;min-width:260px;}}
    .note{{font-size:11px;color:#888;margin-top:18px;border-top:1px solid #ddd;padding-top:10px;}}
  </style>
</head>
<body>

<div class="hdr">
  <h1>📈 Stock Breakout Screener</h1>
  <p>Updated: {run_time} UTC &nbsp;·&nbsp;
     Screened {universe_size} stocks &nbsp;·&nbsp;
     {len(results)} scored &nbsp;·&nbsp;
     Top {TOP_N} shown &nbsp;·&nbsp;
     Large-cap ≥ $5B only</p>
</div>

<!-- Sentiment -->
<div class="card">
  <strong style="font-size:14px;">Market Sentiment Today: {sentiment}</strong>
  <div class="bar-wrap" style="margin-top:10px;">
    <div style="width:{green_pct:.0f}%;background:#43a047;"></div>
    <div style="width:{yellow_pct:.0f}%;background:#fdd835;"></div>
    <div style="width:{red_pct:.0f}%;background:#e53935;"></div>
  </div>
  <div style="font-size:12px;color:#555;">
    🟢 Strong {green_pct:.0f}% &nbsp;&nbsp;
    🟡 Moderate {yellow_pct:.0f}% &nbsp;&nbsp;
    🔴 Weak {red_pct:.0f}%
  </div>
</div>

<!-- Top Picks -->
<h2>🏆 Top {TOP_N} Conviction Picks</h2>
{top_cards}

<!-- Detail Table -->
<h2>📊 Full Conviction Dashboard</h2>
<div style="overflow-x:auto;">
<table>
  <tr>
    <th style="text-align:left;padding-left:10px;">Stock</th>
    <th>Volume<br><small>(w:{WEIGHTS['volume']})</small></th>
    <th>Candle<br><small>(w:{WEIGHTS['breakout']})</small></th>
    <th>MA200<br><small>(w:{WEIGHTS['ma200']})</small></th>
    <th>Inst.Own<br><small>(w:{WEIGHTS['inst_own']})</small></th>
    <th>MFI<br><small>(w:{WEIGHTS['mfi']})</small></th>
    <th>Sector<br><small>(w:{WEIGHTS['sector']})</small></th>
    <th>IV<br><small>(w:{WEIGHTS['iv']})</small></th>
    <th>Last BO<br><small>(w:{WEIGHTS['history']})</small></th>
    <th>Score<br><small>/{MAX_SCORE}</small></th>
  </tr>
  {table_rows}
</table>
</div>

<!-- Charts -->
<h2>🎯 Factor Contribution Chart</h2>
<img src="score_chart.png" alt="Factor contribution">

<div class="two-col" style="margin-top:22px;">
  <div>
    <h2>🌀 Sector Rotation</h2>
    <img src="sector_rotation.png" alt="Sector rotation">
  </div>
  <div>
    <h2>📋 Sector Ranking</h2>
    <table>
      <tr><th style="text-align:left;padding-left:10px;">Sector</th><th>Daily Δ</th></tr>
      {sector_rows}
    </table>
  </div>
</div>

<div class="note">
  ⚠️ For informational purposes only — not financial advice. Do your own research.<br>
  Weights: Institutional Ownership (3) · Sector Alignment (3) · Volume Spike (2) ·
  Breakout Candle (2) · MA200 (2) · MFI (2) · History (2) · IV (1)
</div>

</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    print(f"[{run_time}] Starting stock screener...")

    symbols = pd.read_csv(UNIVERSE_FILE)["Symbol"].dropna().tolist()
    print(f"  Universe: {len(symbols)} tickers")

    # Sector rotation first (used for scoring)
    print("  Fetching sector rotation...")
    sector_data = fetch_sector_rotation()
    sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1], reverse=True)
    sector_rank_map = {s: i + 1 for i, (s, _) in enumerate(sorted_sectors)}
    print(f"  Got {len(sector_data)} sectors")

    results = []

    for symbol in symbols:
        try:
            print(f"  {symbol}...", end=" ", flush=True)

            data = fetch_price_data(symbol)
            if data is None or len(data) < 20:
                print("skip:no data"); continue

            # Scalars — squeeze to avoid Series comparison errors
            close   = float(data["Close"].squeeze().iloc[-1])
            open_   = float(data["Open"].squeeze().iloc[-1])
            vol     = float(data["Volume"].squeeze().iloc[-1])
            avg_vol = float(data["Volume"].squeeze().mean())

            vol_ratio = round(vol / avg_vol, 2) if avg_vol > 0 else 0.0
            breakout  = close > open_ * BRK_THRESHOLD

            # MA200 — need enough data
            close_series = data["Close"].squeeze()
            if len(close_series) >= MA_PERIOD:
                ma200 = float(close_series.rolling(MA_PERIOD).mean().iloc[-1])
            else:
                ma200 = float(close_series.mean())
            above_ma = close > ma200

            # Market cap filter
            mcap = fetch_market_cap(symbol)
            if mcap is not None and mcap < MCAP_MIN:
                print(f"skip:mcap<5B"); continue

            # Institutional ownership
            inst_pct = fetch_institutional_ownership(symbol)

            # MFI (local)
            mfi_val = calc_mfi(data, MFI_PERIOD)

            # Sector
            sector   = get_stock_sector(symbol)
            sec_rank = sector_rank_map.get(sector, 99)

            # IV
            iv_val = fetch_implied_volatility(symbol)

            # History
            hist_date, hist_ret = last_breakout_history(data)

            # Score
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
                "ma200":     ma200,
                "mcap":      mcap,
                "inst_pct":  inst_pct,
                "mfi_val":   mfi_val,
                "sector":    sector,
                "iv_val":    iv_val,
                "hist_date": hist_date,
                "hist_ret":  hist_ret,
            })
            print(f"score={score}/{MAX_SCORE}")

        except Exception as e:
            print(f"error: {e}")

    # Sort & pick top N
    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:TOP_N]

    total = max(len(results), 1)
    green_pct  = sum(1 for r in results if r["tl"] == "🟢") / total * 100
    yellow_pct = sum(1 for r in results if r["tl"] == "🟡") / total * 100
    red_pct    = sum(1 for r in results if r["tl"] == "🔴") / total * 100

    print(f"\n  Scored: {len(results)} stocks | Top {TOP_N} shown")

    generate_sector_chart(sector_data)
    generate_score_chart(top_results)

    html = build_html(top_results, sector_data, run_time, len(symbols),
                      green_pct, yellow_pct, red_pct)

    with open(os.path.join(OUTPUT_DIR, "index.html"), "w") as f:
        f.write(html)

    print("  Report written → output/index.html")
    print("Done ✓")

if __name__ == "__main__":
    main()
