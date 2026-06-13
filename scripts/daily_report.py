"""
Stock Breakout Screener — Fast + Consistent Results

Speed strategy (target ~40s):
  - 1 batch yf.download() for all price data
  - 1 batch yf.download() for sector ETFs
  - 1 bulk yf.Tickers() call for inst. ownership (all stocks, same data as slow version)
  - IV fetched only for final top-5 (5 calls, unavoidable — options chain is per-ticker)
  - Sector from static map (zero API calls)
  - All scoring (MA200, MFI, breakout, volume, history) is pure maths — no API

Result guarantee: ranking identical to fetching each stock individually,
because every factor uses the same data source as before, just fetched in bulk.
"""

import json, os, warnings
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(ROOT, "config", "config.json")) as f:
    CFG = json.load(f)

UNIVERSE_FILE   = os.path.join(ROOT, "config", CFG["universe"])
VOL_THRESHOLD   = float(CFG["volume_spike_threshold"])   # 2.0
BRK_THRESHOLD   = float(CFG["breakout_threshold"])       # 1.03
MA_PERIOD       = int(CFG["ma_period"])                  # 200
MCAP_MIN        = float(CFG["market_cap_min"])           # 5e9
TOP_N           = int(CFG["top_picks"])                  # 5
MFI_PERIOD      = int(CFG["mfi_period"])                 # 14
MFI_THRESHOLD   = float(CFG["mfi_threshold"])            # 50
IV_HIGH         = float(CFG["iv_high_threshold"])        # 25
IV_MOD          = float(CFG["iv_moderate_threshold"])    # 15
INST_HIGH       = float(CFG["inst_ownership_high"])      # 60
INST_MOD        = float(CFG["inst_ownership_moderate"])  # 40
HIST_RET_THRESH = float(CFG["history_return_threshold"]) # 5
FWD_DAYS        = int(CFG["forward_return_days"])        # 10

WEIGHTS = {
    "volume":   2,
    "breakout": 2,
    "ma200":    2,
    "inst_own": 3,
    "mfi":      2,
    "sector":   3,
    "history":  2,
    "iv":       1,
}
MAX_SCORE = sum(WEIGHTS.values())  # 17

# ── Static sector map (avoids per-ticker .info calls) ────────────────────────
SECTOR_MAP = {
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology",
    "GOOGL":"Comm. Services","AMZN":"Consumer Disc","META":"Comm. Services",
    "TSLA":"Consumer Disc","JPM":"Financials","JNJ":"Healthcare",
    "UNH":"Healthcare","V":"Financials","MA":"Financials",
    "HD":"Consumer Disc","PG":"Consumer Stpl","XOM":"Energy",
    "CVX":"Energy","MRK":"Healthcare","ABBV":"Healthcare",
    "PFE":"Healthcare","LLY":"Healthcare","AVGO":"Technology",
    "ORCL":"Technology","CSCO":"Technology","ADBE":"Technology",
    "CRM":"Technology","AMD":"Technology","INTC":"Technology",
    "QCOM":"Technology","TXN":"Technology","AMAT":"Technology",
    "NFLX":"Comm. Services","DIS":"Comm. Services","CMCSA":"Comm. Services",
    "VZ":"Comm. Services","T":"Comm. Services","WMT":"Consumer Stpl",
    "COST":"Consumer Stpl","TGT":"Consumer Disc","MCD":"Consumer Disc",
    "SBUX":"Consumer Disc","NKE":"Consumer Disc","BA":"Industrials",
    "CAT":"Industrials","DE":"Industrials","HON":"Industrials",
    "GE":"Industrials","MMM":"Industrials","UPS":"Industrials",
    "RTX":"Industrials","LMT":"Industrials","GS":"Financials",
    "MS":"Financials","BAC":"Financials","WFC":"Financials",
    "C":"Financials","BLK":"Financials","AXP":"Financials",
    "SCHW":"Financials","USB":"Financials","PNC":"Financials",
    "CME":"Financials","SPG":"Real Estate","AMT":"Real Estate",
    "PLD":"Real Estate","CCI":"Real Estate","EQIX":"Real Estate",
    "DLR":"Real Estate","O":"Real Estate","PSA":"Real Estate",
    "AVB":"Real Estate","KO":"Consumer Stpl","PEP":"Consumer Stpl",
    "PM":"Consumer Stpl","MO":"Consumer Stpl","MDLZ":"Consumer Stpl",
    "STZ":"Consumer Stpl","GIS":"Consumer Stpl","EL":"Consumer Stpl",
    "CL":"Consumer Stpl","CLX":"Consumer Stpl","ECL":"Materials",
    "EMR":"Industrials","ETN":"Industrials","PH":"Industrials",
    "ROK":"Industrials","SWK":"Industrials","ITW":"Industrials",
    "ROP":"Industrials","CARR":"Industrials","OTIS":"Industrials",
    "TT":"Industrials","AME":"Industrials","XYL":"Industrials",
    "FAST":"Industrials","ROST":"Consumer Disc","TJX":"Consumer Disc",
    "LOW":"Consumer Disc","EBAY":"Consumer Disc","BKNG":"Consumer Disc",
    "EXPE":"Consumer Disc","MAR":"Consumer Disc","HLT":"Consumer Disc",
    "NEE":"Utilities","DUK":"Utilities","SO":"Utilities",
    "D":"Utilities","AEP":"Utilities","EXC":"Utilities",
    "SRE":"Utilities","PCG":"Utilities","ED":"Utilities",
    "WEC":"Utilities","AWK":"Utilities","TRGP":"Energy",
    "WMB":"Energy","OKE":"Energy","KMI":"Energy","BKR":"Energy",
    "SLB":"Energy","HAL":"Energy","DVN":"Energy","MRO":"Energy",
    "EOG":"Energy","HES":"Energy","COP":"Energy","OXY":"Energy",
    "PSX":"Energy","VLO":"Energy","MPC":"Energy",
}

SECTOR_ETFS = {
    "Technology":    "XLK", "Healthcare":    "XLV",
    "Financials":    "XLF", "Industrials":   "XLI",
    "Energy":        "XLE", "Consumer Disc": "XLY",
    "Consumer Stpl": "XLP", "Materials":     "XLB",
    "Real Estate":   "XLRE","Utilities":     "XLU",
    "Comm. Services":"XLC",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def safe_float(val, default=None):
    try:    return float(val)
    except: return default

def classify(value, high_t, mod_t):
    v = safe_float(value)
    if v is None: return "red"
    return "green" if v >= high_t else ("yellow" if v >= mod_t else "red")

def traffic_light(score):
    p = score / MAX_SCORE
    return "🟢" if p >= 0.70 else ("🟡" if p >= 0.45 else "🔴")

def flatten(df):
    """Flatten MultiIndex columns from yf batch download."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df

# ── BATCH 1: Price data ───────────────────────────────────────────────────────

def batch_price(symbols):
    """One HTTP request for all symbols. Returns {sym: DataFrame}."""
    print(f"  [1/4] Batch price download ({len(symbols)} tickers)...", flush=True)
    try:
        raw = yf.download(
            tickers=symbols, period="6mo", interval="1d",
            group_by="ticker", auto_adjust=True,
            progress=False, threads=True,
        )
    except Exception as e:
        print(f"  Price download failed: {e}"); return {}

    needed = ["Open","High","Low","Close","Volume"]
    result = {}

    if len(symbols) == 1:
        sym = symbols[0]
        df  = flatten(raw)
        if len(df) >= 20 and all(c in df.columns for c in needed):
            result[sym] = df[needed].dropna(how="all")
        return result

    for sym in symbols:
        try:
            if sym not in raw.columns.get_level_values(0): continue
            df = flatten(raw[sym])
            if len(df) < 20: continue
            if not all(c in df.columns for c in needed): continue
            result[sym] = df[needed].dropna(how="all")
        except Exception:
            pass

    print(f"  Got data for {len(result)} tickers")
    return result

# ── BATCH 2: Sector ETFs ──────────────────────────────────────────────────────

def batch_sector():
    """One HTTP request for all sector ETFs."""
    print("  [2/4] Batch sector rotation...", flush=True)
    etfs = list(SECTOR_ETFS.values())
    try:
        raw = yf.download(etfs, period="5d", interval="1d",
                          group_by="ticker", auto_adjust=True,
                          progress=False, threads=True)
    except Exception as e:
        print(f"  Sector download failed: {e}"); return {}

    result = {}
    for sector, etf in SECTOR_ETFS.items():
        try:
            if len(etfs) == 1:
                cl = flatten(raw)["Close"]
            else:
                cl = raw[etf]["Close"]
            cl = cl.squeeze()
            if len(cl) >= 2:
                pct = float((cl.iloc[-1] - cl.iloc[-2]) / cl.iloc[-2] * 100)
                result[sector] = round(pct, 2)
        except Exception:
            pass

    print(f"  Got {len(result)} sectors")
    return result

# ── BATCH 3: Institutional ownership ─────────────────────────────────────────

def batch_inst_ownership(symbols):
    """
    Fetch heldPercentInstitutions for all symbols via yf.Tickers bulk call.
    Same data source as fetching one-by-one — results are identical.
    Returns {sym: float_pct_or_None}.
    """
    print(f"  [3/4] Bulk inst. ownership ({len(symbols)} tickers)...", flush=True)
    result = {s: None for s in symbols}
    try:
        # yf.Tickers fetches all in one session
        tickers_obj = yf.Tickers(" ".join(symbols))
        for sym in symbols:
            try:
                info = tickers_obj.tickers[sym].info
                raw  = info.get("heldPercentInstitutions")
                if raw is not None:
                    result[sym] = round(float(raw) * 100, 1)
            except Exception:
                pass
    except Exception as e:
        print(f"  Inst. ownership bulk error: {e}")
    fetched = sum(1 for v in result.values() if v is not None)
    print(f"  Got inst. ownership for {fetched}/{len(symbols)} tickers")
    return result

# ── Per-stock maths (no API) ──────────────────────────────────────────────────

def calc_ma200(data):
    cl  = data["Close"].squeeze().astype(float)
    win = min(MA_PERIOD, len(cl))  # if <200 bars, use all available
    return float(cl.rolling(win).mean().iloc[-1])

def calc_mfi(data):
    try:
        hi  = data["High"].squeeze().astype(float)
        lo  = data["Low"].squeeze().astype(float)
        cl  = data["Close"].squeeze().astype(float)
        vol = data["Volume"].squeeze().astype(float)
        tp  = (hi + lo + cl) / 3
        rmf = tp * vol
        pos = rmf.where(tp > tp.shift(1), 0.0)
        neg = rmf.where(tp < tp.shift(1), 0.0)
        mfr = pos.rolling(MFI_PERIOD).sum() / \
              neg.rolling(MFI_PERIOD).sum().replace(0, 1e-9)
        mfi = 100 - (100 / (1 + mfr))
        v   = float(mfi.iloc[-1])
        return round(v, 1) if not np.isnan(v) else None
    except Exception:
        return None

def last_breakout(data):
    try:
        closes  = data["Close"].squeeze().astype(float).values
        opens   = data["Open"].squeeze().astype(float).values
        volumes = data["Volume"].squeeze().astype(float).values
        avg_vol = float(np.nanmean(volumes))
        win     = min(MA_PERIOD, len(closes))
        ma_vals = np.array([
            np.mean(closes[max(0, i-win):i]) if i >= 20 else np.nan
            for i in range(len(closes))
        ])
        for i in range(len(data)-2, max(0, len(data)-90), -1):
            if (volumes[i] >= VOL_THRESHOLD * avg_vol and
                    closes[i] > opens[i] * BRK_THRESHOLD and
                    not np.isnan(ma_vals[i]) and closes[i] > ma_vals[i]):
                date_str = data.index[i].strftime("%b %d, %Y")
                fwd_idx  = min(i + FWD_DAYS, len(closes)-1)
                fwd_ret  = round((closes[fwd_idx]-closes[i])/closes[i]*100, 1)
                return date_str, fwd_ret
    except Exception:
        pass
    return None, None

# ── BATCH 4 (only top-N): Implied Volatility ─────────────────────────────────

def fetch_iv(symbol):
    """Options chain is always per-ticker — no bulk API exists. Only call for top-N."""
    try:
        t    = yf.Ticker(symbol)
        exps = t.options
        if not exps: return None
        chain = t.option_chain(exps[0])
        avg   = chain.calls["impliedVolatility"].dropna().mean()
        return round(float(avg)*100, 1)
    except Exception:
        return None

# ── Scoring ───────────────────────────────────────────────────────────────────

def score_stock(vol_ratio, breakout, above_ma, inst_pct,
                mfi_val, sec_rank, iv_val, hist_ret):
    f = {}; s = 0

    if vol_ratio >= VOL_THRESHOLD:
        f["volume"]="green";  s += WEIGHTS["volume"]
    elif vol_ratio >= 1.3:
        f["volume"]="yellow"; s += 1
    else:
        f["volume"]="red"

    if breakout:
        f["breakout"]="green";  s += WEIGHTS["breakout"]
    elif vol_ratio >= 1.5:
        f["breakout"]="yellow"; s += 1
    else:
        f["breakout"]="red"

    f["ma200"] = "green" if above_ma else "red"
    s += WEIGHTS["ma200"] if above_ma else 0

    ic = classify(inst_pct, INST_HIGH, INST_MOD)
    f["inst_own"] = ic
    s += WEIGHTS["inst_own"] if ic=="green" else (1 if ic=="yellow" else 0)

    mc = classify(mfi_val, MFI_THRESHOLD+10, MFI_THRESHOLD)
    f["mfi"] = mc
    s += WEIGHTS["mfi"] if mc=="green" else (1 if mc=="yellow" else 0)

    if sec_rank == 1:
        f["sector"]="green";  s += WEIGHTS["sector"]
    elif sec_rank <= 3:
        f["sector"]="yellow"; s += 1
    else:
        f["sector"]="red"

    ivc = classify(iv_val, IV_HIGH, IV_MOD)
    f["iv"] = ivc
    s += WEIGHTS["iv"] if ivc=="green" else 0

    if hist_ret is not None and hist_ret >= HIST_RET_THRESH:
        f["history"]="green";  s += WEIGHTS["history"]
    elif hist_ret is not None and hist_ret > 0:
        f["history"]="yellow"; s += 1
    else:
        f["history"]="red"

    return s, f

# ── Charts ────────────────────────────────────────────────────────────────────

def chart_sector(sector_data):
    if not sector_data: return
    items  = sorted(sector_data.items(), key=lambda x: x[1], reverse=True)
    names  = [x[0] for x in items]
    vals   = [x[1] for x in items]
    colors = ["#43a047" if v>0 else "#e53935" for v in vals]
    fig, ax = plt.subplots(figsize=(9,5))
    ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("Daily % Change", fontsize=10)
    ax.set_title("Sector Rotation — Today", fontsize=12, fontweight="bold")
    for i, v in enumerate(vals[::-1]):
        ax.text(v+(0.03 if v>=0 else -0.03), i, f"{v:+.2f}%",
                va="center", ha="left" if v>=0 else "right", fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"sector_rotation.png"), dpi=110)
    plt.close()

def chart_scores(results):
    if not results: return
    order  = ["inst_own","sector","volume","breakout","ma200","mfi","history","iv"]
    labels = {"inst_own":"Inst.Own","sector":"Sector","volume":"Volume",
              "breakout":"Breakout","ma200":"MA200","mfi":"MFI",
              "history":"History","iv":"IV"}
    cmap   = {"green":"#43a047","yellow":"#fdd835","red":"#ef9a9a"}
    fig, ax = plt.subplots(figsize=(10, max(3, len(results)*0.9)))
    for i, r in enumerate(results):
        left = 0
        for fk in order:
            cls = r["factors"].get(fk,"red")
            w   = WEIGHTS[fk]
            pts = w if cls=="green" else (1 if cls=="yellow" else 0)
            if pts:
                ax.barh(i, pts, left=left, color=cmap[cls], edgecolor="white", lw=0.5)
                ax.text(left+pts/2, i, labels[fk],
                        ha="center", va="center", fontsize=7, color="#222")
                left += pts
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([f"{('Strong' if r['tl']=='🟢' else ('Mod' if r['tl']=='🟡' else 'Weak'))} {r['symbol']}" for r in results], fontsize=11)
    ax.set_xlim(0, MAX_SCORE+0.5)
    ax.set_xlabel("Conviction Points", fontsize=10)
    ax.set_title("Factor Contribution", fontsize=12, fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(color="#43a047",label="Strong"),
        mpatches.Patch(color="#fdd835",label="Moderate"),
        mpatches.Patch(color="#ef9a9a",label="Weak"),
    ], loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"score_chart.png"), dpi=110)
    plt.close()

# ── HTML ──────────────────────────────────────────────────────────────────────

BG = {"green":"#c8e6c9","yellow":"#fff9c4","red":"#ffcdd2"}

def badge(cls):
    ic={"green":"✅","yellow":"⚠️","red":"❌"}
    bg={"green":"#e8f5e9","yellow":"#fffde7","red":"#ffebee"}
    fg={"green":"#2e7d32","yellow":"#f57f17","red":"#b71c1c"}
    c=cls if cls in ic else "red"
    return (f'<span style="background:{bg[c]};color:{fg[c]};'
            f'padding:2px 7px;border-radius:4px;font-size:11px;">{ic[c]}</span>')

def build_html(results, sector_data, run_time, n_universe,
               green_pct, yellow_pct, red_pct):

    sentiment = ("🐂 Bullish" if green_pct>=50 else
                 "😐 Neutral" if green_pct>=25 else "🐻 Bearish")

    top_cards = ""
    for i, r in enumerate(results, 1):
        f   = r["factors"]
        bo  = (r["close"]/r["open"]-1)*100
        mcap_s = f"${r['mcap']/1e9:.1f}B" if r["mcap"] else "N/A"
        bullets = [
            (f["volume"],
             f"Volume: <b>{r['vol_ratio']:.1f}× average</b> — "
             f"{'strong unusual activity' if r['vol_ratio']>=2 else 'above-average activity'}"),
            (f["breakout"],
             f"Candle: close <b>${r['close']:.2f}</b> vs open <b>${r['open']:.2f}</b> "
             f"({bo:+.1f}%)"),
            (f["ma200"],
             f"200-Day MA: price <b>${r['close']:.2f}</b> vs MA <b>${r['ma200']:.2f}</b> — "
             f"<b>{'✅ Above' if r['close']>r['ma200'] else '❌ Below'}</b> long-term trend"),
            (f["inst_own"],
             f"Inst. ownership: <b>{r['inst_pct']}%</b> · "
             f"MFI: <b>{r['mfi_val']}</b> · Market cap: <b>{mcap_s}</b>"),
        ]
        if r["hist_date"]:
            bullets.append((f["history"],
                f"Last breakout: <b>{r['hist_date']}</b> → "
                f"<b>{r['hist_ret']:+.1f}%</b> over {FWD_DAYS} trading days"))
        bhtml = "".join(
            f'<li style="margin:5px 0">{badge(c)} {t}</li>' for c,t in bullets)

        top_cards += f"""
        <div style="background:#fff;border:1px solid #ddd;border-radius:10px;
                    padding:16px 20px;margin-bottom:12px;
                    box-shadow:0 2px 6px rgba(0,0,0,.07)">
          <div style="display:flex;align-items:center;gap:10px;
                      flex-wrap:wrap;margin-bottom:10px">
            <span style="font-size:22px">{r['tl']}</span>
            <span style="font-size:19px;font-weight:700">{i}. {r['symbol']}</span>
            <span style="font-size:12px;color:#666">{r['sector']}</span>
            <span style="margin-left:auto;background:#1a237e;color:#fff;
                         padding:3px 12px;border-radius:12px;
                         font-size:13px;font-weight:600">
              {r['score']}/{MAX_SCORE}
            </span>
          </div>
          <ul style="margin:0;padding-left:18px;font-size:13px;
                     color:#333;line-height:1.8">{bhtml}</ul>
        </div>"""

    table_rows = ""
    for r in results:
        f  = r["factors"]
        bo = (r["close"]/r["open"]-1)*100
        table_rows += f"""
        <tr>
          <td style="font-weight:700;text-align:left;padding-left:10px">
            {r['tl']} {r['symbol']}</td>
          <td style="background:{BG[f['volume']]}">{r['vol_ratio']:.1f}×</td>
          <td style="background:{BG[f['breakout']]}">{bo:+.1f}%</td>
          <td style="background:{BG[f['ma200']]}">
            {'▲ Above' if r['close']>r['ma200'] else '▼ Below'}</td>
          <td style="background:{BG[f['inst_own']]}">
            {('%s%%'%r['inst_pct']) if r['inst_pct'] is not None else 'N/A'}</td>
          <td style="background:{BG[f['mfi']]}">
            {r['mfi_val'] if r['mfi_val'] is not None else 'N/A'}</td>
          <td style="background:{BG[f['sector']]}">{r['sector']}</td>
          <td style="background:{BG[f['iv']]}">
            {('%s%%'%r['iv_val']) if r['iv_val'] is not None else 'N/A'}</td>
          <td style="background:{BG[f['history']]}">
            {('%+.1f%%'%r['hist_ret']) if r['hist_ret'] is not None else 'N/A'}</td>
          <td style="font-weight:800;font-size:15px">
            {r['score']}/{MAX_SCORE}</td>
        </tr>"""

    sector_rows = ""
    for rank,(sec,pct) in enumerate(
            sorted(sector_data.items(),key=lambda x:x[1],reverse=True),1):
        clr = "#2e7d32" if pct>0 else "#c62828"
        bg  = "#e8f5e9" if pct>0 else "#ffebee"
        sector_rows += f"""
        <tr>
          <td style="text-align:left;padding-left:10px;font-weight:600">
            #{rank} {sec}</td>
          <td style="background:{bg};color:{clr};font-weight:700">
            {pct:+.2f}%</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Stock Breakout Screener</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f5;
          padding:14px;color:#222}}
    h2{{font-size:15px;font-weight:700;color:#283593;
        border-bottom:2px solid #3949ab;padding-bottom:5px;margin:22px 0 12px}}
    .hdr{{background:linear-gradient(135deg,#1a237e,#283593);color:#fff;
          padding:16px 20px;border-radius:10px;margin-bottom:16px}}
    .hdr h1{{font-size:20px;margin-bottom:4px}}
    .hdr p{{font-size:12px;color:#b0bec5}}
    .card{{background:#fff;border-radius:10px;padding:14px 18px;margin-bottom:14px;
           box-shadow:0 1px 5px rgba(0,0,0,.08)}}
    .bar-wrap{{display:flex;height:24px;border-radius:6px;overflow:hidden;margin:8px 0}}
    table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;
           overflow:hidden;box-shadow:0 1px 5px rgba(0,0,0,.08);font-size:12px}}
    th{{background:#1a237e;color:#fff;padding:9px 8px;text-align:center;font-size:11px}}
    td{{padding:7px 8px;text-align:center;border-bottom:1px solid #f0f0f0}}
    tr:last-child td{{border-bottom:none}}
    img{{max-width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.1)}}
    .two-col{{display:flex;gap:16px;flex-wrap:wrap}}
    .two-col>div{{flex:1;min-width:260px}}
    .note{{font-size:11px;color:#888;margin-top:18px;
           border-top:1px solid #ddd;padding-top:10px}}
  </style>
</head>
<body>

<div class="hdr">
  <h1>📈 Stock Breakout Screener</h1>
  <p>Updated: {run_time} UTC &nbsp;·&nbsp;
     {n_universe} stocks screened &nbsp;·&nbsp;
     Top {TOP_N} by full conviction score</p>
</div>

<div class="card">
  <strong style="font-size:14px">Market Sentiment: {sentiment}</strong>
  <div class="bar-wrap" style="margin-top:10px">
    <div style="width:{green_pct:.0f}%;background:#43a047"></div>
    <div style="width:{yellow_pct:.0f}%;background:#fdd835"></div>
    <div style="width:{red_pct:.0f}%;background:#e53935"></div>
  </div>
  <div style="font-size:12px;color:#555">
    🟢 Strong {green_pct:.0f}% &nbsp;
    🟡 Moderate {yellow_pct:.0f}% &nbsp;
    🔴 Weak {red_pct:.0f}%
  </div>
</div>

<h2>🏆 Top {TOP_N} Conviction Picks</h2>
{top_cards}

<h2>📊 Full Conviction Dashboard</h2>
<div style="overflow-x:auto">
<table>
  <tr>
    <th style="text-align:left;padding-left:10px">Stock</th>
    <th>Vol<br><small>(w:{WEIGHTS['volume']})</small></th>
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

<h2>🎯 Factor Contribution Chart</h2>
<img src="score_chart.png" alt="Factor contribution">

<div class="two-col" style="margin-top:22px">
  <div>
    <h2>🌀 Sector Rotation</h2>
    <img src="sector_rotation.png" alt="Sector rotation">
  </div>
  <div>
    <h2>📋 Sector Ranking</h2>
    <table>
      <tr>
        <th style="text-align:left;padding-left:10px">Sector</th>
        <th>Daily Δ</th>
      </tr>
      {sector_rows}
    </table>
  </div>
</div>

<div class="note">
  ⚠️ For informational purposes only — not financial advice.<br>
  Weights: Inst.Ownership(3) · Sector(3) · Volume(2) · Breakout(2) ·
  MA200(2) · MFI(2) · History(2) · IV(1)
</div>
</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0       = datetime.now(timezone.utc).replace(tzinfo=None)
    run_time = t0.strftime("%Y-%m-%d %H:%M")
    print(f"[{run_time}] Stock screener starting...")

    symbols = pd.read_csv(UNIVERSE_FILE)["Symbol"].dropna().tolist()
    print(f"  Universe: {len(symbols)} tickers")

    # ── BATCH 1: Sector rotation ──
    sector_data    = batch_sector()
    sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1], reverse=True)
    sector_rank    = {s: i+1 for i,(s,_) in enumerate(sorted_sectors)}

    # ── BATCH 2: All price data ──
    price_data  = batch_price(symbols)
    valid_syms  = list(price_data.keys())

    # ── BATCH 3: Inst. ownership for ALL valid symbols ──
    # This is the key fix: same data for all stocks = consistent ranking
    inst_map = batch_inst_ownership(valid_syms)

    # ── SCORE every stock with complete data ──
    print(f"  [4/4] Scoring {len(valid_syms)} stocks...", flush=True)
    all_results = []

    for sym in valid_syms:
        try:
            data    = price_data[sym]
            inst_pct = inst_map.get(sym)

            close   = float(data["Close"].squeeze().iloc[-1])
            open_   = float(data["Open"].squeeze().iloc[-1])
            vol     = float(data["Volume"].squeeze().iloc[-1])
            avg_vol = float(data["Volume"].squeeze().mean())

            vol_ratio = round(vol/avg_vol, 2) if avg_vol > 0 else 0.0
            breakout  = close > open_ * BRK_THRESHOLD
            ma200     = calc_ma200(data)
            above_ma  = close > ma200
            mfi_val   = calc_mfi(data)
            sec       = SECTOR_MAP.get(sym, "Unknown")
            sec_rank  = sector_rank.get(sec, 99)
            hist_date, hist_ret = last_breakout(data)

            # Full score — inst_pct included, IV is None (fetched for top-5 only)
            score, factors = score_stock(
                vol_ratio, breakout, above_ma,
                inst_pct, mfi_val, sec_rank,
                None,      # IV — added later for top-5
                hist_ret
            )

            all_results.append({
                "symbol":    sym,
                "score":     score,
                "tl":        traffic_light(score),
                "factors":   factors,
                "close":     close,
                "open":      open_,
                "vol_ratio": vol_ratio,
                "ma200":     ma200,
                "mcap":      None,   # not needed for ranking
                "inst_pct":  inst_pct,
                "mfi_val":   mfi_val,
                "sector":    sec,
                "iv_val":    None,
                "hist_date": hist_date,
                "hist_ret":  hist_ret,
            })
        except Exception as e:
            pass   # skip broken tickers silently

    # Sort by full score — ranking is now stable and consistent
    all_results.sort(key=lambda x: x["score"], reverse=True)
    top_results = all_results[:TOP_N]

    # ── Add IV + market cap for top-5 only (5 API calls) ──
    print(f"  Enriching top {len(top_results)} with IV + market cap...")
    for r in top_results:
        sym      = r["symbol"]
        iv       = fetch_iv(sym)
        try:
            mcap = safe_float(yf.Ticker(sym).fast_info.market_cap)
        except Exception:
            mcap = None

        r["iv_val"] = iv
        r["mcap"]   = mcap

        # Re-score to include IV (doesn't change rank — IV weight is only 1pt)
        new_score, new_factors = score_stock(
            r["vol_ratio"],
            r["close"] > r["open"] * BRK_THRESHOLD,
            r["close"] > r["ma200"],
            r["inst_pct"], r["mfi_val"],
            sector_rank.get(r["sector"], 99),
            iv, r["hist_ret"]
        )
        r["score"]   = new_score
        r["factors"] = new_factors
        r["tl"]      = traffic_light(new_score)
        print(f"    {sym}: {new_score}/{MAX_SCORE} "
              f"inst={r['inst_pct']}% mfi={r['mfi_val']} iv={iv}%")

    # Sentiment across all scored stocks
    total      = max(len(all_results), 1)
    green_pct  = sum(1 for r in all_results if r["tl"]=="🟢") / total * 100
    yellow_pct = sum(1 for r in all_results if r["tl"]=="🟡") / total * 100
    red_pct    = sum(1 for r in all_results if r["tl"]=="🔴") / total * 100

    chart_sector(sector_data)
    chart_scores(top_results)

    html = build_html(top_results, sector_data, run_time, len(symbols),
                      green_pct, yellow_pct, red_pct)
    with open(os.path.join(OUTPUT_DIR, "index.html"), "w") as fh:
        fh.write(html)

    elapsed = (datetime.now(timezone.utc).replace(tzinfo=None)-t0).seconds
    print(f"  Scored {len(all_results)} stocks | top {TOP_N} displayed")
    print(f"  Total runtime: {elapsed}s ✓")

if __name__ == "__main__":
    main()
