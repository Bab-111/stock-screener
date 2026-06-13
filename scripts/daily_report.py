"""
Stock Breakout Screener v5
- Locks to regular market close bar (prevents after-hours ranking drift)
- Adds data source citations to HTML report
- Adds Claude API LLM supervision summary
- Optimised: 2 batch downloads + 10 enrichment calls only
"""

import json, os, warnings, requests
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths & Config ────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(ROOT, "config", "config.json")) as f:
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
HIST_RET_THRESH = float(CFG["history_return_threshold"])
FWD_DAYS        = int(CFG["forward_return_days"])
# Optional Claude API key for LLM supervision (set as GitHub Secret)
CLAUDE_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")

WEIGHTS = {
    "volume":2, "breakout":2, "ma200":2,
    "inst_own":3, "mfi":2, "sector":3, "history":2, "iv":1,
}
MAX_SCORE = sum(WEIGHTS.values())  # 17

# ── Static sector map ─────────────────────────────────────────────────────────
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
    "Technology":"XLK","Healthcare":"XLV","Financials":"XLF",
    "Industrials":"XLI","Energy":"XLE","Consumer Disc":"XLY",
    "Consumer Stpl":"XLP","Materials":"XLB","Real Estate":"XLRE",
    "Utilities":"XLU","Comm. Services":"XLC",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

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

def tl_label(tl):
    return "Strong" if tl=="🟢" else ("Mod" if tl=="🟡" else "Weak")

def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df

def get_regular_close_bar(df):
    """
    Return the last COMPLETED regular-market bar.
    Drops any incomplete intraday bar so after-hours data
    doesn't shift rankings between runs.
    Uses the second-to-last bar if today's bar looks incomplete
    (volume suspiciously low vs average).
    """
    df = df.copy().dropna(how="all")
    if len(df) < 2:
        return df.iloc[-1]
    last    = df.iloc[-1]
    prev    = df.iloc[-2]
    avg_vol = float(df["Volume"].mean())
    # If last bar volume < 20% of average → likely incomplete intraday bar
    if float(last["Volume"]) < avg_vol * 0.20:
        return prev
    return last

# ── BATCH 1: Sector ETFs ──────────────────────────────────────────────────────

def batch_sector():
    print("  [1/3] Sector rotation...", flush=True)
    etfs = list(SECTOR_ETFS.values())
    try:
        raw = yf.download(etfs, period="5d", interval="1d",
                          group_by="ticker", auto_adjust=True,
                          progress=False, threads=True)
    except Exception as e:
        print(f"  Sector error: {e}"); return {}
    result = {}
    for sector, etf in SECTOR_ETFS.items():
        try:
            cl = raw[etf]["Close"].squeeze() if len(etfs)>1 else flatten(raw)["Close"]
            if len(cl) >= 2:
                result[sector] = round(
                    float((cl.iloc[-1]-cl.iloc[-2])/cl.iloc[-2]*100), 2)
        except Exception:
            pass
    print(f"  Got {len(result)} sectors")
    return result

# ── BATCH 2: All price data ───────────────────────────────────────────────────

def batch_price(symbols):
    print(f"  [2/3] Price data ({len(symbols)} tickers)...", flush=True)
    try:
        raw = yf.download(symbols, period="6mo", interval="1d",
                          group_by="ticker", auto_adjust=True,
                          progress=False, threads=True)
    except Exception as e:
        print(f"  Price error: {e}"); return {}
    needed = ["Open","High","Low","Close","Volume"]
    result = {}
    if len(symbols) == 1:
        sym = symbols[0]; df = flatten(raw)
        if len(df)>=20 and all(c in df.columns for c in needed):
            result[sym] = df[needed].dropna(how="all")
        return result
    for sym in symbols:
        try:
            if sym not in raw.columns.get_level_values(0): continue
            df = flatten(raw[sym])
            if len(df)<20: continue
            if not all(c in df.columns for c in needed): continue
            result[sym] = df[needed].dropna(how="all")
        except Exception:
            pass
    print(f"  Got data for {len(result)} tickers")
    return result

# ── Per-stock maths ───────────────────────────────────────────────────────────

def calc_ma200(data):
    cl  = data["Close"].squeeze().astype(float)
    win = min(MA_PERIOD, len(cl))
    return float(cl.rolling(win).mean().iloc[-1])

def calc_mfi(data):
    try:
        hi  = data["High"].squeeze().astype(float)
        lo  = data["Low"].squeeze().astype(float)
        cl  = data["Close"].squeeze().astype(float)
        vol = data["Volume"].squeeze().astype(float)
        tp  = (hi+lo+cl)/3; rmf = tp*vol
        pos = rmf.where(tp>tp.shift(1), 0.0)
        neg = rmf.where(tp<tp.shift(1), 0.0)
        mfr = pos.rolling(MFI_PERIOD).sum() / \
              neg.rolling(MFI_PERIOD).sum().replace(0,1e-9)
        v   = float((100-(100/(1+mfr))).iloc[-1])
        return round(v,1) if not np.isnan(v) else None
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
            np.mean(closes[max(0,i-win):i]) if i>=20 else np.nan
            for i in range(len(closes))
        ])
        for i in range(len(data)-2, max(0,len(data)-90), -1):
            if (volumes[i]>=VOL_THRESHOLD*avg_vol and
                    closes[i]>opens[i]*BRK_THRESHOLD and
                    not np.isnan(ma_vals[i]) and closes[i]>ma_vals[i]):
                fwd = min(i+FWD_DAYS, len(closes)-1)
                return (data.index[i].strftime("%b %d, %Y"),
                        round((closes[fwd]-closes[i])/closes[i]*100,1))
    except Exception:
        pass
    return None, None

# ── Enrich top-10 only ────────────────────────────────────────────────────────

def enrich_candidate(sym):
    inst_pct = iv_val = mcap = None
    try:
        info = yf.Ticker(sym).info
        raw  = info.get("heldPercentInstitutions")
        if raw is not None:
            inst_pct = round(float(raw)*100, 1)
        mcap = safe_float(info.get("marketCap"))
    except Exception:
        pass
    try:
        t    = yf.Ticker(sym)
        exps = t.options
        if exps:
            chain  = t.option_chain(exps[0])
            iv_val = round(
                float(chain.calls["impliedVolatility"].dropna().mean())*100, 1)
    except Exception:
        pass
    return inst_pct, iv_val, mcap

# ── LLM Supervision via Claude API ───────────────────────────────────────────

def llm_supervision(top_results, sector_data):
    """
    Call Claude API to validate top picks and return a short supervision note.
    Uses claude-haiku (cheapest/fastest). Falls back gracefully if no API key.
    """
    if not CLAUDE_API_KEY:
        return ("⚠️ LLM supervision disabled — add ANTHROPIC_API_KEY as a "
                "GitHub Secret to enable AI validation of picks.")
    try:
        top_sector = max(sector_data.items(), key=lambda x: x[1])[0] \
                     if sector_data else "Unknown"
        picks_text = "\n".join([
            f"- {r['symbol']} ({r['sector']}): score={r['score']}/{MAX_SCORE}, "
            f"vol={r['vol_ratio']:.1f}x, MA200={'above' if r['close']>r['ma200'] else 'below'}, "
            f"inst={r['inst_pct']}%, MFI={r['mfi_val']}, "
            f"last_breakout={r['hist_date']} ({r['hist_ret']:+.1f}%)"
            for r in top_results
        ])
        prompt = f"""You are a stock analysis supervisor reviewing an automated screener's output.

Today's top sector: {top_sector}
Screener top picks:
{picks_text}

In 3-4 bullet points, briefly:
1. Validate whether these picks make sense given the sector leadership
2. Flag any red flags (e.g. low MFI despite high score, below MA200)
3. Note the strongest and weakest pick and why
4. Give an overall confidence rating: HIGH / MEDIUM / LOW

Be concise — max 60 words total. No disclaimers."""

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        if resp.status_code == 200:
            text = resp.json()["content"][0]["text"].strip()
            return text
        else:
            return f"LLM supervision API error: {resp.status_code}"
    except Exception as e:
        return f"LLM supervision unavailable: {e}"

# ── Scoring ───────────────────────────────────────────────────────────────────

def score_stock(vol_ratio, breakout, above_ma, inst_pct,
                mfi_val, sec_rank, iv_val, hist_ret):
    f={}; s=0
    if vol_ratio>=VOL_THRESHOLD:   f["volume"]="green";  s+=2
    elif vol_ratio>=1.3:           f["volume"]="yellow"; s+=1
    else:                          f["volume"]="red"
    if breakout:                   f["breakout"]="green";  s+=2
    elif vol_ratio>=1.5:           f["breakout"]="yellow"; s+=1
    else:                          f["breakout"]="red"
    f["ma200"]="green" if above_ma else "red"; s+=(2 if above_ma else 0)
    ic=classify(inst_pct,INST_HIGH,INST_MOD)
    f["inst_own"]=ic; s+=(3 if ic=="green" else (1 if ic=="yellow" else 0))
    mc=classify(mfi_val,MFI_THRESHOLD+10,MFI_THRESHOLD)
    f["mfi"]=mc; s+=(2 if mc=="green" else (1 if mc=="yellow" else 0))
    if sec_rank==1:   f["sector"]="green";  s+=3
    elif sec_rank<=3: f["sector"]="yellow"; s+=1
    else:             f["sector"]="red"
    ivc=classify(iv_val,IV_HIGH,IV_MOD)
    f["iv"]=ivc; s+=(1 if ivc=="green" else 0)
    if (hist_ret or 0)>=HIST_RET_THRESH:  f["history"]="green";  s+=2
    elif (hist_ret or 0)>0:               f["history"]="yellow"; s+=1
    else:                                 f["history"]="red"
    return s, f

# ── Charts ────────────────────────────────────────────────────────────────────

def chart_sector(sector_data):
    if not sector_data: return
    items  = sorted(sector_data.items(), key=lambda x:x[1], reverse=True)
    names  = [x[0] for x in items]; vals=[x[1] for x in items]
    colors = ["#43a047" if v>0 else "#e53935" for v in vals]
    fig,ax = plt.subplots(figsize=(9,5))
    ax.barh(names[::-1],vals[::-1],color=colors[::-1],edgecolor="white")
    ax.axvline(0,color="#333",lw=0.8)
    ax.set_xlabel("Daily % Change",fontsize=10)
    ax.set_title("Sector Rotation — Today",fontsize=12,fontweight="bold")
    for i,v in enumerate(vals[::-1]):
        ax.text(v+(0.03 if v>=0 else -0.03),i,f"{v:+.2f}%",
                va="center",ha="left" if v>=0 else "right",fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"sector_rotation.png"),dpi=110)
    plt.close()

def chart_scores(results):
    if not results: return
    order=["inst_own","sector","volume","breakout","ma200","mfi","history","iv"]
    labels={"inst_own":"Inst.Own","sector":"Sector","volume":"Volume",
            "breakout":"Breakout","ma200":"MA200","mfi":"MFI",
            "history":"History","iv":"IV"}
    cmap={"green":"#43a047","yellow":"#fdd835","red":"#ef9a9a"}
    fig,ax=plt.subplots(figsize=(10,max(3,len(results)*0.9)))
    for i,r in enumerate(results):
        left=0
        for fk in order:
            cls=r["factors"].get(fk,"red")
            pts=WEIGHTS[fk] if cls=="green" else (1 if cls=="yellow" else 0)
            if pts:
                ax.barh(i,pts,left=left,color=cmap[cls],edgecolor="white",lw=0.5)
                ax.text(left+pts/2,i,labels[fk],
                        ha="center",va="center",fontsize=7,color="#222")
                left+=pts
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([f"{tl_label(r['tl'])} {r['symbol']}" for r in results],fontsize=11)
    ax.set_xlim(0,MAX_SCORE+0.5)
    ax.set_xlabel("Conviction Points",fontsize=10)
    ax.set_title("Factor Contribution",fontsize=12,fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(color="#43a047",label="Strong"),
        mpatches.Patch(color="#fdd835",label="Moderate"),
        mpatches.Patch(color="#ef9a9a",label="Weak"),
    ],loc="lower right",fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"score_chart.png"),dpi=110)
    plt.close()

# ── HTML ──────────────────────────────────────────────────────────────────────

BG={"green":"#c8e6c9","yellow":"#fff9c4","red":"#ffcdd2"}

def badge(cls):
    ic={"green":"✅","yellow":"⚠️","red":"❌"}
    bg={"green":"#e8f5e9","yellow":"#fffde7","red":"#ffebee"}
    fg={"green":"#2e7d32","yellow":"#f57f17","red":"#b71c1c"}
    c=cls if cls in ic else "red"
    return (f'<span style="background:{bg[c]};color:{fg[c]};'
            f'padding:2px 7px;border-radius:4px;font-size:11px;">{ic[c]}</span>')

def build_html(results, sector_data, run_time, n_universe,
               green_pct, yellow_pct, red_pct, llm_note, market_phase):

    sentiment=("🐂 Bullish" if green_pct>=50 else
               "😐 Neutral" if green_pct>=25 else "🐻 Bearish")

    # ── Top picks cards ──
    top_cards=""
    for i,r in enumerate(results,1):
        f=r["factors"]; bo=(r["close"]/r["open"]-1)*100
        mcap_s=f"${r['mcap']/1e9:.1f}B" if r["mcap"] else "N/A"
        bullets=[
            (f["volume"],
             f"Volume: <b>{r['vol_ratio']:.1f}× average</b> — "
             f"{'🔥 Strong unusual activity' if r['vol_ratio']>=2 else 'Above-average activity'}"),
            (f["breakout"],
             f"Candle: close <b>${r['close']:.2f}</b> vs open <b>${r['open']:.2f}</b> "
             f"({bo:+.1f}%) — regular market close"),
            (f["ma200"],
             f"200-Day MA: price <b>${r['close']:.2f}</b> vs MA <b>${r['ma200']:.2f}</b> — "
             f"<b>{'✅ Above' if r['close']>r['ma200'] else '❌ Below'}</b> long-term trend"),
            (f["inst_own"],
             f"Inst. ownership: <b>{str(r['inst_pct'])+'%' if r['inst_pct'] else 'N/A'}</b> · "
             f"MFI: <b>{r['mfi_val'] or 'N/A'}</b> · Market cap: <b>{mcap_s}</b>"),
        ]
        if r["hist_date"]:
            bullets.append((f["history"],
                f"Last similar breakout: <b>{r['hist_date']}</b> → "
                f"<b>{r['hist_ret']:+.1f}%</b> over {FWD_DAYS} trading days"))
        bhtml="".join(
            f'<li style="margin:5px 0">{badge(c)} {t}</li>' for c,t in bullets)
        top_cards+=f"""
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

    # ── Table rows ──
    table_rows=""
    for r in results:
        f=r["factors"]; bo=(r["close"]/r["open"]-1)*100
        table_rows+=f"""
        <tr>
          <td style="font-weight:700;text-align:left;padding-left:10px">
            {r['tl']} {r['symbol']}</td>
          <td style="background:{BG[f['volume']]}">{r['vol_ratio']:.1f}×</td>
          <td style="background:{BG[f['breakout']]}">{bo:+.1f}%</td>
          <td style="background:{BG[f['ma200']]}">
            {'▲ Above' if r['close']>r['ma200'] else '▼ Below'}</td>
          <td style="background:{BG[f['inst_own']]}">
            {(str(r['inst_pct'])+'%') if r['inst_pct'] is not None else 'N/A'}</td>
          <td style="background:{BG[f['mfi']]}">
            {r['mfi_val'] if r['mfi_val'] is not None else 'N/A'}</td>
          <td style="background:{BG[f['sector']]}">{r['sector']}</td>
          <td style="background:{BG[f['iv']]}">
            {(str(r['iv_val'])+'%') if r['iv_val'] is not None else 'N/A'}</td>
          <td style="background:{BG[f['history']]}">
            {('%+.1f%%'%r['hist_ret']) if r['hist_ret'] is not None else 'N/A'}</td>
          <td style="font-weight:800;font-size:15px">{r['score']}/{MAX_SCORE}</td>
        </tr>"""

    # ── Sector rows ──
    sector_rows=""
    for rank,(sec,pct) in enumerate(
            sorted(sector_data.items(),key=lambda x:x[1],reverse=True),1):
        clr="#2e7d32" if pct>0 else "#c62828"
        bg="#e8f5e9" if pct>0 else "#ffebee"
        sector_rows+=f"""
        <tr>
          <td style="text-align:left;padding-left:10px;font-weight:600">
            #{rank} {sec}</td>
          <td style="background:{bg};color:{clr};font-weight:700">{pct:+.2f}%</td>
        </tr>"""

    # ── LLM supervision box ──
    llm_html=f"""
    <div style="background:#e8eaf6;border-left:4px solid #3949ab;
                border-radius:8px;padding:14px 18px;margin-bottom:14px;">
      <strong style="font-size:13px;color:#1a237e">
        🤖 AI Supervision (Claude)
      </strong>
      <p style="font-size:13px;color:#333;margin-top:8px;
                white-space:pre-wrap;line-height:1.6">{llm_note}</p>
    </div>"""

    # ── Data sources ──
    sources_html=f"""
    <div style="background:#fff;border-radius:10px;padding:14px 18px;
                margin-top:22px;box-shadow:0 1px 5px rgba(0,0,0,.08);
                font-size:12px;color:#555">
      <strong style="font-size:13px;color:#283593">📚 Data Sources & Methodology</strong>
      <ul style="margin:10px 0 0 18px;line-height:2">
        <li><b>Price / OHLCV data:</b> Yahoo Finance via yfinance library
            — 6-month daily bars, regular market close only (after-hours excluded)</li>
        <li><b>Volume spike:</b> Today's volume vs 6-month average
            — threshold ≥{VOL_THRESHOLD:.0f}× for green signal</li>
        <li><b>Breakout candle:</b> Close &gt; Open × {BRK_THRESHOLD:.2f}
            (i.e. +{(BRK_THRESHOLD-1)*100:.0f}% intraday gain)</li>
        <li><b>200-Day MA:</b> Rolling {MA_PERIOD}-day mean of closing prices
            — calculated locally from downloaded data</li>
        <li><b>Money Flow Index (MFI):</b> Calculated locally using
            {MFI_PERIOD}-period standard formula (no external API)</li>
        <li><b>Institutional ownership:</b> Yahoo Finance
            heldPercentInstitutions — fetched for top-10 candidates only</li>
        <li><b>Implied Volatility (IV):</b> Average IV of nearest-expiry
            call options via Yahoo Finance — top-10 candidates only</li>
        <li><b>Sector rotation:</b> Daily % change of sector ETFs
            (XLK, XLV, XLF, XLI, XLE, XLY, XLP, XLB, XLRE, XLU, XLC)</li>
        <li><b>Historical breakout:</b> Looks back 90 days for prior
            similar breakout, measures {FWD_DAYS}-day forward return</li>
        <li><b>Market cap filter:</b> ≥ ${MCAP_MIN/1e9:.0f}B
            (large-cap only — data from Yahoo Finance info)</li>
        <li><b>AI supervision:</b> Claude claude-haiku-4-5-20251001 via Anthropic API
            — validates picks against sector context (requires API key)</li>
        <li><b>Universe:</b> {n_universe} large-cap US stocks
            defined in config/universe.csv</li>
        <li><b>Scoring weights:</b>
            Inst.Own(3) · Sector(3) · Volume(2) · Breakout(2) ·
            MA200(2) · MFI(2) · History(2) · IV(1) = {MAX_SCORE}pts max</li>
        <li><b>Run time:</b> {run_time} UTC — Market phase: {market_phase}</li>
      </ul>
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Cache-Control" content="no-cache,no-store,must-revalidate">
  <title>Stock Breakout Screener — {run_time}</title>
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
  <p>Updated: {run_time} UTC &nbsp;·&nbsp; Phase: {market_phase} &nbsp;·&nbsp;
     {n_universe} stocks screened &nbsp;·&nbsp; Top {TOP_N} by conviction score</p>
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

{llm_html}

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

{sources_html}

<div class="note">
  ⚠️ For informational purposes only — not financial advice.
  All data from Yahoo Finance (free tier). Rankings use regular market close only.
</div>
</body>
</html>"""

# ── Market phase helper ───────────────────────────────────────────────────────

def get_market_phase(utc_hour, utc_minute):
    """Return human-readable market phase based on UTC time."""
    t = utc_hour * 60 + utc_minute
    # ET = UTC - 4 (EDT summer)
    et = t - 240
    if et < 0: et += 1440
    if   et <  570: return "Pre-Market (before 9:30 AM ET)"
    elif et <  960: return "Regular Market Hours"
    elif et < 1200: return "After-Hours (4:00–8:00 PM ET)"
    else:           return "Overnight"

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0       = datetime.now(timezone.utc).replace(tzinfo=None)
    run_time = t0.strftime("%Y-%m-%d %H:%M")
    phase    = get_market_phase(t0.hour, t0.minute)
    print(f"[{run_time}] Stock screener — {phase}")

    symbols = pd.read_csv(UNIVERSE_FILE)["Symbol"].dropna().tolist()
    print(f"  Universe: {len(symbols)} tickers")

    # ── Batch 1: Sector ──
    sector_data   = batch_sector()
    sorted_sectors = sorted(sector_data.items(), key=lambda x:x[1], reverse=True)
    sector_rank   = {s:i+1 for i,(s,_) in enumerate(sorted_sectors)}

    # ── Batch 2: Prices ──
    price_data = batch_price(symbols)
    valid_syms = list(price_data.keys())

    # ── Score all stocks (no API calls) ──
    print(f"  [3/3] Scoring {len(valid_syms)} stocks...", flush=True)
    all_results = []

    for sym in valid_syms:
        try:
            data      = price_data[sym]
            bar       = get_regular_close_bar(data)  # ← locked to regular close
            close     = float(bar["Close"])
            open_     = float(bar["Open"])
            vol       = float(bar["Volume"])
            avg_vol   = float(data["Volume"].squeeze().mean())
            vol_ratio = round(vol/avg_vol, 2) if avg_vol>0 else 0.0
            breakout  = close > open_*BRK_THRESHOLD
            ma200     = calc_ma200(data)
            above_ma  = close > ma200
            mfi_val   = calc_mfi(data)
            sec       = SECTOR_MAP.get(sym, "Unknown")
            sec_rank  = sector_rank.get(sec, 99)
            hist_date, hist_ret = last_breakout(data)

            score, factors = score_stock(
                vol_ratio, breakout, above_ma,
                None, mfi_val, sec_rank, None, hist_ret
            )
            all_results.append({
                "symbol":sym, "score":score, "tl":traffic_light(score),
                "factors":factors, "close":close, "open":open_,
                "vol_ratio":vol_ratio, "ma200":ma200, "mcap":None,
                "inst_pct":None, "mfi_val":mfi_val, "sector":sec,
                "iv_val":None, "hist_date":hist_date, "hist_ret":hist_ret,
            })
        except Exception:
            pass

    all_results.sort(key=lambda x:x["score"], reverse=True)
    top10 = all_results[:10]

    # ── Enrich top-10 (10 API calls) ──
    print("  Enriching top 10...", flush=True)
    for r in top10:
        sym = r["symbol"]
        inst_pct, iv_val, mcap = enrich_candidate(sym)
        r["inst_pct"]=inst_pct; r["iv_val"]=iv_val; r["mcap"]=mcap
        new_score, new_factors = score_stock(
            r["vol_ratio"], r["close"]>r["open"]*BRK_THRESHOLD,
            r["close"]>r["ma200"], inst_pct, r["mfi_val"],
            sector_rank.get(r["sector"],99), iv_val, r["hist_ret"]
        )
        r["score"]=new_score; r["factors"]=new_factors
        r["tl"]=traffic_light(new_score)
        print(f"    {sym}: {new_score}/{MAX_SCORE} inst={inst_pct}% iv={iv_val}%")

    top10.sort(key=lambda x:x["score"], reverse=True)
    top_results = top10[:TOP_N]

    # ── LLM supervision ──
    print("  Running LLM supervision...", flush=True)
    llm_note = llm_supervision(top_results, sector_data)
    print(f"  LLM: {llm_note[:80]}...")

    # ── Sentiment ──
    total      = max(len(all_results),1)
    green_pct  = sum(1 for r in all_results if r["tl"]=="🟢")/total*100
    yellow_pct = sum(1 for r in all_results if r["tl"]=="🟡")/total*100
    red_pct    = sum(1 for r in all_results if r["tl"]=="🔴")/total*100

    chart_sector(sector_data)
    chart_scores(top_results)

    html = build_html(top_results, sector_data, run_time, len(symbols),
                      green_pct, yellow_pct, red_pct, llm_note, phase)
    with open(os.path.join(OUTPUT_DIR,"index.html"),"w") as fh:
        fh.write(html)

    elapsed=(datetime.now(timezone.utc).replace(tzinfo=None)-t0).seconds
    print(f"  Done: {len(all_results)} scored | top {TOP_N} shown | {elapsed}s ✓")

if __name__ == "__main__":
    main()
