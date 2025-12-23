#!/usr/bin/env python3
"""
Daily (EOD) Multi-Agent Trading Workflow (AAPL + GOOGL) using Strands Graph.

- Not realtime: daily closes only
- Simple signals in Technical Analyst agent (SMA20/50, RSI14, 5D momentum)
- Fundamentals + Sentiment kept intentionally simple
- Risk Manager gates trades
- Portfolio Manager allocates capital
- Execution stub prints orders

DISCLAIMER: Educational example only. Not financial advice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
from dotenv import load_dotenv
import requests 

from openai import OpenAI

import numpy as np
import pandas as pd
import yfinance as yf

from strands import Agent, ToolContext, tool
from strands.multiagent import GraphBuilder
from strands.multiagent.graph import GraphState
from strands.multiagent.base import Status

import os

import re
from dataclasses import dataclass
from typing import List, Dict, Literal

import requests
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


load_dotenv()


# -----------------------
# Prompts (force 1-word output)
# -----------------------
SYSTEM_PROMPT = """You are a professional financial sentiment analyst.
Analyze ONLY the sentiment expressed in the provided text samples.
Ignore fundamentals, valuation, macro, and any external knowledge.

Output MUST be EXACTLY one word:
BULLISH
NEUTRAL
BEARISH
No other text.
"""

#Signal = Literal["BUY", "NEUTRAL", "SELL"]
Signal = Literal["BULLISH", "NEUTRAL", "BEARISH"]


os.environ['AWS_PROFILE'] = 'default'
os.environ['AWS_REGION'] = 'eu-central-1'  # Use a region where you have model access
API_KEY_FINANCIAL_DATA  = os.getenv("FMP_API_KEY")
BASE_FINANCIAL_DATA = "https://financialmodelingprep.com/stable" # /api/v3"

share_list = ["AAPL", "GOOGL"]  # default list
Action = Literal["BUY", "SELL", "HOLD"]
Bias = Literal["BULLISH", "BEARISH", "NEUTRAL"] 
openai_api_key = ""



def make_user_prompt(ticker: str, social_texts: List[str]) -> str:
    bullets = "\n".join(f"- {t}" for t in social_texts[:120])  # cap tokens
    return f"""Ticker: {ticker}

Social/forum sentiment samples (most recent first):
{bullets}

Rules:
- BUY: strong bullish tone, optimism, hype, accumulation, confidence
- SELL: strong bearish tone, fear, panic, capitulation, distribution
- NEUTRAL: mixed, unclear, low conviction

Return ONLY one word: BULLISH, NEUTRAL, or BEARISH.
"""

# -----------------------
# Data sources (NO REDDIT)
# -----------------------
TICKER_RE = re.compile(r"[^A-Z0-9\.\-]")

def normalize_ticker(ticker: str) -> str:
    return TICKER_RE.sub("", ticker.strip().upper())

def fetch_stocktwits(ticker: str, limit: int = 60) -> List[str]:
    t = normalize_ticker(ticker)
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{t}.json"
    headers = {
        "User-Agent": "sentiment-research/1.0 (personal use)",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://stocktwits.com/",
        "Origin": "https://stocktwits.com",
    }
    r = requests.get(url, headers=headers, timeout=20)

    # Cloudflare challenge detection (cf-mitigated: challenge)
    if r.headers.get("cf-mitigated", "").lower() == "challenge":
        print("StockTwits blocked by Cloudflare challenge. Skipping StockTwits.")
        return []

    # Some blocks return HTML instead of JSON
    if "text/html" in (r.headers.get("Content-Type", "") or "").lower():
        print("StockTwits returned HTML (likely block page). Skipping StockTwits.")
        return []

    if r.status_code == 403:
        print("StockTwits 403 Forbidden. Skipping StockTwits.")
        return []
    r.raise_for_status()
    data = r.json()
    out: List[str] = []
    for msg in (data.get("messages") or [])[:limit]:
        body = (msg.get("body") or "").strip()
        if body:
            out.append(body)
    return out

def fetch_google_news_rss(ticker: str, limit: int = 40) -> List[str]:
    q = requests.utils.quote(f"{ticker} stock")
    rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    out: List[str] = []
    for e in feed.entries[:limit]:
        title = (getattr(e, "title", "") or "").strip()
        summary = (getattr(e, "summary", "") or "").strip()
        txt = " | ".join([x for x in [title, summary] if x])
        if txt:
            out.append(txt)
    return out

def fetch_finviz_rss(ticker: str, limit: int = 40) -> List[str]:
    t = normalize_ticker(ticker)
    rss_url = f"https://finviz.com/rss.ashx?t={t}"
    feed = feedparser.parse(rss_url)
    out: List[str] = []
    for e in feed.entries[:limit]:
        title = (getattr(e, "title", "") or "").strip()
        summary = (getattr(e, "summary", "") or "").strip()
        txt = " | ".join([x for x in [title, summary] if x])
        if txt:
            out.append(txt)
    return out

# -----------------------
# Hybrid scorer
# -----------------------
@dataclass
class HybridOutput:
    ticker: str
    signal: Signal
    vader_mean: float
    vader_bucket: int
    llm_label: Signal
    llm_bucket: int
    hybrid_score: float
    counts: Dict[str, int]

class HybridSentiment:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        w_vader: float = 0.60,
        w_llm: float = 0.40,
        vader_pos_th: float = 0.12,
        vader_neg_th: float = -0.12,
        final_buy_th: float = 0.35,
        final_sell_th: float = -0.35,
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.w_vader = w_vader
        self.w_llm = w_llm
        self.vader_pos_th = vader_pos_th
        self.vader_neg_th = vader_neg_th
        self.final_buy_th = final_buy_th
        self.final_sell_th = final_sell_th
        self.analyzer = SentimentIntensityAnalyzer()

    def _vader_mean(self, texts: List[str]) -> float:
        if not texts:
            return 0.0
        scores = [self.analyzer.polarity_scores(t)["compound"] for t in texts]
        return sum(scores) / len(scores)

    def _vader_bucket(self, mean: float) -> int:
        if mean >= self.vader_pos_th:
            return 1
        if mean <= self.vader_neg_th:
            return -1
        return 0

    def _llm_label(self, ticker: str, texts: List[str]) -> Signal:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_user_prompt(ticker, texts)},
            ],
        )
        out = resp.choices[0].message.content.strip().upper()
        # Return ONLY one word: BULLISH, NEUTRAL, or BEARISH.
        return out if out in ("BULLISH", "NEUTRAL", "BEARISH") else "NEUTRAL"  # hard-guard

    @staticmethod
    def _label_to_bucket(label: Signal) -> int:
        return {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}[label]

    def _final_label(self, hybrid_score: float) -> Signal:
        if hybrid_score >= self.final_buy_th:
            return "BULLISH"
        if hybrid_score <= self.final_sell_th:
            return "BEARISH"
        return "NEUTRAL"

    def score(self, ticker: str, social_texts: List[str], vader_texts: List[str]) -> HybridOutput:
        # VADER runs on broader text (social + headlines) for stability
        v_mean = self._vader_mean(vader_texts)
        v_bucket = self._vader_bucket(v_mean)

        # LLM runs on social only (better reflects “crowd mood”)
        llm = self._llm_label(ticker, social_texts + vader_texts)
        l_bucket = self._label_to_bucket(llm)

        # Disagreement safety
        opposite = (v_bucket == 1 and l_bucket == -1) or (v_bucket == -1 and l_bucket == 1)
        very_strong_vader = abs(v_mean) >= 0.35

        if opposite and not very_strong_vader:
            hybrid_score = 0.0
            final = "NEUTRAL"
        else:
            hybrid_score = self.w_vader * v_bucket + self.w_llm * l_bucket
            final = self._final_label(hybrid_score)

        return HybridOutput(
            ticker=ticker,
            signal=final,
            vader_mean=v_mean,
            vader_bucket=v_bucket,
            llm_label=llm,
            llm_bucket=l_bucket,
            hybrid_score=hybrid_score,
            counts={"social": len(social_texts), "vader_texts": len(vader_texts)},
        )

# -----------------------
# Runner
# -----------------------
def get_signal(ticker: str) -> HybridOutput:
    t = normalize_ticker(ticker)

    social = []
    try:
        social = fetch_stocktwits(t, limit=60)
    except Exception as e:
        print   (f"Error fetching StockTwits data {e}" )    
        social = []

    headlines = []
    try:
        headlines += fetch_google_news_rss(t, limit=40)
    except Exception:
        pass
    try:
        headlines += fetch_finviz_rss(t, limit=40)
    except Exception:
        pass

    # VADER sees everything; LLM sees only social
    vader_texts = social + headlines

    hs = HybridSentiment()
    return hs.score(t, social_texts=social, vader_texts=vader_texts)


# -----------------------------
# Helper: "wait for all deps"
# (Pattern from Strands Graph docs)
# -----------------------------
def all_dependencies_complete(required_nodes: List[str]):
    def check_all_complete(state: GraphState) -> bool:
        return all(
            node_id in state.results and state.results[node_id].status == Status.COMPLETED
            for node_id in required_nodes
        )
    return check_all_complete



def get_json(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def get_fundamentals(ticker: str) -> dict:

    print ("***** get_fundamentals:******")
    fundamentals = {}
    try:

        income = get_json(f"{BASE_FINANCIAL_DATA}/income-statement?symbol={ticker}&limit=1&apikey={API_KEY_FINANCIAL_DATA}")[0]
        balance = get_json(f"{BASE_FINANCIAL_DATA}/balance-sheet-statement?symbol={ticker}&limit=1&apikey={API_KEY_FINANCIAL_DATA}")[0]
        cashflow = get_json(f"{BASE_FINANCIAL_DATA}/cash-flow-statement?symbol={ticker}&limit=1&apikey={API_KEY_FINANCIAL_DATA}")[0]
        profile = get_json(f"{BASE_FINANCIAL_DATA}/profile?symbol={ticker}&apikey={API_KEY_FINANCIAL_DATA}")[0]

        fundamentals = {
            "ticker": ticker.upper(),
            "fiscal_date": income["date"],
            "currency": income["reportedCurrency"],

            "revenue": float(income["revenue"]),
            "net_income": float(income["netIncome"]),
            "free_cash_flow": float(cashflow["freeCashFlow"]),

            "total_assets": float(balance["totalAssets"]),
            "total_liabilities": float(balance["totalLiabilities"]),

            "market_cap": float(profile["marketCap"])
            # Does not exist 
            # "pe_ratio": float(profile["pe"]),
            # "eps": float(profile["eps"]),
        }
        print (f"***** end get_fundamentals:{fundamentals}******")
    except Exception as e:
        print(f"Error fetching fundamentals for {ticker}: {e}")
    return fundamentals


# -----------------------------
# Indicators (close-only)
# -----------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder-style smoothing via EMA."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


# ============================================================
# TOOLS (each agent uses 1 main tool)
# Tools use ToolContext to access invocation_state (shared state)
# ============================================================

@tool(context=True)
def fetch_daily_closes(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Market Data Tool:
    Fetch daily close prices for configured symbols (defaults: share_list).
    Stores a DataFrame of closes in invocation_state["closes_df"].
    """
    state = tool_context.invocation_state
    symbols: List[str] = state.get("symbols", share_list)
    lookback_days: int = int(state.get("lookback_days", 200))

    # Pull a wider calendar window to ensure we have enough trading days.
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=int(lookback_days * 1.8))

    df = yf.download(
        tickers=symbols,
        start=start.date(),
        end=(end + pd.Timedelta(days=1)).date(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if df is None or len(df) == 0:
        raise RuntimeError("No data returned from data source (yfinance).")

    # yfinance returns MultiIndex columns when multiple tickers requested
    if isinstance(df.columns, pd.MultiIndex):
        closes = df["Close"].copy()
    else:
        closes = df[["Close"]].copy()
        closes.columns = [symbols[0]]

    closes = closes.dropna(how="all").ffill().tail(lookback_days)

    # Store for other agents
    state["closes_df"] = closes

    snapshot = {
        sym: {
            "as_of": str(closes.index[-1].date()),
            "last_close": float(closes[sym].iloc[-1]),
            "n_points": int(closes[sym].dropna().shape[0]),
        }
        for sym in closes.columns
    }

    return {
        "symbols": list(closes.columns),
        "as_of": str(closes.index[-1].date()),
        "daily_close_snapshot": snapshot,
    }


@tool(context=True)
def compute_technical_signals(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Technical Analyst Tool:
    Uses invocation_state["closes_df"] to compute SMA20, SMA50, RSI14, 5D return,
    then emits a simple BUY/SELL/HOLD signal + confidence.
    Stores results in invocation_state["tech_signals"].
    """
    state = tool_context.invocation_state
    closes: pd.DataFrame = state.get("closes_df")
    if closes is None:
        raise RuntimeError("Missing closes_df in invocation_state. Run Market Data first.")

    results: List[Dict[str, Any]] = []
    as_of = str(closes.index[-1].date())

    for sym in closes.columns:
        s = closes[sym].dropna()
        if len(s) < 60:
            results.append(
                {"symbol": sym, "as_of": as_of, "signal": "HOLD", "confidence": 0.0, "reason": "Insufficient history"}
            )
            continue

        sma20 = s.rolling(20).mean()
        sma50 = s.rolling(50).mean()
        rsi14 = rsi(s, 14)
        ret5 = s.pct_change(5)

        c = float(s.iloc[-1])
        s20 = float(sma20.iloc[-1])
        s50 = float(sma50.iloc[-1])
        r14 = float(rsi14.iloc[-1])
        m5 = float(ret5.iloc[-1])

        # Simple rules
        if (c > s20 > s50) and (r14 < 70):
            signal: Action = "BUY"
            reason = "Uptrend (close>SMA20>SMA50) with RSI<70"
        elif (c < s20 < s50) and (r14 > 30):
            signal = "SELL"
            reason = "Downtrend (close<SMA20<SMA50) with RSI>30"
        else:
            signal = "HOLD"
            reason = "No clear trend/momentum alignment"

        # Confidence: trend strength + momentum (very simple scaling)
        trend_strength = (s20 - s50) / max(1e-9, abs(s50))
        conf = float(np.clip(0.5 + 8.0 * trend_strength + 3.0 * m5, 0.0, 1.0))
        if signal == "HOLD":
            conf = float(np.clip(conf - 0.25, 0.0, 1.0))

        results.append(
            {
                "symbol": sym,
                "as_of": as_of,
                "close": c,
                "sma20": s20,
                "sma50": s50,
                "rsi14": r14,
                "ret_5d": m5,
                "signal": signal,
                "confidence": conf,
                "reason": reason,
            }
        )

    state["tech_signals"] = results
    return {"as_of": as_of, "tech_signals": results}


def get_fundamental_bias_from_llm(fundamental_indicators: Any) -> str:
    """
    Call LLM to classify fundamentals into one of: BULLISH, NEUTRAL, BEARISH.
    Robustly handle different response shapes:
      - If LLM returns JSON with a 'classification' field -> use it
      - If LLM returns plain text -> extract the keyword
      - On error -> return "NEUTRAL"
    """
    valid = {"BULLISH", "NEUTRAL", "BEARISH"}
    schema = {
        "name": "llm_fundamental_analysis_result",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "ticker": {"type": "string"},
                "classification": {"type": "string"}       # BEARISH, BULLISH, NEUTRAL
            },
            "required": ["ticker", "classification"]
        }
    }

    try:
        openai = OpenAI(api_key=openai_api_key)
        ticker = (fundamental_indicators.get("ticker") if isinstance(fundamental_indicators, dict) else None) or "MSFT"
        prompt_bias = f"""
        You are a financial analyst.
        Based ONLY on the following fundamentals, classify the stock {ticker} for a likely investment over the next 7 days.
        Provide your answer as one of these exact words: BULLISH, NEUTRAL, BEARISH. The fundamentals are:
        {fundamental_indicators}
        """

        messages = [{"role": "user", "content": prompt_bias}]
        response = openai.chat.completions.create(
            model="gpt-4.1-nano",
            response_format={"type": "json_schema", "json_schema": schema},
            messages=messages
        )

        raw_content = response.choices[0].message.content

        # Normalize different possible response types
        candidate = None
        if isinstance(raw_content, dict):
            # If the SDK already parsed JSON
            candidate = raw_content.get("classification") or raw_content.get("Classification")
        else:
            # raw_content might be a JSON string or plain text
            text = (str(raw_content) or "").strip()
            # Try JSON parse
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    candidate = parsed.get("classification") or parsed.get("Classification")
            except Exception:
                # Fallback: search for the keyword in text
                for k in valid:
                    if k in text.upper():
                        candidate = k
                        break

        if candidate:
            out = candidate.strip().upper()
            return out if out in valid else "NEUTRAL"

    except Exception as e:
        print(f"Error in LLM call for fundamental bias: {e}")

    return "NEUTRAL"  # default on error or unrecognized output


@tool(context=True)
def compute_fundamentals_bias(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Fundamentals Tool (kept intentionally simple):
    Stores results in invocation_state["fund_bias"].
    """
    state = tool_context.invocation_state
    closes: pd.DataFrame = state.get("closes_df")
    if closes is None:
        raise RuntimeError("Missing closes_df in invocation_state. Run Market Data first.")

    as_of = str(closes.index[-1].date())
    bias = []
    for sym in closes.columns:
        fundamental_indicators = get_fundamentals(sym)
        # GET bias from LLM CALL (robust single-word string)
        classification = get_fundamental_bias_from_llm(fundamental_indicators)
        # Ensure safe fallback and normalization
        classification = (classification or "NEUTRAL").strip().upper()
        if classification not in ("BULLISH", "NEUTRAL", "BEARISH"):
            classification = "NEUTRAL"

        bias.append({"symbol": sym, "as_of": as_of, "bias": classification, "note": fundamental_indicators})

    state["fund_bias"] = bias
    return {"as_of": as_of, "fundamentals": bias}


@tool(context=True)
def compute_sentiment_bias(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Sentiment Tool (kept intentionally simple):
    Uses last 5-day return sign as a *proxy* for sentiment.
    Stores results in invocation_state["sent_bias"].

    (In production you'd use news/social + LLM classification, but we're keeping this minimal.)
    """
    state = tool_context.invocation_state
    closes: pd.DataFrame = state.get("closes_df")
    if closes is None:
        raise RuntimeError("Missing closes_df in invocation_state. Run Market Data first.")

    as_of = str(closes.index[-1].date())
    out = []
    for sym in closes.columns:

        sentimentAnalysys_bias = get_signal(sym)
        # s = closes[sym].dropna()
        #m5 = float(s.pct_change(5).iloc[-1]) if len(s) > 6 else 0.0
        #bias: Bias = "BULLISH" if m5 > 0.01 else ("BEARISH" if m5 < -0.01 else "NEUTRAL")
        #bias = sentimentAnalysys_bias.signal
        out.append({"symbol": sym, "as_of": as_of, "bias": sentimentAnalysys_bias.signal}) # , "proxy": "5d_return", "value": m5})
    state["sent_bias"] = out
    return {"as_of": as_of, "sentiment": out}


@tool(context=True)
def risk_gatekeeper(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Risk Manager Tool:
    Combines Tech + Fundamentals + Sentiment into a score and applies risk limits.

    Outputs approved trades with max_weight caps.
    Stores results in invocation_state["approved_trades"].
    """
    state = tool_context.invocation_state

    tech = state.get("tech_signals", [])
    fund = {x["symbol"]: x for x in state.get("fund_bias", [])}
    sent = {x["symbol"]: x for x in state.get("sent_bias", [])}
    closes: pd.DataFrame = state.get("closes_df")

    if closes is None or not tech:
        raise RuntimeError("Missing prerequisites for Risk Manager (closes_df / tech_signals).")

    # Simple risk config
    capital = float(state.get("capital", 100_000))
    max_position_pct = float(state.get("max_position_pct", 0.10))      # 10% max per name
    target_risk_pct = float(state.get("target_risk_pct", 0.01))        # 1% risk budget per name
    min_score = float(state.get("min_score", 0.55))                    # gate threshold
    long_only = bool(state.get("long_only", True))

    approved = []
    as_of = str(closes.index[-1].date())

    # 20D vol from close-to-close returns
    rets = closes.pct_change().dropna(how="all")
    vol20 = rets.rolling(20).std().iloc[-1].to_dict()

    def bias_to_num(b: Bias) -> float:
        return 1.0 if b == "BULLISH" else (-1.0 if b == "BEARISH" else 0.0)

    for t in tech:
        sym = t["symbol"]
        tech_sig = t["signal"]
        tech_conf = float(t["confidence"])
        tech_num = 1.0 if tech_sig == "BUY" else (-1.0 if tech_sig == "SELL" else 0.0)

        fnum = bias_to_num(fund.get(sym, {}).get("bias", "NEUTRAL"))
        snum = bias_to_num(sent.get(sym, {}).get("bias", "NEUTRAL"))

        # Weighted combined score (simple)
        score = 0.6 * (tech_num * tech_conf) + 0.2 * fnum + 0.2 * snum

        # Risk sizing cap based on volatility (very simple)
        v = float(vol20.get(sym, np.nan))
        if not np.isfinite(v) or v <= 0:
            v = 0.02  # fallback

        # weight so that (weight * vol) ~= target_risk_pct, capped
        max_weight_by_risk = float(np.clip(target_risk_pct / v, 0.0, max_position_pct))

        # Gatekeeping rules
        if long_only and tech_sig != "BUY":
            continue
        if score < min_score:
            continue

        approved.append(
            {
                "symbol": sym,
                "as_of": as_of,
                "action": "BUY",
                "score": float(score),
                "vol20": float(v),
                "max_weight": max_weight_by_risk,
                "note": "Approved by risk gate",
            }
        )

    state["approved_trades"] = approved
    return {"as_of": as_of, "approved_trades": approved}


@tool(context=True)
def build_portfolio_plan(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Portfolio Manager Tool:
    Prioritize approved trades, allocate capital, and output an order list.

    Stores results in invocation_state["trade_plan"].
    """
    state = tool_context.invocation_state
    approved: List[Dict[str, Any]] = state.get("approved_trades", [])
    closes: pd.DataFrame = state.get("closes_df")
    capital = float(state.get("capital", 100_000))

    if closes is None:
        raise RuntimeError("Missing closes_df in invocation_state.")

    as_of = str(closes.index[-1].date())

    if not approved:
        plan = {"as_of": as_of, "cash": capital, "orders": [], "note": "No trades passed risk gate"}
        state["trade_plan"] = plan
        return plan

    # Prioritize by score (desc)
    approved_sorted = sorted(approved, key=lambda x: x["score"], reverse=True)

    # Allocate weights proportional to (score * max_weight), then normalize to <= 1.0
    raw = np.array([max(0.0, t["score"]) * max(0.0, t["max_weight"]) for t in approved_sorted], dtype=float)
    if raw.sum() <= 0:
        weights = np.zeros_like(raw)
    else:
        weights = raw / raw.sum()

    # Apply each trade's max_weight cap, then re-normalize (simple one-pass)
    capped = np.array([min(weights[i], approved_sorted[i]["max_weight"]) for i in range(len(weights))], dtype=float)
    total = capped.sum()
    if total > 1.0:
        capped = capped / total

    orders = []
    used = 0.0
    for i, trade in enumerate(approved_sorted):
        sym = trade["symbol"]
        w = float(capped[i])
        price = float(closes[sym].dropna().iloc[-1])
        dollars = capital * w
        qty = int(dollars // price)  # integer shares

        if qty <= 0:
            continue

        used += qty * price
        orders.append(
            {
                "symbol": sym,
                "action": "BUY",
                "qty": qty,
                "limit_price": None,  # market in this demo
                "est_price": price,
                "weight": w,
                "score": trade["score"],
            }
        )

    plan = {
        "as_of": as_of,
        "capital": capital,
        "estimated_used": float(used),
        "estimated_cash_left": float(capital - used),
        "orders": orders,
        "prioritization": [t["symbol"] for t in approved_sorted],
    }

    state["trade_plan"] = plan
    return plan


@tool(context=True)
def execute_trades_stub(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Execution Tool (stub):
    Prints orders. In production, replace with broker API calls.
    """
    state = tool_context.invocation_state
    plan = state.get("trade_plan", {})
    orders = plan.get("orders", [])

    # Stub side effect:
    print("\n=== EXECUTION (STUB) ===")
    for o in orders:
        print(f"PLACE ORDER: {o['action']} {o['qty']} {o['symbol']} (est_price={o['est_price']})")
    print("========================\n")

    return {"executed": True, "n_orders": len(orders), "note": "Stub only - integrate broker API here"}


# ============================================================
# AGENTS (each agent calls its tool and returns the tool output)
# ============================================================

def make_agents(model_id: Optional[str] = None) -> Dict[str, Agent]:
    """
    model_id is optional. If omitted, Strands uses Bedrock defaults.
    You can also pass a Bedrock model id string directly (e.g., "us.amazon.nova-premier-v1:0").  # see docs
    """
    common_kwargs = {}
    if model_id:
        common_kwargs["model"] = model_id

    market_data_agent = Agent(
        name="market_data",
        tools=[fetch_daily_closes],
        system_prompt="You are the Market Data Agent. Call the tool and return ONLY its JSON.",
        **common_kwargs,
    )

    technical_agent = Agent(
        name="technical_analyst",
        tools=[compute_technical_signals],
        system_prompt="You are the Technical Analyst Agent. Call the tool and return ONLY its JSON.",
        **common_kwargs,
    )

    fundamentals_agent = Agent(
        name="fundamentals",
        tools=[compute_fundamentals_bias],
        system_prompt="You are the Fundamentals Agent. Call the tool and return ONLY its JSON.",
        **common_kwargs,
    )

    sentiment_agent = Agent(
        name="sentiment",
        tools=[compute_sentiment_bias],
        system_prompt="You are the Sentiment Agent. Call the tool and return ONLY its JSON.",
        **common_kwargs,
    )

    risk_agent = Agent(
        name="risk_manager",
        tools=[risk_gatekeeper],
        system_prompt="You are the Risk Manager (gatekeeper). Call the tool and return ONLY its JSON.",
        **common_kwargs,
    )

    portfolio_agent = Agent(
        name="portfolio_manager",
        tools=[build_portfolio_plan],
        system_prompt="You are the Portfolio Manager. Call the tool and return ONLY its JSON.",
        **common_kwargs,
    )

    execution_agent = Agent(
        name="execution",
        tools=[execute_trades_stub],
        system_prompt="You are the Execution Agent. Call the tool and return ONLY its JSON.",
        **common_kwargs,
    )

    return {
        "market_data": market_data_agent,
        "technical": technical_agent,
        "fundamentals": fundamentals_agent,
        "sentiment": sentiment_agent,
        "risk": risk_agent,
        "portfolio": portfolio_agent,
        "execution": execution_agent,
    }


# ============================================================
# GRAPH (workflow orchestration)
# ============================================================

def build_graph(agents: Dict[str, Agent]):
    builder = GraphBuilder()

    builder.add_node(agents["market_data"], "market_data")
    builder.add_node(agents["technical"], "technical")
    builder.add_node(agents["fundamentals"], "fundamentals")
    builder.add_node(agents["sentiment"], "sentiment")
    builder.add_node(agents["risk"], "risk")
    builder.add_node(agents["portfolio"], "portfolio")
    builder.add_node(agents["execution"], "execution")

    # Market data fans out
    builder.add_edge("market_data", "technical")
    builder.add_edge("market_data", "fundamentals")
    builder.add_edge("market_data", "sentiment")

    # Risk waits for ALL (tech + fund + sent) using conditional edges
    wait_all = all_dependencies_complete(["technical", "fundamentals", "sentiment"])
    builder.add_edge("technical", "risk", condition=wait_all)
    builder.add_edge("fundamentals", "risk", condition=wait_all)
    builder.add_edge("sentiment", "risk", condition=wait_all)

    builder.add_edge("risk", "portfolio")
    builder.add_edge("portfolio", "execution")

    builder.set_entry_point("market_data")
    return builder.build()


def main():
    # Shared state (passed behind the scenes to all agents/tools)
    shared_state = {
        "symbols": share_list,     # "GOOG" also valid, but we pick GOOGL explicitly
        "lookback_days": 200,
        "capital": 100_000,
        "max_position_pct": 0.10,        # 10% cap per stock
        "target_risk_pct": 0.01,         # 1% risk budget
        "min_score": 0.55,               # gate threshold
        "long_only": True,
    }

    # Optional: specify a Bedrock model ID string (otherwise defaults apply).  :contentReference[oaicite:4]{index=4}
    agents = make_agents(model_id=None)
    graph = build_graph(agents)

    result = graph(
        f"Run the daily (EOD) trading workflow for {share_list} using close prices only.",
        invocation_state=shared_state,
    )

    print(f"\nGraph Status: {result.status}")

    # The final plan is in shared_state (written by tools)
    plan = shared_state.get("trade_plan", {})
    print("\n=== FINAL TRADE PLAN (JSON) ===")
    print(json.dumps(plan, indent=2, default=str))


if __name__ == "__main__":
    # GET imput  parameters from user as a list of stock share      
    # and pass them to main function
    # For now, we keep it static                               
    shares = input("Enter a list of stock shares separated by commas (e.g., AAPL,GOOGL,MSFT): ")
    if not shares:
        shares = "AAPL,GOOGL"  # default
    share_list = [share.strip() for share in shares.split(",") if share.strip()]
    openai_api_key = input("Enter a OpenAI API Key: ")

    if openai_api_key:
        print(f"La clave API de OpenAI existe y empieza por {openai_api_key[:8]}")
    else:
        print("La clave API de OpenAI no existe - Dirígete a la guía de solución de problemas en la carpeta de configuración.")
    
    main()

    # out = get_signal("AAPL")
    # print(out)
    # print("FINAL SIGNAL:", out.signal)
