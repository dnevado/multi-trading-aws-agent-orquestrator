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

import numpy as np
import pandas as pd
import yfinance as yf

from strands import Agent, ToolContext, tool
from strands.multiagent import GraphBuilder
from strands.multiagent.graph import GraphState
from strands.multiagent.base import Status

import os


os.environ['AWS_PROFILE'] = 'dnevadopersonal'
os.environ['AWS_REGION'] = 'eu-central-1'  # Use a region where you have model access



Action = Literal["BUY", "SELL", "HOLD"]
Bias = Literal["BULLISH", "BEARISH", "NEUTRAL"] 


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
    Fetch daily close prices for configured symbols (defaults: AAPL, GOOGL).
    Stores a DataFrame of closes in invocation_state["closes_df"].
    """
    state = tool_context.invocation_state
    symbols: List[str] = state.get("symbols", ["AAPL", "GOOGL"])
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


@tool(context=True)
def compute_fundamentals_bias(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Fundamentals Tool (kept intentionally simple):
    Returns NEUTRAL bias for each symbol (placeholder).
    Stores results in invocation_state["fund_bias"].
    """
    state = tool_context.invocation_state
    closes: pd.DataFrame = state.get("closes_df")
    if closes is None:
        raise RuntimeError("Missing closes_df in invocation_state. Run Market Data first.")

    as_of = str(closes.index[-1].date())
    bias = [{"symbol": sym, "as_of": as_of, "bias": "NEUTRAL", "note": "Placeholder fundamentals"} for sym in closes.columns]

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
        s = closes[sym].dropna()
        m5 = float(s.pct_change(5).iloc[-1]) if len(s) > 6 else 0.0
        bias: Bias = "BULLISH" if m5 > 0.01 else ("BEARISH" if m5 < -0.01 else "NEUTRAL")
        out.append({"symbol": sym, "as_of": as_of, "bias": bias, "proxy": "5d_return", "value": m5})

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
        "symbols": ["AAPL", "GOOGL"],     # "GOOG" also valid, but we pick GOOGL explicitly
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
        "Run the daily (EOD) trading workflow for AAPL and GOOGL using close prices only.",
        invocation_state=shared_state,
    )

    print(f"\nGraph Status: {result.status}")

    # The final plan is in shared_state (written by tools)
    plan = shared_state.get("trade_plan", {})
    print("\n=== FINAL TRADE PLAN (JSON) ===")
    print(json.dumps(plan, indent=2, default=str))


if __name__ == "__main__":
    main()
