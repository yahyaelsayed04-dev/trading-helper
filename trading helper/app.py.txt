# app.py
import os, datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

st.set_page_config(page_title="AI Trading Helper", layout="wide")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker (e.g., AAPL, MSFT, EURUSD=X)", value="AAPL")
    start = st.date_input("Start date", dt.date(2022, 1, 1))
    end = st.date_input("End date", dt.date.today())
    short_win = st.number_input("Short SMA", min_value=2, value=10)
    long_win = st.number_input("Long SMA", min_value=3, value=30)
    sl = st.number_input("Stop-loss (%)", value=5.0)      # illustrative; not used in this simple backtest
    tp = st.number_input("Take-profit (%)", value=10.0)   # illustrative; not used in this simple backtest

st.title("AI Trading Helper")

# 1) Load data
@st.cache_data(show_spinner=False)
def load_prices(symbol, start, end):
    data = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    return data.dropna()

data = load_prices(ticker, start, end)
if data.empty:
    st.warning("No data returned. Try another symbol or date range.")
    st.stop()

st.subheader(f"Price data: {ticker}")
st.line_chart(data["Close"])

# 2) Simple SMA crossover signals (vectorized)
df = data.copy()
df["SMA_S"] = df["Close"].rolling(int(short_win)).mean()
df["SMA_L"] = df["Close"].rolling(int(long_win)).mean()
df["Signal"] = (df["SMA_S"] > df["SMA_L"]).astype(int)
df["Position"] = df["Signal"].shift(1).fillna(0)  # avoid look-ahead
df["Ret"] = df["Close"].pct_change().fillna(0)
df["StratRet"] = df["Position"] * df["Ret"]
df["Equity"] = (1 + df["StratRet"]).cumprod()

# Metrics
def max_drawdown(equity):
    roll_max = equity.cummax()
    dd = equity/roll_max - 1.0
    return dd.min()

total_ret = df["Equity"].iloc[-1] - 1
cagr = (df["Equity"].iloc[-1]) ** (252/len(df)) - 1 if len(df) > 252 else np.nan
vol = df["StratRet"].std() * np.sqrt(252)
sharpe = (df["StratRet"].mean() * 252) / vol if vol > 0 else np.nan
mdd = max_drawdown(df["Equity"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total return", f"{total_ret:.2%}")
col2.metric("CAGR", f"{cagr:.2%}" if pd.notna(cagr) else "—")
col3.metric("Sharpe (naïve)", f"{sharpe:.2f}" if pd.notna(sharpe) else "—")
col4.metric("Max DD", f"{mdd:.2%}")

st.subheader("Strategy equity curve")
st.line_chart(df["Equity"])

st.subheader("Signals (1=long, 0=flat)")
st.area_chart(df["Position"])

st.subheader("Sample of results")
st.dataframe(df.tail(10))

# 3) Optional: AI explanation of the backtest (provider-agnostic placeholder)
with st.expander("Ask AI to critique these results"):
    st.write("Paste an API key for your preferred provider below and click 'Explain'. "
             "Keep the AI as an analyst (no trade execution).")
    model = st.selectbox("Provider", ["(choose)", "OpenAI", "Anthropic", "Other"])
    api_key = st.text_input("API key", type="password")
    if st.button("Explain") and api_key and model != "(choose)":
        summary = (
            f"Ticker: {ticker}\nPeriod: {start} to {end}\n"
            f"Short SMA: {short_win}, Long SMA: {long_win}\n"
            f"Total return: {total_ret:.2%}, CAGR: {cagr if pd.isna(cagr) else f'{cagr:.2%}'}\n"
            f"Sharpe: {sharpe if pd.isna(sharpe) else f'{sharpe:.2f}'}, Max DD: {mdd:.2%}\n"
            "Please assess overfitting risk, regime sensitivity, and suggest robustness checks."
        )
        st.code(summary)

        # Pseudocode: swap for the provider’s official client per docs.
        # Example (OpenAI / Anthropic) intentionally omitted to avoid hardcoding model names / SDK drift.
        st.info("Use the provider's official docs to send `summary` as the prompt and display the response.")
