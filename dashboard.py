import streamlit as st
import requests
import plotly.graph_objects as go
from datetime import timedelta

API_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(page_title="Hybrid Stock Forecaster", layout="wide")

st.title("🧠 Hybrid AI Forecaster")
st.markdown("Combines **Stacked LSTM** (Math) with **Fundamentals & News** (Reality).")

with st.sidebar:
    ticker = st.text_input("Enter Ticker", "NVDA").upper()
    if st.button("Run Hybrid Analysis"):
        with st.spinner("Training LSTM & Analyzing Fundamentals..."):
            try:
                response = requests.post(API_URL, json={"ticker": ticker})
                if response.status_code == 200:
                    st.session_state['hybrid_data'] = response.json()
                else:
                    st.error("Prediction failed.")
            except Exception as e:
                st.error(f"Error: {e}")

if 'hybrid_data' in st.session_state:
    data = st.session_state['hybrid_data']
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${data['current_price']}")
    c2.metric("Fundamental Score", f"{data['fund_score']}/1.0")
    c3.metric("News Sentiment", f"{data['sent_score']}")
    
    raw_end = data['raw_forecast'][-1]
    tuned_end = data['tuned_forecast'][-1]
    impact = round(tuned_end - raw_end, 2)
    c4.metric("Reality Adjustment", f"${impact}", delta="Bias Applied")

    st.subheader("🔮 Forecast: Math vs. Reality")
    
    import yfinance as yf
    last_date = yf.Ticker(data['symbol']).history(period="1d").index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=data['raw_forecast'], 
        mode='lines', 
        name='Raw LSTM (Blind Math)',
        line=dict(color='gray', dash='dot')
    ))
    
    line_color = '#00ff00' if tuned_end > data['current_price'] else '#ff0000'
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=data['tuned_forecast'], 
        mode='lines', 
        name='Fine-Tuned (Fund + News)',
        line=dict(color=line_color, width=4)
    ))
    
    fig.update_layout(height=500, template="plotly_dark", title=f"30-Day Projection for {data['symbol']}")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📝 Analyst Verdict")
    st.write(data['report'])
    # c_left, c_right = st.columns([2, 1])
    # with c_left:
    #     st.subheader("📝 Analyst Verdict")
    #     st.write(data['report'])
    # with c_right:
    #     st.subheader("📰 Key Headlines")
    #     for h in data.get('headlines', []):
    #         st.caption(f"• {h}")