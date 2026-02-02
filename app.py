import streamlit as st
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from models.black_scholes import BlackScholes
from models.monte_carlo import MonteCarloPricing
from models.implied_volatility import calculate_implied_volatility

# --- PAGE SETTINGS ---
st.set_page_config(
    page_title="Quantitative Finance Engine",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 5px;
    }
    .stPlotlyChart {
        background-color: #0e1117;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

@st.cache_data
def get_risk_free_rate():
    try:
        treasury = yf.Ticker("^TNX")
        hist = treasury.history(period="1d")
        if hist.empty:
            return 0.0425
        return hist['Close'].iloc[-1] / 100
    except:
        return 0.05

@st.cache_data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    if hist.empty:
        return None, None
    hist['Log_Ret'] = np.log(hist['Close'] / hist['Close'].shift(1))
    volatility = hist['Log_Ret'].std() * np.sqrt(252)
    current_price = hist['Close'].iloc[-1]
    return current_price, volatility

@st.cache_data
def get_historical_data(ticker, period="2y"): # Changed max -> 2y for speed
    """Fetches historical data. optimized for performance."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist

@st.cache_data
def get_market_summary():
    tickers = ['SPY', 'QQQ', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMD', 'COIN']
    summary = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if len(hist) < 2: continue
            current = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            change = ((current - prev) / prev) * 100
            summary.append({'ticker': ticker, 'price': current, 'change': change})
        except: continue
    return summary

# --- CHARTS ---

def plot_tradingview_chart(ticker):
    """Creates a TradingView-style interactive candlestick chart."""
    # Fetch 2 years of data for smooth scrolling, but enough history for analysis
    df = get_historical_data(ticker, period="2y")
    
    if df.empty:
        st.error("No data found.")
        return None

    fig = go.Figure()

    # Candlestick trace
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=ticker,
        increasing_line_color='#26a69a', # TradingView Green
        decreasing_line_color='#ef5350'  # TradingView Red
    ))

    # Add Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(window=50).mean(), mode='lines', name='MA 50', line=dict(color='orange', width=1)))
    
    # Layout with Range Slider
    fig.update_layout(
        title=dict(text=f"{ticker} Price Action", font=dict(size=24)),
        yaxis_title="Price ($)",
        template="plotly_dark",
        xaxis_rangeslider_visible=False, # Piilotetaan oletus slider (se on ruma), käytetään zoomia
        height=600,
        hovermode="x unified",
        dragmode="pan", # Allows dragging the chart
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # TÄRKEÄ: Poistetaan viikonloput (breaks) -> Graafi ei näytä "reikäiseltä"
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]), # hide weekends
        ],
        rangeslider=dict(visible=True, thickness=0.05), # Pieni, siisti scroll bar alhaalla
        type="date"
    )

    # Time range selector buttons
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="ALL")
            ]),
            bgcolor="#2b2b2b",
            font=dict(color="white")
        )
    )

    return fig

def plot_heatmap(spot, strike, T, r, sigma):
    min_price = spot * 0.7
    max_price = spot * 1.3
    spot_prices = np.linspace(min_price, max_price, 50)
    
    call_prices = []
    put_prices = []
    
    for s in spot_prices:
        model = BlackScholes(s, strike, T, r, sigma)
        call_prices.append(model.calculate_price('call'))
        put_prices.append(model.calculate_price('put'))
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_prices, y=call_prices, mode='lines', name='Call Option', line=dict(color='#26a69a', width=3)))
    fig.add_trace(go.Scatter(x=spot_prices, y=put_prices, mode='lines', name='Put Option', line=dict(color='#ef5350', width=3, dash='dash')))
    
    fig.add_vline(x=spot, line_width=1, line_dash="dash", line_color="gray", annotation_text="Spot")
    fig.add_vline(x=strike, line_width=1, line_dash="dot", line_color="white", annotation_text="Strike")
    
    fig.update_layout(
        title=f"Option Price Sensitivity (Strike: ${strike})",
        xaxis_title="Stock Price ($)",
        yaxis_title="Option Price ($)",
        template="plotly_dark",
        hovermode="x unified"
    )
    return fig

def plot_monte_carlo_paths(S, K, T, r, sigma, n_sims=50, n_steps=100):
    dt = T / n_steps
    paths = np.zeros((n_steps + 1, n_sims))
    paths[0] = S
    
    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_sims)
        paths[t] = paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
    fig = go.Figure()
    
    # Plot first 50 paths
    for i in range(min(n_sims, 50)):
        fig.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(width=1, color='rgba(38, 166, 154, 0.2)'), showlegend=False))
        
    # Plot Mean Path
    fig.add_trace(go.Scatter(y=paths.mean(axis=1), mode='lines', name='Mean Path', line=dict(color='white', width=3, dash='dash')))
    fig.add_hline(y=K, line_color="#ef5350", line_dash="dot", annotation_text="Strike Price")
    
    fig.update_layout(
        title=f"Monte Carlo Simulation ({n_sims} iterations)",
        xaxis_title="Time Steps",
        yaxis_title="Price ($)",
        template="plotly_dark"
    )
    return fig

# --- PAGES ---

def show_dashboard():
    st.title("📊 Market Dashboard")
    st.markdown("Real-time technical analysis engine.")
    
    # Top Metrics
    with st.spinner("Fetching market pulse..."):
        summary_data = get_market_summary()
    
    cols = st.columns(4)
    for i, item in enumerate(summary_data):
        col = cols[i % 4]
        col.metric(label=item['ticker'], value=f"${item['price']:.2f}", delta=f"{item['change']:.2f}%")
    
    st.markdown("---")
    
    # TRADINGVIEW STYLE CHART SECTION
    st.subheader("🔎 Technical Analysis")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_ticker = st.selectbox("Select Asset", [x['ticker'] for x in summary_data] + ['GOOGL', 'AMZN', 'META'])
    
    with col2:
        st.caption(f"Interactive Chart: **{selected_ticker}** (Scroll to zoom, Drag to pan)")

    chart_fig = plot_tradingview_chart(selected_ticker)
    if chart_fig:
        # config={'scrollZoom': True} mahdollistaa hiiren rullalla zoomauksen
        st.plotly_chart(chart_fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})


def show_calculator():
    st.title("🧮 Option Pricing Engine")
    
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Parameters")
    ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()
    strike_price = st.sidebar.number_input("Strike Price ($)", value=500.0, step=5.0)
    expiry_date = st.sidebar.date_input("Expiry Date", min_value=datetime.today())
    manual_sigma = st.sidebar.checkbox("Override Volatility?")
    
    try:
        spot_price, sigma_auto = get_stock_data(ticker)
        r = get_risk_free_rate()
        if spot_price is None:
            st.error(f"Ticker '{ticker}' not found.")
            st.stop()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    if manual_sigma:
        sigma = st.sidebar.slider("Volatility (σ)", 0.01, 1.0, float(sigma_auto), 0.01)
    else:
        sigma = sigma_auto
        st.sidebar.info(f"Annualized Volatility (1Y): {sigma:.2%}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Spot Price", f"${spot_price:.2f}")
    col2.metric("Strike Price", f"${strike_price:.2f}")
    col3.metric("Risk-Free Rate", f"{r:.2%}")
    col4.metric("Time to Maturity", f"{(expiry_date - datetime.today().date()).days / 365:.2f} Years")

    T = (expiry_date - datetime.today().date()).days / 365.0

    if T > 0:
        bs = BlackScholes(spot_price, strike_price, T, r, sigma)
        call_price = bs.calculate_price('call')
        put_price = bs.calculate_price('put')
        call_greeks = bs.calculate_greeks('call')
        put_greeks = bs.calculate_greeks('put')

        tab1, tab2, tab3, tab4 = st.tabs(["Prices & Greeks", "Sensitivity Analysis", "Implied Volatility", "Monte Carlo"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"### CALL Value: ${call_price:.2f}")
                st.write(f"**Delta:** {call_greeks['delta']:.4f}")
                st.write(f"**Gamma:** {call_greeks['gamma']:.4f}")
                st.write(f"**Theta:** {call_greeks['theta']/365:.2f}")
                st.write(f"**Vega:** {call_greeks['vega']:.2f}")
            with c2:
                st.error(f"### PUT Value: ${put_price:.2f}")
                st.write(f"**Delta:** {put_greeks['delta']:.4f}")
                st.write(f"**Gamma:** {put_greeks['gamma']:.4f}")
                st.write(f"**Theta:** {put_greeks['theta']/365:.2f}")
                st.write(f"**Vega:** {put_greeks['vega']:.2f}")

        with tab2:
            fig = plot_heatmap(spot_price, strike_price, T, r, sigma)
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("IV Calculator")
            c1, c2 = st.columns(2)
            with c1:
                mp_call = st.number_input("Call Market Price ($)", value=float(call_price))
                if st.button("Calc Call IV"):
                    iv = calculate_implied_volatility(mp_call, spot_price, strike_price, T, r, 'call')
                    st.write(f"IV: {iv:.2%}" if iv else "Error")
            with c2:
                mp_put = st.number_input("Put Market Price ($)", value=float(put_price))
                if st.button("Calc Put IV"):
                    iv = calculate_implied_volatility(mp_put, spot_price, strike_price, T, r, 'put')
                    st.write(f"IV: {iv:.2%}" if iv else "Error")

        with tab4:
            st.subheader("Monte Carlo Simulation")
            sim_iterations = st.slider("Iterations", 1000, 50000, 5000)
            if st.button("Run Simulation"):
                mc = MonteCarloPricing(spot_price, strike_price, T, r, sigma, sim_iterations)
                st.write(f"MC Call: ${mc.calculate_price('call'):.2f}")
                fig_paths = plot_monte_carlo_paths(spot_price, strike_price, T, r, sigma)
                st.plotly_chart(fig_paths, use_container_width=True)

    else:
        st.warning("Select future expiry.")

# --- NAVIGATION ---
page = st.sidebar.radio("Navigation", ["Market Dashboard", "Pricing Calculator"])

if page == "Market Dashboard":
    show_dashboard()
else:
    show_calculator()