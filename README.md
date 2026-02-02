# Quantitative Option Pricing Engine

An interactive financial dashboard for real-time option pricing and risk analysis. Built with **Python**, **Streamlit**, and **Plotly**.

## Features
- **Real-Time Data:** Fetches live market data using Yahoo Finance API.
- **Black-Scholes Model:** Calculates theoretical option prices and Greeks ($\Delta, \Gamma, \Theta, \nu, \rho$).
- **Monte Carlo Simulation:** Simulates thousands of future price paths to verify theoretical pricing.
- **Interactive Analytics:** - TradingView-style candlestick charts with zoom & pan.
  - 3D-like volatility surface heatmaps using Plotly.
  - Implied Volatility (IV) reverse calculation.

## Technologies
- **Core:** Python 3.10+
- **Math/Quant:** NumPy, SciPy
- **Data:** yfinance
- **Visualization:** Plotly, Streamlit

## Installation

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/jimivayrynen/OptionPricingEngine.git](https://github.com/jimivayrynen/OptionPricingEngine.git)
   cd OptionPricingEngine