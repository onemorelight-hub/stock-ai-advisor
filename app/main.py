import pandas as pd
from prophet import Prophet
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import plotly.graph_objects as go
import logging
import yfinance as yf
from fpdf import FPDF
from uuid import uuid4
import ta

# Setup logging
# logging.basicConfig(level=logging.INFO, filename="stock_forecast.log", format="%(asctime)s - %(levelname)s - %(message)s")


# Streamlit configuration
st.set_page_config(page_title="Stock AI Advisor", layout="wide", initial_sidebar_state="expanded")

# CSS for light theme styling
st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    .stButton>button { background-color: #1e88e5; color: white; border-radius: 5px; }
    .stTabs { background-color: #ffffff; border-radius: 10px; padding: 10px; }
    .metric-box { border: 1px solid #ddd; border-radius: 5px; padding: 10px; background-color: #fafafa; }
    .tooltip { position: relative; display: inline-block; cursor: pointer; }
    .tooltip .tooltiptext { visibility: hidden; width: 200px; background-color: #555; color: #fff; text-align: center; border-radius: 5px; padding: 5px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -100px; opacity: 0; transition: opacity 0.3s; }
    .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
    .warning-box { border: 2px solid #ff9800; border-radius: 5px; padding: 10px; background-color: #fff3e0; }
    .info-box { border: 2px solid #2196F3; border-radius: 5px; padding: 10px; background-color: #e3f2fd; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock AI Advisor")
st.markdown("**Advanced stock price forecasting. Upload your data to get started.**")
st.markdown("**Expected CSV Format**: Columns `Date`, `Price`, `Vol.` from Investing.com (10 years of data recommended).")

# NIFTY 50 tickers
NIFTY_50 = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 
    'BAJFINANCE.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS', 'ITC.NS', 'AXISBANK.NS', 'DMART.NS', 'HCLTECH.NS', 'MARUTI.NS', 
    'TITAN.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'SUNPHARMA.NS', 'BAJAJFINSV.NS', 'LT.NS', 'TECHM.NS', 'WIPRO.NS', 
    'INDUSINDBK.NS', 'ADANIPORTS.NS', 'POWERGRID.NS', 'NTPC.NS', 'HDFCLIFE.NS', 'SBILIFE.NS', 'DIVISLAB.NS', 
    'BRITANNIA.NS', 'JSWSTEEL.NS', 'TATASTEEL.NS', 'GRASIM.NS', 'CIPLA.NS', 'EICHERMOT.NS', 'SHREECEM.NS', 
    'HEROMOTOCO.NS', 'DRREDDY.NS', 'TATACONSUM.NS', 'BPCL.NS', 'ONGC.NS', 'COALINDIA.NS', 'IOC.NS', 'HINDALCO.NS', 
    'UPL.NS', 'GAIL.NS', 'ADANIENT.NS', 'NHPC.NS'
]

@st.cache_data
def preprocess_data(df, ticker):
    """Preprocess stock data for Prophet modeling with flexible column mapping."""
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0) if df.columns.nlevels > 1 else df.columns
        df.columns = df.columns.str.strip().str.lower()

        date_cols = [col for col in df.columns if col in ['date', 'trade date', 'trading date']]
        price_cols = [col for col in df.columns if col in ['price', 'close', 'closing price', 'adjusted close']]
        vol_cols = [col for col in df.columns if col in ['vol.', 'volume', 'vol', 'trade volume']]

        if not date_cols:
            date_col = st.selectbox(f"Select Date column for {ticker}", df.columns, key=f"date_{ticker}")
        else:
            date_col = date_cols[0]
        if not price_cols:
            price_col = st.selectbox(f"Select Price column for {ticker}", df.columns, key=f"price_{ticker}")
        else:
            price_col = price_cols[0]
        vol_col = vol_cols[0] if vol_cols else None

        required_columns = {date_col, price_col}
        if not required_columns.issubset(set(df.columns)):
            raise ValueError(f"CSV for {ticker} must contain Date and Price columns. Selected: {date_col}, {price_col}")

        df = df.copy()
        df = df.rename(columns={date_col: 'date', price_col: 'price', vol_col: 'vol.' if vol_col else None})
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors='coerce')
        df["price"] = pd.to_numeric(df["price"], errors='coerce')
        
        if "vol." in df.columns:
            df["vol."] = pd.to_numeric(df["vol."], errors='coerce')
        else:
            df["vol."] = np.nan

        df = df[["date", "price", "vol."]].dropna(subset=["date", "price"])
        df = df.rename(columns={"date": "ds", "price": "y", "vol.": "vol"})
        df['y'] = np.log(df['y'] + 1)
        df['vol'] = df['vol'].fillna(method='ffill').fillna(method='bfill').fillna(df['vol'].median() if df['vol'].notna().sum() > 0 else 0)
        df = df.sort_values('ds').reset_index(drop=True)
        
        if len(df) < 252:
            raise ValueError(f"Insufficient data for {ticker}: At least 1 year of data required.")
        
        return df
    except Exception as e:
        logging.error(f"Preprocessing error for {ticker}: {str(e)}")
        st.error(f"Error processing data for {ticker}: {str(e)}. Please ensure CSV has valid Date and Price columns.")
        return None

@st.cache_data
def fetch_yfinance_data(ticker, years=10):
    """Fetch real-time stock data from Yahoo Finance for Indian stocks."""
    if not ticker.endswith('.NS'):
        st.error(f"Invalid ticker {ticker}: Only Indian stocks with .NS suffix are supported.")
        return None
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False, group_by='column')
        if stock_data.empty:
            raise ValueError(f"No data fetched from Yahoo Finance for {ticker}.")
        
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)
        stock_data = stock_data.reset_index()
        df = stock_data[['Date', 'Close', 'Volume']].rename(columns={'Date': 'date', 'Close': 'price', 'Volume': 'vol.'})
        return df
    except Exception as e:
        logging.error(f"Yahoo Finance fetch error for {ticker}: {str(e)}")
        st.error(f"Error fetching data from Yahoo Finance for {ticker}: {str(e)}")
        return None

@st.cache_data
def calculate_cagr(df, timeframe):
    """Calculate historical CAGR and timeframe-specific cap based on volume trends."""
    start_price = np.exp(df['y'].iloc[0]) - 1
    end_price = np.exp(df['y'].iloc[-1]) - 1
    years = (df['ds'].max() - df['ds'].min()).days / 365.25
    if years > 0 and start_price > 0:
        cagr = ((end_price / start_price) ** (1 / years) - 1) * 100
        
        if 'vol' in df.columns and df['vol'].notna().sum() > 0:
            vol_start = df['vol'].iloc[:int(len(df)/10)].mean()
            vol_end = df['vol'].iloc[-int(len(df)/10):].mean()
            vol_growth = (vol_end - vol_start) / vol_start if vol_start > 0 else 0
        else:
            vol_growth = 0

        if timeframe in ['10 Days', '1 Month', '2 Months', '3 Months']:
            alpha = 1.0 + 0.1 * vol_growth if vol_growth > 0.5 else 1.0
            cagr_min, cagr_max = 10.0, 20.0
        elif timeframe in ['6 Months', '1 Year', '2 Years', '3 Years']:
            alpha = 0.9 + 0.05 * vol_growth if vol_growth > 0.5 else 0.9
            cagr_min, cagr_max = 8.0, 15.0
        else:
            alpha = 0.8 + 0.05 * vol_growth if vol_growth > 0.5 else 0.8
            cagr_min, cagr_max = 6.0, 12.0
        
        cagr_cap = min(max(alpha * cagr, cagr_min), cagr_max)
        return cagr, cagr_cap
    return None, None

@st.cache_data
def calculate_technical_indicators(df):
    """Calculate RSI, MACD, Bollinger Bands, Stochastic Oscillator, and ATR."""
    df = df.copy()
    df['price'] = np.exp(df['y']) - 1
    
    df['rsi'] = ta.momentum.RSIIndicator(df['price'], window=14).rsi()
    latest_rsi = df['rsi'].iloc[-1]
    rsi_signal = "Buy" if latest_rsi < 30 else "Sell" if latest_rsi > 70 else "Hold"
    
    df['macd'] = ta.trend.MACD(df['price'], window_slow=26, window_fast=12, window_sign=9).macd()
    df['macd_signal_line'] = ta.trend.MACD(df['price'], window_slow=26, window_fast=12, window_sign=9).macd_signal()
    latest_macd = df['macd'].iloc[-1]
    latest_macd_signal = df['macd_signal_line'].iloc[-1]
    macd_signal = "Buy" if latest_macd > latest_macd_signal else "Sell" if latest_macd < latest_macd_signal else "Hold"
    
    bb = ta.volatility.BollingerBands(df['price'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    latest_price = df['price'].iloc[-1]
    latest_bb_upper = df['bb_upper'].iloc[-1]
    latest_bb_lower = df['bb_lower'].iloc[-1]
    bb_signal = "Buy" if latest_price <= latest_bb_lower else "Sell" if latest_price >= latest_bb_upper else "Hold"
    
    stoch = ta.momentum.StochasticOscillator(df['price'], df['price'], df['price'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    latest_k = df['stoch_k'].iloc[-1]
    latest_d = df['stoch_d'].iloc[-1]
    stoch_signal = "Buy" if latest_k > latest_d and latest_k < 20 else "Sell" if latest_k < latest_d and latest_k > 80 else "Hold"
    
    df['atr'] = ta.volatility.AverageTrueRange(df['price'], df['price'], df['price'], window=14).average_true_range()
    latest_atr = df['atr'].iloc[-1]
    atr_threshold = 0.03 * latest_price
    price_trend = df['price'].iloc[-1] - df['price'].iloc[-10] if len(df) >= 10 else 0
    atr_signal = "Buy" if latest_atr < 0.01 * latest_price and price_trend > 0 else "Sell" if latest_atr > atr_threshold and price_trend < 0 else "Hold"
    
    return df, {
        'rsi': latest_rsi, 'rsi_signal': rsi_signal,
        'macd_signal': macd_signal,
        'bb_signal': bb_signal, 'bb_upper': latest_bb_upper, 'bb_lower': latest_bb_lower,
        'stoch_k': latest_k, 'stoch_d': latest_d, 'stoch_signal': stoch_signal,
        'atr': latest_atr, 'atr_signal': atr_signal
    }

@st.cache_data
def monte_carlo_simulation(forecast, periods, n_simulations=1000):
    """Run Monte Carlo simulations for risk analysis."""
    try:
        last_price = forecast['yhat'].iloc[-periods-1]
        volatility = np.std(np.diff(np.log(forecast['yhat'] + 1))[-252:]) * np.sqrt(252)
        daily_vol = volatility / np.sqrt(252)
        
        simulations = []
        for _ in range(n_simulations):
            prices = [last_price]
            for _ in range(periods):
                drift = (forecast['yhat'].iloc[-1] - last_price) / periods / last_price
                shock = np.random.normal(0, daily_vol)
                price = prices[-1] * np.exp(drift + shock)
                prices.append(price)
            simulations.append(prices)
        
        simulations = np.array(simulations)
        lower_bound = np.percentile(simulations[:, -1], 5)
        upper_bound = np.percentile(simulations[:, -1], 95)
        return lower_bound, upper_bound
    except Exception as e:
        logging.error(f"Monte Carlo error: {str(e)}")
        return None, None

@st.cache_resource
def train_and_forecast(df, periods, timeframe_years, timeframe, cagr_cap):
    """Train Prophet model and generate forecast with timeframe-specific CAGR cap."""
    try:
        df_range = df[df['ds'] > df['ds'].max() - pd.DateOffset(years=timeframe_years)].copy()
        model = Prophet(
            changepoint_prior_scale=0.05 if timeframe_years > 3 else 0.1,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_prior_scale=10.0
        )
        if 'vol' in df_range.columns:
            df_range['vol'] = df_range['vol'].fillna(df_range['vol'].median())
            model.add_regressor('vol')

        model.fit(df_range)
        future = model.make_future_dataframe(periods=periods)
        if 'vol' in df_range.columns:
            vol_df = df_range[['ds', 'vol']].set_index('ds').resample('D').ffill().reset_index()
            future = future.merge(vol_df, on='ds', how='left')
            future['vol'] = future['vol'].fillna(method='ffill').fillna(method='bfill').fillna(df_range['vol'].median())

        forecast = model.predict(future)
        forecast['yhat'] = np.exp(forecast['yhat']) - 1
        forecast['yhat_lower'] = np.exp(forecast['yhat_lower']) - 1
        forecast['yhat_upper'] = np.exp(forecast['yhat_upper']) - 1
        
        if periods > 365 and cagr_cap:
            years = periods / 365.25
            max_growth = (1 + cagr_cap / 100) ** years
            max_price = (np.exp(df['y'].iloc[-1]) - 1) * max_growth
            forecast['yhat'] = forecast['yhat'].clip(upper=max_price)
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(upper=max_price)
        
        return model, forecast
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        st.error(f"Error training model: {str(e)}")
        return None, None

def plot_forecast(model, forecast, periods, timeframe_name, current_price, lower_bound=None, upper_bound=None, ticker="Stock"):
    """Generate interactive Plotly forecast plot with Monte Carlo bounds."""
    fig = go.Figure()
    historical = forecast[forecast['ds'] <= forecast['ds'].iloc[-periods-1]]
    predicted = forecast[forecast['ds'] >= forecast['ds'].iloc[-periods-1]]
    
    fig.add_trace(go.Scatter(x=historical['ds'], y=historical['yhat'], name="Historical Fit", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=predicted['ds'], y=predicted['yhat'], name="Forecast", line=dict(color='green')))
    fig.add_trace(go.Scatter(x=predicted['ds'], y=predicted['yhat_upper'], name="Upper CI", line=dict(color='green', dash='dash'), opacity=0.2))
    fig.add_trace(go.Scatter(x=predicted['ds'], y=predicted['yhat_lower'], name="Lower CI", line=dict(color='green', dash='dash'), opacity=0.2, fill='tonexty'))
    if lower_bound and upper_bound:
        fig.add_trace(go.Scatter(x=predicted['ds'], y=[lower_bound] * len(predicted), name="MC 5th %ile", line=dict(color='red', dash='dot'), opacity=0.3))
        fig.add_trace(go.Scatter(x=predicted['ds'], y=[upper_bound] * len(predicted), name="MC 95th %ile", line=dict(color='red', dash='dot'), opacity=0.3))
    fig.add_vline(x=forecast['ds'].iloc[-periods-1], line=dict(color='red', dash='dash'))
    fig.add_hline(y=current_price, line=dict(color='purple', dash='dot'), annotation_text="Current Price")
    
    fig.update_layout(
        title=f"Forecast for {ticker} - {timeframe_name}",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        template="plotly_white",
        hovermode="x unified",
        height=400
    )
    return fig

def backtest_model(df, periods, timeframe_years, timeframe, cagr_cap):
    """Backtest the model on the last year of data."""
    try:
        train_end = df['ds'].max() - pd.DateOffset(years=1)
        train_df = df[df['ds'] <= train_end].copy()
        test_df = df[(df['ds'] > train_end) & (df['ds'] <= train_end + pd.DateOffset(days=periods))].copy()
        
        if len(test_df) < periods * 0.5:
            return None, None, None
        
        model, forecast = train_and_forecast(train_df, periods, timeframe_years, timeframe, cagr_cap)
        if model is None or forecast is None:
            return None, None, None
        forecast_test = forecast[forecast['ds'].isin(test_df['ds'])][['ds', 'yhat']]
        merged = forecast_test.merge(test_df[['ds', 'y']], on='ds')
        merged['y'] = np.exp(merged['y']) - 1
        
        mae = mean_absolute_error(merged['y'], merged['yhat'])
        mape = mean_absolute_percentage_error(merged['y'], merged['yhat']) * 100
        return mae, mape, len(merged)
    except Exception as e:
        logging.error(f"Backtest error: {str(e)}")
        return None, None, None


def vote_final_action(decisions, technicals):
    """Determine final investment action based on trend votes and technical indicators."""
    vote = {"Buy": 0, "Sell": 0, "Hold": 0}
    for name, change in decisions.items():
        if change > 2:
            vote["Buy"] += 1
        elif change < -2:
            vote["Sell"] += 1
        else:
            vote["Hold"] += 1
    
    for signal in [technicals['rsi_signal'], technicals['macd_signal'], technicals['bb_signal'], technicals['stoch_signal'], technicals['atr_signal']]:
        if signal == "Buy":
            vote["Buy"] += 1
        elif signal == "Sell":
            vote["Sell"] += 1
        elif signal == "Hold":
            vote["Hold"] += 1
    
    final = max(vote, key=vote.get)
    return final, vote

def analyze_timeframes(df, current_price, ticker):
    """Analyze multiple investment horizons and provide recommendations for a single ticker."""
    timeframes = {
        '10 Days': (10, 1),
        '1 Month': (30, 1),
        '2 Months': (60, 1),
        '3 Months': (90, 2),
        '6 Months': (180, 3),
        '1 Year': (365, 5),
        '2 Years': (730, 7),
        '3 Years': (1095, 10),
        '5 Years': (1825, 10),
        '7 Years': (2555, 10),
        '10 Years': (3650, 10)
    }
    
    decisions = {}
    buy_prices = {}
    sell_prices = {}
    profit_potentials = {}
    backtest_results = {}
    plotly_figs = {}
    mc_bounds = {}
    cagr_caps = {}
    
    df, technicals = calculate_technical_indicators(df)
    
    progress_bar = st.progress(0)
    for i, (name, (periods, years)) in enumerate(timeframes.items()):
        with st.spinner(f"Analyzing {ticker} for {name}..."):
            cagr, cagr_cap = calculate_cagr(df, name)
            cagr_caps[name] = cagr_cap
            model, forecast = train_and_forecast(df, periods, years, name, cagr_cap)
            if model is None or forecast is None:
                continue
            trend_change = forecast.iloc[-1]['yhat'] - forecast.iloc[-periods-1]['yhat']
            decisions[name] = trend_change
            
            lower_bound, upper_bound = monte_carlo_simulation(forecast, periods)
            mc_bounds[name] = {'lower': lower_bound, 'upper': upper_bound}
            
            fig = plot_forecast(model, forecast, periods, name, current_price, lower_bound, upper_bound, ticker)
            plotly_figs[name] = fig
            
            df_range = df[df['ds'] > df['ds'].max() - pd.DateOffset(years=years)]
            quantile = 0.25 if periods > 180 else 0.20
            floor = current_price * 0.95 if periods > 180 else current_price * 0.9
            buy_price = max(np.exp(df_range['y'].quantile(quantile)) - 1, floor)
            buy_prices[name] = buy_price
            
            quantile = 0.70 if periods > 365 else 0.80
            forecast_period = forecast.iloc[-periods:]['yhat'].quantile(quantile)
            sell_prices[name] = forecast_period
            
            profit_potentials[name] = ((sell_prices[name] - buy_prices[name]) / buy_prices[name] * 100) if buy_prices[name] > 0 else 0
            
            mae, mape, test_size = backtest_model(df, periods, years, name, cagr_cap)
            backtest_results[name] = {'mae': mae, 'mape': mape, 'test_size': test_size}
            
            progress_bar.progress((i + 1) / len(timeframes))
    
    short_term_buy = min([buy_prices[name] for name in ['10 Days', '1 Month', '2 Months', '3 Months']])
    long_term_buy = np.average(
        [buy_prices[name] for name in ['6 Months', '1 Year', '2 Years', '3 Years', '5 Years', '7 Years', '10 Years']],
        weights=[1, 1, 2, 2, 3, 3, 3]
    )
    
    final_action, vote_breakdown = vote_final_action(decisions, technicals)
    
    return {
        'decisions': decisions,
        'buy_prices': buy_prices,
        'sell_prices': sell_prices,
        'profit_potentials': profit_potentials,
        'backtest_results': backtest_results,
        'short_term_buy': short_term_buy,
        'long_term_buy': long_term_buy,
        'plotly_figs': plotly_figs,
        'cagr': cagr,
        'cagr_caps': cagr_caps,
        'technicals': technicals,
        'mc_bounds': mc_bounds,
        'final_action': final_action,
        'vote_breakdown': vote_breakdown,
        'current_price': current_price
    }

# Main UI
st.sidebar.markdown("### 📈 Stock AI Advisor")
st.sidebar.markdown("**Developed by Anjan Jana**")
st.sidebar.markdown("<a href='mailto:link2anjan@gmail.com'>**Email: link2anjan@gmail.com**</a>", unsafe_allow_html=True)

data_source = st.sidebar.radio("Choose data source", ["Upload CSV", "Fetch from Yahoo Finance"])

uploaded_file = None
ticker = ""
years = 10

if data_source == "Upload CSV":
    st.sidebar.markdown("Upload a CSV with 10 years of stock data.")
    st.sidebar.markdown("**Required columns**: Date (or similar), Price/Close, Vol./Volume <span class='tooltip'>ⓘ<span class='tooltiptext'>Date in DD-MM-YYYY or similar, Price as numeric, Volume as numeric or with K/M/B suffixes.</span></span>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"], help="Ensure CSV has Date, Price, and optionally Volume columns.")
    ticker_input = st.sidebar.text_input("Enter ticker for CSV (e.g., NHPC.NS)", value="NHPC.NS")
    if ticker_input and not ticker_input.endswith('.NS'):
        st.sidebar.error("Ticker must end with .NS for Indian stocks.")
    else:
        ticker = ticker_input.strip() if ticker_input else ""
else:
    st.sidebar.markdown("Select or enter an Indian stock ticker (.NS) from Yahoo Finance.")
    ticker_select = st.sidebar.selectbox("Select from NIFTY 50", [""] + NIFTY_50, index=0)
    ticker_manual = st.sidebar.text_input("Or enter ticker (e.g., NHPC.NS)", value="")
    submit_button = st.sidebar.button("Submit")
    
    if submit_button:
        if ticker_select and ticker_manual:
            st.sidebar.error("Please select a ticker from the dropdown or enter one manually, not both.")
        elif ticker_select:
            ticker = ticker_select
        elif ticker_manual:
            if not ticker_manual.endswith('.NS'):
                st.sidebar.error("Ticker must end with .NS for Indian stocks.")
            else:
                ticker = ticker_manual.strip()
        if not ticker:
            st.sidebar.error("Please select or enter a valid ticker.")
        years = st.sidebar.slider("Years of data", min_value=1, max_value=20, value=10)

if (uploaded_file and ticker) or (data_source == "Fetch from Yahoo Finance" and ticker and submit_button):
    ticker_results = {}
    with st.spinner(f"Processing data for {ticker}..."):
        if data_source == "Upload CSV":
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = fetch_yfinance_data(ticker, years)
            if raw_df is None:
                st.stop()
        
        df = preprocess_data(raw_df, ticker)
        if df is None:
            st.stop()
        
        current_price = np.exp(df['y'].iloc[-1]) - 1
        st.markdown(f"**Current Stock Price for {ticker} (as of {df['ds'].max().strftime('%Y-%m-%d')}): INR {current_price:.2f}**")
        cagr, _ = calculate_cagr(df, '10 Years')
        if cagr:
            st.markdown(f"**Historical CAGR for {ticker}**: {cagr:.2f}% <span class='tooltip'>ⓘ<span class='tooltiptext'>Compound Annual Growth Rate based on historical data.</span></span>", unsafe_allow_html=True)
        
        results = analyze_timeframes(df, current_price, ticker)
        ticker_results[ticker] = results
        
        st.markdown(f"""
        <div class='warning-box'>
        ⚠️ <b>Long-Term Forecast Warning for {ticker}</b>: Predictions for 5+ years are capped at {results['cagr_caps'].get('10 Years', 10):.2f}% CAGR based on historical trends and volume analysis. Combine with fundamental analysis.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"### 📊 Technical Indicators for {ticker}")
        st.markdown(f"<div class='metric-box'>RSI: {results['technicals']['rsi']:.2f} ({results['technicals']['rsi_signal']}) <span class='tooltip'>ⓘ<span class='tooltiptext'>RSI > 70 indicates Sell, < 30 indicates Buy.</span></span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'>MACD Signal: {results['technicals']['macd_signal']} <span class='tooltip'>ⓘ<span class='tooltiptext'>MACD line above signal line suggests Buy, below suggests Sell.</span></span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'>Bollinger Bands: {results['technicals']['bb_signal']} (Upper: INR {results['technicals']['bb_upper']:.2f}, Lower: INR {results['technicals']['bb_lower']:.2f}) <span class='tooltip'>ⓘ<span class='tooltiptext'>Price above upper band suggests Sell, below lower band suggests Buy.</span></span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'>Stochastic Oscillator: %K={results['technicals']['stoch_k']:.2f}, %D={results['technicals']['stoch_d']:.2f} ({results['technicals']['stoch_signal']}) <span class='tooltip'>ⓘ<span class='tooltiptext'>%K > %D and < 20 suggests Buy, %K < %D and > 80 suggests Sell.</span></span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'>ATR: {results['technicals']['atr']:.2f} ({results['technicals']['atr_signal']}) <span class='tooltip'>ⓘ<span class='tooltiptext'>Low ATR with upward trend suggests Buy, high ATR with downward trend suggests Sell.</span></span></div>", unsafe_allow_html=True)
        
        tabs = st.tabs(["Overview", "Short-Term", "Medium-Term", "Long-Term", "Charts", "Backtest"])
        
        with tabs[0]:
            st.markdown(f"### 📊 Overview for {ticker}")
            st.markdown(f"**Final Recommendation**: {results['final_action']} <span class='tooltip'>ⓘ<span class='tooltiptext'>Based on trend votes and technical indicators (RSI, MACD, BB, Stochastic, ATR).</span></span>", unsafe_allow_html=True)
            st.markdown(f"**Short-Term Buy Price (10D–3M)**: INR {results['short_term_buy']:.2f} <span class='tooltip'>ⓘ<span class='tooltiptext'>Minimum buy price for short-term horizons.</span></span>", unsafe_allow_html=True)
            st.markdown(f"**Long-Term Buy Price (6M–10Y)**: INR {results['long_term_buy']:.2f} <span class='tooltip'>ⓘ<span class='tooltiptext'>Weighted average buy price for long-term horizons.</span></span>", unsafe_allow_html=True)
            st.markdown("#### Vote Breakdown")
            st.write(results['vote_breakdown'])
        
        with tabs[1]:
            st.markdown(f"### 📅 Short-Term Horizons (10 Days to 3 Months) for {ticker}")
            for name in ['10 Days', '1 Month', '2 Months', '3 Months']:
                st.markdown(f"#### {name}")
                st.markdown(f"<div class='metric-box'>Trend Change: INR {results['decisions'][name]:.2f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'>Buy Price: INR {results['buy_prices'][name]:.2f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'>Sell Price: INR {results['sell_prices'][name]:.2f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'>Potential Profit: {results['profit_potentials'][name]:.2f}%</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'>CAGR Cap: {results['cagr_caps'][name]:.2f}%</div>", unsafe_allow_html=True)
                if results['backtest_results'][name]['mape']:
                    st.markdown(f"<div class='metric-box'>Backtest MAPE: {results['backtest_results'][name]['mape']:.2f}% (over {results['backtest_results'][name]['test_size']} days)</div>", unsafe_allow_html=True)
                if results['mc_bounds'][name]['lower']:
                    st.markdown(f"<div class='metric-box'>Monte Carlo 5th–95th Percentile: INR {results['mc_bounds'][name]['lower']:.2f}–INR {results['mc_bounds'][name]['upper']:.2f}</div>", unsafe_allow_html=True)
        
        with tabs[2]:
            st.markdown(f"### 📅 Medium-Term Horizons (6 Months to 3 Years) for {ticker}")
            for name in ['6 Months', '1 Year', '2 Years', '3 Years']:
                st.markdown(f"#### {name}")
                st.markdown(f"<div class='metric-box'>Trend Change: INR {results['decisions'][name]:.2f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'>Buy Price: INR {results['buy_prices'][name]:.2f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'>Sell Price: INR {results['sell_prices'][name]:.2f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'>Potential Profit: {results['profit_potentials'][name]:.2f}%</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'>CAGR Cap: {results['cagr_caps'][name]:.2f}%</div>", unsafe_allow_html=True)
                if results['backtest_results'][name]['mape']:
                    st.markdown(f"<div class='metric-box'>Backtest MAPE: {results['backtest_results'][name]['mape']:.2f}% (over {results['backtest_results'][name]['test_size']} days)</div>", unsafe_allow_html=True)
                if results['mc_bounds'][name]['lower']:
                    st.markdown(f"<div class='metric-box'>Monte Carlo 5th–95th Percentile: INR {results['mc_bounds'][name]['lower']:.2f}–INR {results['mc_bounds'][name]['upper']:.2f}</div>", unsafe_allow_html=True)
        
        with tabs[3]:
            st.markdown(f"### 📅 Long-Term Horizons (5 to 10 Years) for {ticker}")
            for name in ['5 Years', '7 Years', '10 Years']:
                st.markdown(f"#### {name}")
                st.markdown(f"<div class='metric-box'>Trend Change: INR {results['decisions'][name]:.2f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'>Buy Price: INR {results['buy_prices'][name]:.2f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'>Sell Price: INR {results['sell_prices'][name]:.2f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'>Potential Profit: {results['profit_potentials'][name]:.2f}%</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'>CAGR Cap: {results['cagr_caps'][name]:.2f}%</div>", unsafe_allow_html=True)
                if results['backtest_results'][name]['mape']:
                    st.markdown(f"<div class='metric-box'>Backtest MAPE: {results['backtest_results'][name]['mape']:.2f}% (over {results['backtest_results'][name]['test_size']} days)</div>", unsafe_allow_html=True)
                if results['mc_bounds'][name]['lower']:
                    st.markdown(f"<div class='metric-box'>Monte Carlo 5th–95th Percentile: INR {results['mc_bounds'][name]['lower']:.2f}–INR {results['mc_bounds'][name]['upper']:.2f}</div>", unsafe_allow_html=True)
        
        with tabs[4]:
            st.markdown(f"### 📈 Forecast Charts for {ticker}")
            for name, fig in results['plotly_figs'].items():
                st.markdown(f"#### {name}")
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[5]:
            st.markdown(f"### 🔍 Backtest Results for {ticker}")
            for name, result in results['backtest_results'].items():
                if result['mape']:
                    st.markdown(f"#### {name}")
                    st.markdown(f"- MAE: INR {result['mae']:.2f}")
                    st.markdown(f"- MAPE: {result['mape']:.2f}%")
                    st.markdown(f"- Test Period: {result['test_size']} days")
                else:
                    st.markdown(f"#### {name}: Insufficient data for backtesting")
    
else:
    st.markdown("""
    <div class='info-box'>
    Please select a data source, upload a CSV file with a valid .NS ticker, or select/enter an Indian stock ticker and click Submit to start the analysis. <span class='tooltip'>ⓘ<span class='tooltiptext'>CSV should include Date, Price/Close, and optionally Vol./Volume columns. Yahoo Finance fetches data for .NS tickers (e.g., NHPC.NS).</span></span>
    </div>
    """, unsafe_allow_html=True)
