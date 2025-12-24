import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import concurrent.futures
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import streamlit.components.v1 as components
import altair as alt
import requests
import io
import random
import trading_bot as bot # Import your new file

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="AI Market Hunter (Precision)", page_icon="🦅")

st.markdown("""
<style>
    div.stButton > button {
        width: 100%;
        background-color: #1E1E1E;
        border: 1px solid #444;
        color: white;
    }
    div.stButton > button:hover {
        border-color: #00FF00;
        color: #00FF00;
    }
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        background-color: #0E1117;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. REAL-TIME LIST BUILDER (API) ---
@st.cache_data(ttl=3600) 
def fetch_realtime_symbols(region):
    """
    Connects to official sources to get the full list of active stocks.
    """
    try:
        if region == "🇮🇳 India (NSE)":
            url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
            headers = {"User-Agent": "Mozilla/5.0"}
            s = requests.get(url, headers=headers).content
            df = pd.read_csv(io.StringIO(s.decode('utf-8')))
            return [f"{x}.NS" for x in df['SYMBOL'].tolist()]
            
        elif region == "🇺🇸 USA (S&P 500)":
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            return table[0]['Symbol'].tolist()
            
    except Exception as e:
        return []
    return []

# --- 3. HELPER: ZOOMED SPARKLINE ---
def make_sparkline(data_series, color_hex):
    df = data_series.reset_index(drop=True).to_frame(name='price')
    df['index'] = df.index
    
    chart = alt.Chart(df).mark_line(
        color=color_hex, strokeWidth=2
    ).encode(
        x=alt.X('index', axis=None),
        y=alt.Y('price', scale=alt.Scale(zero=False, padding=1), axis=alt.Axis(labels=True, tickCount=3)),
        tooltip=['price']
    ).properties(height=80, width='container').configure_axis(grid=False).configure_view(strokeWidth=0)
    
    return chart

# --- 4. STRATEGY CLASS ---
class TrendStrategy(Strategy):
    fast_ma = 9
    slow_ma = 21
    rsi_limit = 60 # Stricter default
    def init(self):
        self.sma1 = self.I(ta.sma, pd.Series(self.data.Close), length=self.fast_ma)
        self.sma2 = self.I(ta.sma, pd.Series(self.data.Close), length=self.slow_ma)
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), length=14)
    def next(self):
        if crossover(self.sma1, self.sma2) and self.rsi < self.rsi_limit:
            self.buy(size=0.99)
        elif crossover(self.sma2, self.sma1):
            self.position.close()

# --- 5. LOGIC: PROCESS STOCK (Connects to trading_bot.py) ---
def process_single_ticker(ticker, data_chunk_ignored, wallet, fast_ignored, slow_ignored, rsi_ignored):
    """
    Now we just ask the 'trading_bot' file to do the work.
    We ignore the sliders for now to enforce the 'Strict' rules in the bot.
    """
    try:
        # We pass the Ticker and Wallet Size to the bot
        # The bot downloads its own fresh data to be safe
        result = bot.analyze_ticker_precision(ticker, wallet)
        return result
    except Exception as e:
        return None

# --- 6. SESSION STATE ---
if 'page' not in st.session_state: st.session_state.page = "scanner"
if 'selected_ticker' not in st.session_state: st.session_state.selected_ticker = None

def set_ticker(ticker):
    st.session_state.selected_ticker = ticker
    st.session_state.page = "details"

def go_home():
    st.session_state.page = "scanner"
    st.session_state.selected_ticker = None

# --- 7. SIDEBAR ---
with st.sidebar:
    st.header("🦅 Precision Scanner")
    
    region = st.selectbox("1. Select Market", ["🇮🇳 India (NSE)", "🇺🇸 USA (S&P 500)"])
    wallet = st.number_input("2. Wallet Size (Max Stock Price)", value=100, step=50)
    batch_size = st.slider("Sample Size", 50, 300, 100)
    
    st.divider()
    st.caption("Strategy Settings (Strict)")
    sma_fast = st.slider("Fast Trend", 5, 50, 9)
    sma_slow = st.slider("Slow Trend", 20, 200, 21)
    # Default RSI lowered to 60 for higher safety
    rsi_limit = st.slider("RSI Limit", 40, 80, 60)
    
    if st.button("🔎 START PRECISION SCAN", type="primary"):
        st.session_state.scan_requested = True

# --- 8. PAGE 1: SCANNER ---
if st.session_state.page == "scanner":
    st.title(f"Live Scanner: {region}")
    
    if st.session_state.get('scan_requested', False):
        
        with st.spinner("Fetching Official Market List..."):
            full_list = fetch_realtime_symbols(region)
            
        if not full_list:
            st.error("Connection Error. Could not fetch list.")
        else:
            # Shuffle to find new opportunities
            random.shuffle(full_list)
            scan_list = full_list[:batch_size]
            
            st.info(f"Analyzing {len(scan_list)} stocks with high-precision filters...")
            
            # Batch Download (Slower but accurate)
            results = []
            progress = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Downloading High-Res Data..."):
                full_data = yf.download(scan_list, period="6mo", group_by='ticker', progress=False)
            
            total = len(scan_list)
            for i, ticker in enumerate(scan_list):
                try:
                    if len(scan_list) > 1:
                        if ticker in full_data.columns.levels[0]:
                            stock_df = full_data[ticker]
                        else: continue
                    else: stock_df = full_data
                    
                    data = process_single_ticker(ticker, stock_df, wallet, sma_fast, sma_slow, rsi_limit)
                    if data: results.append(data)
                    
                    status_text.text(f"Checking {i+1}/{total}: {ticker}...")
                    progress.progress((i+1)/total)
                except: pass
            
            progress.empty()
            status_text.empty()

            if not results:
                st.warning("😕 No stocks found. This is normal for 'High Precision' mode. Try increasing Wallet size or RSI limit.")
            else:
                results.sort(key=lambda x: x['Status'], reverse=True)
                st.success(f"🎉 Found {len(results)} high-quality trades!")
                
                for i in range(0, len(results), 3):
                    cols = st.columns(3)
                    for j in range(3):
                        if i + j < len(results):
                            item = results[i+j]
                            with cols[j]:
                                with st.container(border=True):
                                    c1, c2 = st.columns([2, 1])
                                    c1.metric(item['Ticker'], f"₹{item['Price']:.2f}")
                                    c2.markdown(f"**{item['Status']}**")
                                    
                                    chart = make_sparkline(item['Chart'], item['Color'])
                                    st.altair_chart(chart, use_container_width=True)
                                    
                                    st.caption(f"RSI: {item['RSI']:.1f} | Buy: {item['Shares']} Shares")
                                    st.button(f"Analyze {item['Ticker']}", key=f"btn_{item['Ticker']}", on_click=set_ticker, args=(item['Ticker'],))

# --- 9. PAGE 2: DETAILS ---
elif st.session_state.page == "details":
    ticker = st.session_state.selected_ticker
    st.button("← Back to Results", on_click=go_home)
    st.title(f"Deep Dive: {ticker}")
    
    with st.spinner(f"Simulating Strategy..."):
        try:
            df = yf.download(ticker, period="2y", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            
            if df.empty: st.error("No Data.")
            else:
                TrendStrategy.fast_ma = sma_fast
                TrendStrategy.slow_ma = sma_slow
                TrendStrategy.rsi_limit = rsi_limit
                # Simulate with 10x wallet to show potential return
                bt = Backtest(df, TrendStrategy, cash=wallet*10, commission=.002)
                stats = bt.run()
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Net Profit", f"{stats['Return [%]']:.2f}%", delta=f"{stats['Return [%]']:.2f}%")
                m2.metric("Win Rate", f"{stats['Win Rate [%]']:.2f}%")
                m3.metric("Final Equity", f"₹{stats['Equity Final [$]']:,.2f}")
                m4.metric("Max Drawdown", f"{stats['Max. Drawdown [%]']:.2f}%")
                
                st.subheader("Trade Visualization")
                bt.plot(open_browser=False, filename='plot.html')
                with open('plot.html', 'r', encoding='utf-8') as f:
                    components.html(f.read(), height=600, scrolling=True)
        except Exception as e: st.error(str(e))