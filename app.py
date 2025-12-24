import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import streamlit.components.v1 as components
import requests
import io
import math

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Penny Stock Sniper (Live)")

st.title("🎯 Penny Stock Sniper (Live Menu)")
st.markdown("**Status:** Dropdown now shows LIVE prices and daily trends.")

# --- 2. STOCK DATA & LIVE PRICES ---
@st.cache_data(ttl=300) # Re-fetch live prices every 5 minutes
def get_live_prices(ticker_list):
    """
    Downloads live data for ALL tickers in the list at once
    to create a formatted 'Menu String' for each stock.
    """
    price_map = {}
    
    # Only fetch if list is small (to prevent crashing on 2000+ stocks)
    if len(ticker_list) > 100:
        return {}

    try:
        # Download 1 day of data for all tickers in one shot
        data = yf.download(ticker_list, period="5d", group_by='ticker', progress=False)
        
        for ticker in ticker_list:
            try:
                # Handle single-level vs multi-level columns
                if len(ticker_list) == 1:
                    stock_data = data
                else:
                    stock_data = data[ticker]
                
                # Get last two rows to calculate change
                if not stock_data.empty:
                    last_price = stock_data['Close'].iloc[-1]
                    prev_price = stock_data['Close'].iloc[-2]
                    change = ((last_price - prev_price) / prev_price) * 100
                    
                    # Formatting: Green for Up, Red for Down
                    symbol = "🟢" if change >= 0 else "🔴"
                    
                    # Create the display string: "IDEA.NS  ₹14.50 🔴 -1.2%"
                    price_map[ticker] = f"{ticker}   ₹{last_price:.2f} {symbol} {change:+.1f}%"
            except:
                price_map[ticker] = ticker # Fallback if data fails
                
    except Exception as e:
        print(f"Error fetching batch prices: {e}")
        
    return price_map

@st.cache_data(ttl=24*3600)
def get_stock_lists():
    # List 1: Penny Stocks (₹20 - ₹60)
    popular_penny = [
        "IDEA.NS", "YESBANK.NS", "SUZLON.NS", "UCOBANK.NS", "IOB.NS", 
        "MAHABANK.NS", "CENTRALBK.NS", "NHPC.NS", "SJVN.NS", "RENUKA.NS", 
        "TRIDENT.NS", "HCC.NS", "SOUTHBANK.NS", "MOREPENLAB.NS", "PNB.NS",
        "ZOMATO.NS", "IDFCFIRSTB.NS", "GMRINFRA.NS", "BHEL.NS", "IRFC.NS"
    ]
    
    # List 2: SUPER CHEAP (< ₹20)
    super_cheap = [
        "GTLINFRA.NS", "RPOWER.NS", "JPPOWER.NS", "VIKASECO.NS", 
        "AKSHOPTFBR.NS", "FCSSOFT.NS", "BHANDARI.NS", "MPSINFOTEC.NS",
        "3IINFOLTD.NS", "ALANKIT.NS", "RTNPOWER.NS", "ORIENTGREEN.NS",
        "INVENTURE.NS", "SADBHAV.NS", "MTNL.NS", "IFCI.NS"
    ]
    
    # List 3: Full NSE List
    try:
        url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        df = pd.read_csv(io.StringIO(response.text))
        full_list = [f"{sym}.NS" for sym in df['SYMBOL'].tolist()]
        full_list.sort()
    except:
        full_list = []

    return popular_penny, super_cheap, full_list

# Load Lists
with st.spinner("Connecting to NSE..."):
    pop_list, cheap_list, full_list = get_stock_lists()

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("1. Asset Selection")

filter_type = st.sidebar.radio(
    "Select Category:", 
    ["🔥 Popular Penny", "💰 Super Cheap (< ₹20)", "🌏 Search All"]
)

# Logic to switch lists and fetch prices
if filter_type == "Popular Penny":
    current_list = pop_list
    with st.spinner("Fetching Live Prices..."):
        price_display_map = get_live_prices(pop_list)
        
elif filter_type == "Super Cheap (< ₹20)":
    current_list = cheap_list
    with st.spinner("Scanning Bargain Bin..."):
        price_display_map = get_live_prices(cheap_list)
        
else:
    current_list = full_list
    price_display_map = {} # Too many to fetch live

# --- THE MAGIC DROPDOWN ---
# We use 'format_func' to swap the boring name for the Rich Text name
selected_ticker = st.sidebar.selectbox(
    "Choose Stock", 
    current_list,
    format_func=lambda x: price_display_map.get(x, x) # Shows price if available
)

st.sidebar.header("2. Timeframe")
period_option = st.sidebar.select_slider(
    "Test Duration:",
    options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    value="6mo"
)

st.sidebar.header("3. Setup")
cash = st.sidebar.number_input("Wallet Balance (₹)", 100, 10000000, 2000, step=100)
sma_fast = st.sidebar.slider("Fast Trend", 3, 50, 9)
sma_slow = st.sidebar.slider("Slow Trend", 10, 100, 21)
rsi_limit = st.sidebar.slider("RSI Limit", 30, 90, 70)

# --- 4. STRATEGY ---
class BulkBuyStrategy(Strategy):
    def init(self):
        self.sma1 = self.I(ta.sma, pd.Series(self.data.Close), length=sma_fast)
        self.sma2 = self.I(ta.sma, pd.Series(self.data.Close), length=sma_slow)
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), length=14)

    def next(self):
        if crossover(self.sma1, self.sma2) and self.rsi < rsi_limit:
            self.buy(size=0.99)
        elif crossover(self.sma2, self.sma1):
            self.position.close()

def get_market_data(ticker, period):
    interval = "1h" if period in ["1mo", "3mo"] else "1d"
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

# --- 5. EXECUTION ---
if st.button("🚀 Run Test"):
    try:
        with st.spinner(f"Scanning {selected_ticker}..."):
            df = get_market_data(selected_ticker, period_option)
            
            if df.empty:
                st.error("❌ No data found.")
            else:
                current_price = df['Close'].iloc[-1]
                max_shares = math.floor(cash / current_price)
                
                st.subheader(f"Analysis: {selected_ticker}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Price", f"₹{current_price:,.2f}")
                
                if max_shares < 1:
                    c2.error("0 Shares")
                    st.error(f"❌ You need ₹{current_price:.2f} minimum.")
                else:
                    c2.success(f"{max_shares} Shares potential")
                    c3.metric("Wallet", f"₹{cash:,.2f}")

                    bt = Backtest(df, BulkBuyStrategy, cash=cash, commission=.002)
                    stats = bt.run()
                    
                    st.markdown("---")
                    m1, m2, m3, m4 = st.columns(4)
                    
                    profit = stats['Return [%]']
                    m1.metric("Profit/Loss", f"{profit:.2f}%", delta=f"{profit:.2f}%")
                    
                    win_rate = stats['Win Rate [%]']
                    win_disp = "0% (No Trades)" if pd.isna(win_rate) else f"{win_rate:.2f}%"
                    m2.metric("Win Rate", win_disp)
                    
                    m3.metric("Final Value", f"₹{stats['Equity Final [$]']:,.2f}")
                    m4.metric("Risk", f"{stats['Max. Drawdown [%]']:.2f}%")

                    st.subheader(f"Performance ({period_option})")
                    bt.plot(open_browser=False, filename='plot.html')
                    with open('plot.html', 'r', encoding='utf-8') as f:
                        components.html(f.read(), height=700, scrolling=True)

    except Exception as e:
        st.error(f"Error: {e}")