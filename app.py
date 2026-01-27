import streamlit as st
import pandas as pd
import requests
import io
import time
import altair as alt
import trading_bot as bot
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import streamlit.components.v1 as components

# --- CONFIG ---
st.set_page_config(layout="wide", page_title="Stock Algo", page_icon="📈")

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
</style>
""", unsafe_allow_html=True)

# --- STATE ---
if 'page' not in st.session_state: st.session_state.page = "scanner"
if 'selected_ticker' not in st.session_state: st.session_state.selected_ticker = None
if 'is_scanning' not in st.session_state: st.session_state.is_scanning = False
if 'scanned_results' not in st.session_state: st.session_state.scanned_results = []
if 'scan_logs' not in st.session_state: st.session_state.scan_logs = []
if 'scan_index' not in st.session_state: st.session_state.scan_index = 0
if 'currency_symbol' not in st.session_state: st.session_state.currency_symbol = "₹"
# New counters to track progress
if 'total_scanned' not in st.session_state: st.session_state.total_scanned = 0
if 'total_rejected' not in st.session_state: st.session_state.total_rejected = 0

# --- NAVIGATION ---
def go_to_details(ticker):
    st.session_state.selected_ticker = ticker
    st.session_state.is_scanning = False 
    st.session_state.page = "details"

def go_home():
    st.session_state.page = "scanner"

# --- STRATEGY ---
class TrendStrategy(Strategy):
    fast_ma = 9
    slow_ma = 21
    def init(self):
        self.sma1 = self.I(lambda x: pd.Series(x).rolling(self.fast_ma).mean(), self.data.Close)
        self.sma2 = self.I(lambda x: pd.Series(x).rolling(self.slow_ma).mean(), self.data.Close)
    def next(self):
        if crossover(self.sma1, self.sma2): self.buy()
        elif crossover(self.sma2, self.sma1): self.position.close()

# --- HELPERS ---
def make_sparkline(data_series, color_hex):
    df = data_series.reset_index(drop=True).to_frame(name='price')
    df['index'] = df.index
    chart = alt.Chart(df).mark_line(color=color_hex, strokeWidth=2).encode(
        x=alt.X('index', axis=None),
        y=alt.Y('price', scale=alt.Scale(zero=False, padding=1), axis=alt.Axis(labels=True, title=None, tickCount=3)),
        tooltip=['price']
    ).properties(height=60, width='container').configure_axis(grid=False).configure_view(strokeWidth=0)
    return chart

@st.cache_data(ttl=3600)
def fetch_realtime_symbols(region):
    try:
        # 1. INDIA (NSE)
        if region == "🇮🇳 India (NSE)":
            url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
            headers = {"User-Agent": "Mozilla/5.0"}
            s = requests.get(url, headers=headers).content
            df = pd.read_csv(io.StringIO(s.decode('utf-8')))
            return [f"{x}.NS" for x in df['SYMBOL'].tolist()]

        # 2. USA (S&P 500)
        elif region == "🇺🇸 USA (S&P 500)":
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            df = tables[0]
            symbols = df['Symbol'].str.replace('.', '-', regex=False).tolist()
            return symbols

        # 3. UK (FTSE 100)
        elif region == "🇬🇧 UK (FTSE 100)":
            url = 'https://en.wikipedia.org/wiki/FTSE_100_Index'
            tables = pd.read_html(url)
            for table in tables:
                if 'Ticker' in table.columns:
                    return [f"{x}.L" for x in table['Ticker'].tolist()]
            return []
    except Exception as e:
        st.error(f"Error fetching symbols: {e}")
        return []
    return []

# --- PAGE 1: SCANNER ---
if st.session_state.page == "scanner":
    
    with st.sidebar:
        st.header("Scanner")
        
        region = st.selectbox("Market", ["🇮🇳 India (NSE)", "🇺🇸 USA (S&P 500)", "🇬🇧 UK (FTSE 100)"])
        
        if "India" in region: st.session_state.currency_symbol = "₹"
        elif "USA" in region: st.session_state.currency_symbol = "$"
        elif "UK" in region: st.session_state.currency_symbol = "£"
        
        wallet = st.number_input(f"Account Balance ({st.session_state.currency_symbol})", value=100000, step=1000)
        
        col1, col2 = st.columns(2)
        if col1.button("▶ START", type="primary"):
            st.session_state.is_scanning = True
            st.session_state.scan_logs = []
            st.session_state.total_scanned = 0 # Reset counters
            st.session_state.total_rejected = 0
            st.rerun()
        if col2.button("⏹ STOP"):
            st.session_state.is_scanning = False
            st.rerun()
            
        st.divider()
        st.caption(f"📊 Scanned: {st.session_state.total_scanned}")
        st.caption(f"❌ Rejected: {st.session_state.total_rejected}")
        st.divider()
        st.subheader("📝 Live Logs")
        for log in reversed(st.session_state.scan_logs[-10:]):
            st.text(log)

    st.title("Live Market Feed")

    # 1. RENDER RESULTS
    if st.session_state.scanned_results:
        results = st.session_state.scanned_results
        for i in range(0, len(results), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(results):
                    item = results[i+j]
                    with cols[j]:
                        with st.container(border=True):
                            c1, c2 = st.columns([2, 1])
                            curr = st.session_state.currency_symbol
                            c1.metric(item['Ticker'], f"{curr}{item['Price']:.2f}")
                            
                            score_color = "green" if item['AI_Score'] > 60 else "orange"
                            c2.markdown(f"**{item['Status']}**")
                            c2.markdown(f":{score_color}[AI: {item['AI_Score']}%]")
                            
                            chart = make_sparkline(item['Chart'], item['Color'])
                            st.altair_chart(chart, use_container_width=True)
                            
                            st.caption(f"Stop: {item['Stop_Loss']:.2f} | Target: {item['Take_Profit']:.2f}")
                            st.caption(f"Position: {item['Shares']} shares")
                            
                            if "DIAMOND" in item['Status']:
                                st.info("💎 DIAMOND SETUP")

                            st.button(f"Analyze {item['Ticker']}", 
                                      key=f"btn_{item['Ticker']}", 
                                      on_click=go_to_details, 
                                      args=(item['Ticker'],))

    # 2. SCANNING LOGIC (BATCH MODE)
    if st.session_state.is_scanning:
        full_list = fetch_realtime_symbols(region)
        
        if not full_list:
            st.error("Could not fetch symbols. Check connection.")
            st.session_state.is_scanning = False
        else:
            # --- BATCH SCANNING LOOP ---
            batch_size = 5 
            
            for _ in range(batch_size):
                if st.session_state.scan_index >= len(full_list): 
                    st.session_state.scan_index = 0
                
                ticker = full_list[st.session_state.scan_index]
                
                st.session_state.total_scanned += 1
                
                try:
                    result, message = bot.analyze_ticker_precision(ticker, wallet)
                    st.session_state.scan_logs.append(f"{ticker}: {message}")
                    
                    if result:
                        st.session_state.scanned_results = [r for r in st.session_state.scanned_results if r['Ticker'] != ticker]
                        st.session_state.scanned_results.append(result)
                        st.session_state.scanned_results.sort(key=lambda x: (x['Status'] == "🔥 STRONG BUY", x['AI_Score']), reverse=True)
                        st.toast(f"Found Gem: {ticker}!")
                    else:
                        st.session_state.total_rejected += 1
                        
                except Exception as e:
                    st.session_state.scan_logs.append(f"{ticker}: Error {str(e)}")
                
                st.session_state.scan_index += 1
            
            st.toast(f"Scanning... ({st.session_state.total_scanned} analyzed)")
            time.sleep(0.01)
            st.rerun()

# --- PAGE 2: DETAILS ---
elif st.session_state.page == "details":
    ticker = st.session_state.selected_ticker
    st.button("← Back to Feed", on_click=go_home)
    st.title(f"Deep Analysis: {ticker}")
    
    import yfinance as yf
    
    with st.spinner("Running 2-Year Backtest Simulation..."):
        try:
            df = yf.download(ticker, period="2y", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            
            bt = Backtest(df, TrendStrategy, cash=100000, commission=.002)
            stats = bt.run()
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Net Profit", f"{stats['Return [%]']:.2f}%", delta=f"{stats['Return [%]']:.2f}%")
            m2.metric("Win Rate", f"{stats['Win Rate [%]']:.2f}%")
            m3.metric("Trades", int(stats['# Trades']))
            m4.metric("Max Drawdown", f"{stats['Max. Drawdown [%]']:.2f}%")
            
            st.subheader("Interactive Strategy Chart")
            bt.plot(open_browser=False, filename='plot.html')
            with open('plot.html', 'r', encoding='utf-8') as f:
                components.html(f.read(), height=600, scrolling=True)
                
        except Exception as e:
            st.error(f"Analysis failed: {e}")