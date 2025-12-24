import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import concurrent.futures
import streamlit.components.v1 as components
import altair as alt
import requests
import io
import random
import time
import trading_bot as bot # Your strict logic file

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="AI Infinity Scanner", page_icon="🦅")

st.markdown("""
<style>
    /* Compact Cards */
    div.stButton > button {
        width: 100%;
        background-color: #1E1E1E;
        border: 1px solid #444;
        color: white;
        font-size: 12px;
    }
    div.stButton > button:hover {
        border-color: #00FF00;
        color: #00FF00;
    }
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        border: 1px solid #333;
        padding: 10px;
        border-radius: 8px;
        background-color: #0E1117;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'page' not in st.session_state: st.session_state.page = "scanner"
if 'selected_ticker' not in st.session_state: st.session_state.selected_ticker = None
if 'scanned_results' not in st.session_state: st.session_state.scanned_results = []
if 'is_scanning' not in st.session_state: st.session_state.is_scanning = False
if 'scan_index' not in st.session_state: st.session_state.scan_index = 0

# --- 3. HELPER FUNCTIONS ---
@st.cache_data(ttl=3600)
def fetch_realtime_symbols(region):
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
    except: return []
    return []

def make_sparkline(data_series, color_hex):
    df = data_series.reset_index(drop=True).to_frame(name='price')
    df['index'] = df.index
    chart = alt.Chart(df).mark_line(color=color_hex, strokeWidth=2).encode(
        x=alt.X('index', axis=None),
        y=alt.Y('price', scale=alt.Scale(zero=False, padding=1), axis=alt.Axis(labels=True, tickCount=2)),
        tooltip=['price']
    ).properties(height=60, width='container').configure_axis(grid=False).configure_view(strokeWidth=0)
    return chart

# --- 4. NAVIGATION ---
def set_ticker(ticker):
    st.session_state.selected_ticker = ticker
    st.session_state.is_scanning = False # Stop scanning when viewing details
    st.session_state.page = "details"

def go_home():
    st.session_state.page = "scanner"

def start_scan():
    st.session_state.is_scanning = True
    st.session_state.scanned_results = [] # Clear old results
    st.session_state.scan_index = 0

def stop_scan():
    st.session_state.is_scanning = False

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("🦅 Infinity Scanner")
    region = st.selectbox("Market", ["🇮🇳 India (NSE)", "🇺🇸 USA (S&P 500)"])
    wallet = st.number_input("Max Price (₹/$)", value=2000, step=100)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ START LIVE", type="primary"):
            start_scan()
    with col2:
        if st.button("⏹ STOP"):
            stop_scan()
            
    st.divider()
    st.caption(f"Status: {'🟢 Running' if st.session_state.is_scanning else '🔴 Stopped'}")
    st.caption(f"Found: {len(st.session_state.scanned_results)} gems")

# --- 6. PAGE 1: INFINITY SCANNER ---
if st.session_state.page == "scanner":
    st.title("Live Market Feed")
    
    # Placeholder for the Live Status Bar
    status_placeholder = st.empty()
    
    # Placeholder for the Grid (We update this constantly)
    grid_placeholder = st.empty()

    # --- THE INFINITY LOOP ---
    if st.session_state.is_scanning:
        
        # 1. Get Full List (Once)
        full_list = fetch_realtime_symbols(region)
        
        # 2. Main Loop
        while st.session_state.is_scanning:
            
            # A. Determine Batch (Chunks of 20 stocks)
            start_i = st.session_state.scan_index
            end_i = start_i + 20
            
            # Reset if we reached the end
            if start_i >= len(full_list):
                st.session_state.scan_index = 0
                continue
                
            batch = full_list[start_i:end_i]
            
            # B. Update Status UI
            status_placeholder.markdown(f"""
                <div style="padding: 10px; background-color: #0E1117; border: 1px solid #333; border-radius: 5px; color: #00FF00;">
                    ⚡ Scanning stocks {start_i} to {end_i} of {len(full_list)}...
                </div>
            """, unsafe_allow_html=True)
            
            # C. Download Data (Batch)
            try:
                # We assume the bot file is correct. We fetch strict logic.
                full_data = yf.download(batch, period="1y", group_by='ticker', progress=False)
                
                # D. Analyze Batch
                new_finds = []
                for ticker in batch:
                    try:
                        if len(batch) > 1:
                            if ticker in full_data.columns.levels[0]:
                                df = full_data[ticker]
                            else: continue
                        else: df = full_data
                        
                        # Use your EXISTING bot logic
                        # We reconstruct the 'df' to match what the bot expects
                        # Or simply call the logic function if you refactored properly.
                        # For speed, we call the bot's processing function logic here:
                        
                        result = bot.analyze_ticker_precision(ticker, wallet) 
                        # Note: We are re-fetching inside the bot which is safer for accuracy but slower.
                        # To make it fast, we should refactor bot to accept DF. 
                        # For now, let's trust the bot's internal fetch for "High Precision".
                        
                        if result:
                            # Avoid duplicates
                            if not any(d['Ticker'] == ticker for d in st.session_state.scanned_results):
                                new_finds.append(result)
                    except: pass
                
                # E. Update Session State
                if new_finds:
                    st.session_state.scanned_results = new_finds + st.session_state.scanned_results
                    
                # F. Render Grid (Only if we have results)
                if st.session_state.scanned_results:
                    with grid_placeholder.container():
                        # Loop through results
                        results = st.session_state.scanned_results
                        for i in range(0, len(results), 3):
                            cols = st.columns(3)
                            for j in range(3):
                                if i + j < len(results):
                                    item = results[i+j]
                                    with cols[j]:
                                        with st.container(border=True):
                                            c1, c2 = st.columns([2, 1])
                                            c1.metric(item['Ticker'], f"₹{item['Price']:.2f}")
                                            
                                            score_color = "green" if item['AI_Score'] > 60 else "orange"
                                            c2.markdown(f"**{item['Status']}**")
                                            c2.markdown(f":{score_color}[AI: {item['AI_Score']}%]")
                                            
                                            chart = make_sparkline(item['Chart'], item['Color'])
                                            st.altair_chart(chart, use_container_width=True)
                                            
                                            st.markdown(f"""
                                            <div style="font-size: 12px; color: #888; margin-bottom: 5px;">
                                                🛑 Stop: <span style="color: #FF4B4B;">{item['Stop_Loss']:.2f}</span> | 
                                                🎯 Target: <span style="color: #00FF00;">{item['Take_Profit']:.2f}</span>
                                            </div>
                                            """, unsafe_allow_html=True)
                                            
                                            st.button(f"Analyze {item['Ticker']}", key=f"btn_{item['Ticker']}_{random.randint(0,100000)}", on_click=set_ticker, args=(item['Ticker'],))

            except Exception as e:
                pass

            # G. Increment Index
            st.session_state.scan_index += 20
            
            # Tiny sleep to let Streamlit UI breathe
            time.sleep(0.1)
            
    # Show results even if stopped
    elif st.session_state.scanned_results:
        with grid_placeholder.container():
            results = st.session_state.scanned_results
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
                                st.altair_chart(make_sparkline(item['Chart'], item['Color']), use_container_width=True)
                                st.button(f"Analyze {item['Ticker']}", key=f"btn_static_{item['Ticker']}", on_click=set_ticker, args=(item['Ticker'],))

# --- 7. PAGE 2: DETAILS ---
elif st.session_state.page == "details":
    # (Reuse your existing details logic here)
    # Ideally, import this from a separate view file to keep code clean
    # For now, a simple placeholder to prove navigation works:
    st.button("← Back to Live Feed", on_click=go_home)
    st.title(f"Deep Analysis: {st.session_state.selected_ticker}")
    # ... Your existing Detail View code goes here ...