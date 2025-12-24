import streamlit as st
import pandas as pd
import requests
import io
import time
import altair as alt  # Required for the custom charts
import trading_bot as bot

# --- CONFIG ---
st.set_page_config(layout="wide", page_title="AI Infinity Scanner", page_icon="🦅")

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
if 'is_scanning' not in st.session_state: st.session_state.is_scanning = False
if 'scanned_results' not in st.session_state: st.session_state.scanned_results = []
if 'scan_logs' not in st.session_state: st.session_state.scan_logs = []
if 'scan_index' not in st.session_state: st.session_state.scan_index = 0

# --- HELPER: ZOOMED CHART (The Fix) ---
def make_sparkline(data_series, color_hex):
    # Convert data to a proper format for Altair
    df = data_series.reset_index(drop=True).to_frame(name='price')
    df['index'] = df.index
    
    # Create the chart
    chart = alt.Chart(df).mark_line(
        color=color_hex, 
        strokeWidth=2
    ).encode(
        # X-Axis: Hidden to keep it clean
        x=alt.X('index', axis=None),
        
        # Y-Axis: scale=alt.Scale(zero=False) forces it to ZOOM IN
        y=alt.Y('price', 
                scale=alt.Scale(zero=False, padding=1), 
                axis=alt.Axis(labels=True, title=None, tickCount=3) # Show prices on the left
        ),
        tooltip=['price']
    ).properties(
        height=60, 
        width='container'
    ).configure_axis(
        grid=False # Remove grid lines for a cleaner look
    ).configure_view(
        strokeWidth=0 # Remove border
    )
    
    return chart

@st.cache_data(ttl=3600)
def fetch_realtime_symbols(region):
    try:
        if region == "🇮🇳 India (NSE)":
            url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
            headers = {"User-Agent": "Mozilla/5.0"}
            s = requests.get(url, headers=headers).content
            df = pd.read_csv(io.StringIO(s.decode('utf-8')))
            return [f"{x}.NS" for x in df['SYMBOL'].tolist()]
    except: return []
    return []

# --- SIDEBAR ---
with st.sidebar:
    st.header("🦅 Infinity Scanner")
    region = st.selectbox("Market", ["🇮🇳 India (NSE)"])
    wallet = st.number_input("Max Price (₹)", value=2000, step=100)
    
    col1, col2 = st.columns(2)
    if col1.button("▶ START", type="primary"):
        st.session_state.is_scanning = True
        st.session_state.scan_logs = [] 
    if col2.button("⏹ STOP"):
        st.session_state.is_scanning = False
        
    st.divider()
    st.subheader("📝 Live Logs")
    for log in st.session_state.scan_logs[-15:]:
        st.text(log)

# --- MAIN PAGE ---
st.title("Live Market Feed")

grid_placeholder = st.empty()
status_placeholder = st.empty()

if st.session_state.is_scanning:
    full_list = fetch_realtime_symbols(region)
    
    while st.session_state.is_scanning:
        if st.session_state.scan_index >= len(full_list):
            st.session_state.scan_index = 0
            
        ticker = full_list[st.session_state.scan_index]
        
        # Analyze
        result, message = bot.analyze_ticker_precision(ticker, wallet)
        
        # Log
        st.session_state.scan_logs.append(f"{ticker}: {message}")
        
        # Add to Results
        if result:
            if not any(d['Ticker'] == ticker for d in st.session_state.scanned_results):
                st.session_state.scanned_results.insert(0, result) 
        
        # Update Status
        status_placeholder.info(f"Scanning: {ticker} ... {message}")
        
        # Render Grid
        if st.session_state.scanned_results:
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
                                    
                                    score_color = "green" if item['AI_Score'] > 60 else "orange"
                                    c2.markdown(f"**{item['Status']}**")
                                    c2.markdown(f":{score_color}[AI: {item['AI_Score']}%]")
                                    
                                    # --- THE FIX IS HERE ---
                                    # Use the custom Altair chart instead of st.line_chart
                                    chart = make_sparkline(item['Chart'], item['Color'])
                                    st.altair_chart(chart, use_container_width=True)
                                    
                                    st.caption(f"Stop: {item['Stop_Loss']:.1f} | Target: {item['Take_Profit']:.1f}")
                                    st.button(f"Analyze {item['Ticker']}", key=f"btn_{item['Ticker']}_{i}_{j}") # Unique key

        st.session_state.scan_index += 1
        time.sleep(0.01)