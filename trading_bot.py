import yfinance as yf
import pandas as pd
import pandas_ta as ta

# --- CONFIGURATION (The Brain) ---
class BotConfig:
    SMA_FAST = 9
    SMA_SLOW = 21
    RSI_MIN = 30
    RSI_MAX = 70
    ADX_THRESHOLD = 20  # Trend Strength (Below 20 = Weak/Choppy)
    VOLUME_THRESHOLD = 1.2  # Volume must be 20% higher than average

def fetch_data(ticker, period="6mo"):
    """
    Downloads data specifically formatted for Technical Analysis.
    """
    try:
        df = yf.download(ticker, period=period, progress=False)
        
        # Handle MultiIndex (yfinance bug fix)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Clean Data
        df = df.dropna()
        
        # Ensure we have enough data for indicators (need at least 21 days)
        if len(df) < 50: 
            return None
            
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def calculate_indicators(df):
    """
    Adds Technical Indicators to the DataFrame.
    """
    # 1. Moving Averages (Trend)
    df['SMA_Fast'] = ta.sma(df['Close'], length=BotConfig.SMA_FAST)
    df['SMA_Slow'] = ta.sma(df['Close'], length=BotConfig.SMA_SLOW)
    
    # 2. RSI (Momentum)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # 3. ADX (Trend Strength) - Crucial for accuracy!
    # ADX tells us if the market is actually trending or just moving sideways.
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx['ADX_14']
    
    # 4. Volume SMA (Conviction)
    df['Vol_SMA'] = ta.sma(df['Volume'], length=20)
    
    return df

def analyze_ticker_precision(ticker, wallet_size):
    """
    The 'Sniper' function. Returns data ONLY if strict criteria are met.
    """
    # 1. Get Data
    df = fetch_data(ticker)
    if df is None: return None
    
    # 2. Calculate Indicators
    df = calculate_indicators(df)
    
    # 3. Get Latest Values (The "Now" Candle)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    price = last['Close']
    
    # --- FILTER 1: Wallet ---
    if price > wallet_size: return None
    
    # --- FILTER 2: Dead Stock Check ---
    # If volume is near zero, nobody is trading it. Dangerous.
    if last['Volume'] < 500: return None

    # --- THE SNIPER LOGIC (4 Checks) ---
    
    # Check A: Golden Cross / Uptrend
    # We want the Fast SMA to be ABOVE Slow SMA
    is_uptrend = last['SMA_Fast'] > last['SMA_Slow']
    
    # Check B: RSI Value
    # Not too expensive (>70) and not crashing (<30)
    is_fair_price = (last['RSI'] > BotConfig.RSI_MIN) and (last['RSI'] < BotConfig.RSI_MAX)
    
    # Check C: ADX Strength (The Accuracy Booster)
    # ADX > 20 means there is a REAL trend. ADX < 20 means noise.
    is_strong_trend = last['ADX'] > BotConfig.ADX_THRESHOLD
    
    # Check D: Volume Spike (The Whales)
    # Is current volume higher than the 20-day average?
    # This confirms that "Smart Money" is interested today.
    is_high_volume = last['Volume'] > (last['Vol_SMA'] * BotConfig.VOLUME_THRESHOLD)

    # --- FINAL VERDICT ---
    if is_uptrend and is_fair_price and is_strong_trend:
        
        # Determine Signal Strength
        strength = "✅ GOOD"
        color = "#00FF00" # Green
        
        # If Volume is HUGE, it's a "STRONG" buy
        if is_high_volume:
            strength = "🔥 STRONG (High Volume)"
            
        # Determine Chart Color (Profit/Loss over last 30 days)
        sparkline = df['Close'].tail(30).reset_index(drop=True)
        chart_color = "#00FF00" if sparkline.iloc[-1] >= sparkline.iloc[0] else "#FF4B4B"

        return {
            "Ticker": ticker,
            "Price": price,
            "RSI": last['RSI'],
            "ADX": last['ADX'],
            "Status": strength,
            "Chart": sparkline,
            "Color": chart_color,
            "Shares": int(wallet_size // price)
        }
        
    return None