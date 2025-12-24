import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --- CONFIGURATION ---
class BotConfig:
    SMA_FAST = 9
    SMA_SLOW = 21
    RSI_MIN = 30
    RSI_MAX = 75       # Stricter Max (Avoid buying tops)
    ADX_THRESHOLD = 20 # Used only for Strong Buy confirmation
    VOLUME_THRESHOLD = 1.0

def fetch_data(ticker, period="1y"):
    try:
        # Download data
        df = yf.download(ticker, period=period, progress=False)
        
        # Fix Multi-Level Columns (Common yfinance bug)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Basic cleaning
        df = df.dropna()
        
        # Need enough data for ML training (at least 60 bars)
        if len(df) < 60: return None
        return df
    except: return None

def calculate_indicators(df):
    df = df.copy()
    
    # 1. Trend Indicators
    df['SMA_Fast'] = ta.sma(df['Close'], length=BotConfig.SMA_FAST)
    df['SMA_Slow'] = ta.sma(df['Close'], length=BotConfig.SMA_SLOW)
    
    # 2. Momentum Indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    # NEW: RSI Slope (Is momentum gaining speed? Positive = Yes)
    df['RSI_Slope'] = df['RSI'].diff(3) 
    
    # 3. Strength Indicators
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx['ADX_14']
    
    # 4. Volume Indicators
    df['Vol_SMA'] = ta.sma(df['Volume'], length=20)
    
    # 5. Volatility (Bollinger & ATR)
    bb = ta.bbands(df['Close'], length=20)
    df['BB_Upper'] = bb['BBU_20_2.0']
    df['BB_Lower'] = bb['BBL_20_2.0']
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # --- ML FEATURES (Normalized inputs for the Brain) ---
    
    # Distance from Slow SMA (Are we overextended?)
    df['SMA_Dist'] = (df['Close'] - df['SMA_Slow']) / df['SMA_Slow']
    
    # Relative Volume (Are people trading this?)
    df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA']
    
    # Normalized RSI
    df['RSI_Norm'] = df['RSI'] / 100
    
    # Bollinger Position (0 = Bottom band, 1 = Top band)
    df['BB_Pos'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    return df

def train_and_predict(df):
    try:
        data = df.copy().dropna()
        
        # --- IMPROVED TARGET ---
        # Old: Did it go up next day? (Noisy)
        # New: Did it go up > 1% in the next 2 days? (Quality Move)
        future_return = data['Close'].shift(-2) / data['Close'] - 1
        data['Target'] = (future_return > 0.01).astype(int)
        
        # Updated Feature List
        features = ['SMA_Dist', 'Vol_Ratio', 'RSI_Norm', 'ADX', 'RSI_Slope', 'BB_Pos']
        data = data.dropna()
        
        if len(data) < 50: return 50.0

        # Train on past data
        # We drop the last 2 rows because they don't have a "Future Target" yet
        X = data[features][:-2] 
        y = data['Target'][:-2]
        
        # Predict on TODAY'S live data
        X_live = data[features].tail(1)
        
        # Random Forest (200 Trees for stability)
        model = RandomForestClassifier(n_estimators=200, min_samples_split=10, random_state=42)
        model.fit(X, y)
        
        # Get Probability of Up Move
        probability = model.predict_proba(X_live)[0][1]
        
        return round(probability * 100, 1)
    except: return 50.0

def analyze_ticker_precision(ticker, wallet_size):
    df = fetch_data(ticker)
    if df is None: return None, "❌ Data Error"
    
    df = calculate_indicators(df)
    last = df.iloc[-1]
    price = last['Close']
    
    # --- FILTERS ---
    if price > wallet_size: return None, "Expensive"
    if last['Volume'] < 1000: return None, "Low Volume"

    # Basic Check: Is it in an uptrend?
    uptrend = last['SMA_Fast'] > last['SMA_Slow']
    safe_rsi = last['RSI'] < BotConfig.RSI_MAX
    
    if uptrend and safe_rsi:
        
        # Run the upgraded AI Brain
        ai_score = train_and_predict(df)
        
        # --- BADGE LOGIC ---
        status = "✅ UPTREND"
        
        # STRICT Rules for "Strong Buy"
        # 1. AI Score must be very high (>75%)
        # 2. Trend must be proven (ADX > 20)
        # 3. Momentum must be increasing (RSI Slope > 0)
        if (ai_score > 75) and (last['ADX'] > 20) and (last['RSI_Slope'] > 0):
            status = "🔥 STRONG BUY"
        elif ai_score > 60:
            status = "🦅 AI CONFIRMED"

        sparkline = df['Close'].tail(30).reset_index(drop=True)
        color = "#00FF00" if sparkline.iloc[-1] >= sparkline.iloc[0] else "#FF4B4B"

        # Trade Plan
        atr = last['ATR']
        stop_loss = price - (2 * atr)
        take_profit = price + (4 * atr) # Aim for 2:1 Reward Ratio

        result = {
            "Ticker": ticker,
            "Price": price,
            "RSI": last['RSI'],
            "AI_Score": ai_score,
            "Status": status,
            "Chart": sparkline,
            "Color": color,
            "Shares": int(wallet_size // price),
            "Stop_Loss": stop_loss,
            "Take_Profit": take_profit
        }
        return result, "OK"
        
    return None, "Weak Trend/RSI"