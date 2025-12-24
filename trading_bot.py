import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --- CONFIGURATION (Balanced) ---
class BotConfig:
    SMA_FAST = 9
    SMA_SLOW = 21
    RSI_MIN = 30
    RSI_MAX = 75       # Loosened from 70 to 75
    ADX_THRESHOLD = 15 # Loosened from 20 to 15 (Catches earlier trends)
    VOLUME_THRESHOLD = 1.0 # Loosened from 1.2 to 1.0 (Average volume is okay)

def fetch_data(ticker, period="1y"): # Reduced to 1y for speed
    try:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if len(df) < 50: return None
        return df
    except: return None

def calculate_indicators(df):
    df = df.copy()
    
    # Standard
    df['SMA_Fast'] = ta.sma(df['Close'], length=BotConfig.SMA_FAST)
    df['SMA_Slow'] = ta.sma(df['Close'], length=BotConfig.SMA_SLOW)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx['ADX_14']
    df['Vol_SMA'] = ta.sma(df['Volume'], length=20)
    
    # Advanced (MACD, BB, ATR)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    
    bb = ta.bbands(df['Close'], length=20)
    df['BB_Upper'] = bb['BBU_20_2.0']
    df['BB_Lower'] = bb['BBL_20_2.0']
    
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # ML Features
    df['SMA_Diff'] = (df['SMA_Fast'] - df['SMA_Slow']) / df['SMA_Slow']
    df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA']
    df['RSI_Norm'] = df['RSI'] / 100
    df['MACD_Diff'] = (df['MACD'] - df['MACD_Signal'])
    df['BB_Pos'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    return df

def train_and_predict(df):
    try:
        data = df.copy().dropna()
        # Target: Price Up next day
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        
        features = ['SMA_Diff', 'Vol_Ratio', 'RSI_Norm', 'ADX', 'MACD_Diff', 'BB_Pos']
        data = data.dropna()
        
        if len(data) < 50: return 50.0 # Not enough data to train

        X = data[features][:-1] 
        y = data['Target'][:-1]
        X_live = data[features].tail(1)
        
        model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        model.fit(X, y)
        probability = model.predict_proba(X_live)[0][1]
        
        return round(probability * 100, 1)
    except Exception as e:
        print(f"ML Error: {e}")
        return 50.0

def analyze_ticker_precision(ticker, wallet_size):
    # 1. Get Data
    df = fetch_data(ticker)
    if df is None: 
        # print(f"{ticker}: No Data")
        return None
    
    # 2. Add Indicators
    df = calculate_indicators(df)
    last = df.iloc[-1]
    price = last['Close']
    
    # --- FILTERS ---
    
    # Filter 1: Price
    if price > wallet_size: 
        return None
        
    # Filter 2: Volume (Dead Stock Check)
    if last['Volume'] < 1000: 
        # print(f"{ticker}: Low Volume")
        return None

    # Filter 3: The "Setup" Check
    uptrend = last['SMA_Fast'] > last['SMA_Slow']
    safe_rsi = last['RSI'] < BotConfig.RSI_MAX
    # We allow ADX to be slightly lower (15) to catch early moves
    strong_trend = last['ADX'] > BotConfig.ADX_THRESHOLD 
    
    # DEBUG: Print why it failed (Check your terminal)
    if not uptrend:
        # print(f"{ticker}: Failed Trend (Fast < Slow)")
        return None
    if not safe_rsi:
        # print(f"{ticker}: RSI too high ({last['RSI']:.1f})")
        return None
    if not strong_trend:
        # print(f"{ticker}: Trend too weak (ADX {last['ADX']:.1f})")
        return None

    # --- IF IT PASSED ALL FILTERS ---
    
    # Run AI Brain
    ai_score = train_and_predict(df)
    
    status = "✅ UPTREND"
    if ai_score > 60: status = "🦅 AI CONFIRMED"
    if ai_score > 75: status = "🔥 STRONG BUY"

    sparkline = df['Close'].tail(30).reset_index(drop=True)
    color = "#00FF00" if sparkline.iloc[-1] >= sparkline.iloc[0] else "#FF4B4B"

    # Trade Plan
    atr = last['ATR']
    stop_loss = price - (2 * atr)
    take_profit = price + (3 * atr) # Reduced target for higher win rate

    return {
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