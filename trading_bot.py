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
    RSI_MAX = 80       # Relaxed to 80
    ADX_THRESHOLD = 15 
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
        
        # Ensure enough data exists
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
    
    # Advanced
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
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        
        features = ['SMA_Diff', 'Vol_Ratio', 'RSI_Norm', 'ADX', 'MACD_Diff', 'BB_Pos']
        data = data.dropna()
        
        if len(data) < 50: return 50.0

        X = data[features][:-1] 
        y = data['Target'][:-1]
        X_live = data[features].tail(1)
        
        model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        model.fit(X, y)
        probability = model.predict_proba(X_live)[0][1]
        
        return round(probability * 100, 1)
    except: return 50.0

def analyze_ticker_precision(ticker, wallet_size):
    df = fetch_data(ticker)
    if df is None: return None
    
    df = calculate_indicators(df)
    last = df.iloc[-1]
    price = last['Close']
    
    # --- BASIC FILTERS (Must Pass) ---
    if price > wallet_size: return None
    if last['Volume'] < 1000: return None # Dead stock check

    # --- TREND FILTERS (Flexible) ---
    uptrend = last['SMA_Fast'] > last['SMA_Slow']
    safe_rsi = last['RSI'] < BotConfig.RSI_MAX
    
    # We REMOVED the strict ADX rejection here.
    # Now we just require a basic Uptrend OR a Safe RSI.
    if uptrend and safe_rsi:
        
        # Run AI
        ai_score = train_and_predict(df)
        
        # --- BADGE LOGIC (Tiered) ---
        status = "⚠️ WATCHLIST" # Default
        
        # Tier 1: The "Perfect" Trade
        if (last['ADX'] > 20) and (ai_score > 70):
            status = "🔥 STRONG BUY"
            
        # Tier 2: The "Good" Trade
        elif (ai_score > 60):
            status = "🦅 AI CONFIRMED"
            
        # Tier 3: Basic Uptrend
        elif uptrend:
            status = "✅ UPTREND"

        sparkline = df['Close'].tail(30).reset_index(drop=True)
        color = "#00FF00" if sparkline.iloc[-1] >= sparkline.iloc[0] else "#FF4B4B"

        atr = last['ATR']
        stop_loss = price - (2 * atr)
        take_profit = price + (3 * atr)

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
        
    return None