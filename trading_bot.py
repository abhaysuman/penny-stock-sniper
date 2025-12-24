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
    RSI_MAX = 70
    ADX_THRESHOLD = 20
    VOLUME_THRESHOLD = 1.2

def fetch_data(ticker, period="2y"):
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
    
    # --- 1. EXISTING INDICATORS ---
    df['SMA_Fast'] = ta.sma(df['Close'], length=BotConfig.SMA_FAST)
    df['SMA_Slow'] = ta.sma(df['Close'], length=BotConfig.SMA_SLOW)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx['ADX_14']
    df['Vol_SMA'] = ta.sma(df['Volume'], length=20)
    
    # --- 2. NEW INDICATORS (More Eyes) ---
    # MACD: The King of Momentum
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    
    # Bollinger Bands: Volatility Squeeze detection
    bb = ta.bbands(df['Close'], length=20)
    df['BB_Upper'] = bb['BBU_20_2.0']
    df['BB_Lower'] = bb['BBL_20_2.0']
    
    # ATR: For calculating Stop Loss & Profit Targets
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # --- 3. ML FEATURES (Normalized for AI) ---
    df['SMA_Diff'] = (df['SMA_Fast'] - df['SMA_Slow']) / df['SMA_Slow']
    df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA']
    df['RSI_Norm'] = df['RSI'] / 100
    df['MACD_Diff'] = (df['MACD'] - df['MACD_Signal']) # Positive = Bullish Momentum
    df['BB_Pos'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']) # 0 = Bottom, 1 = Top
    
    return df

def train_and_predict(df):
    try:
        data = df.copy().dropna()
        
        # Target: Price Up AND Volatility Up (Better quality target)
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        
        # Updated Feature List
        features = ['SMA_Diff', 'Vol_Ratio', 'RSI_Norm', 'ADX', 'MACD_Diff', 'BB_Pos']
        data = data.dropna()
        
        X = data[features][:-1] 
        y = data['Target'][:-1]
        X_live = data[features].tail(1)
        
        # Train
        model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        model.fit(X, y)
        probability = model.predict_proba(X_live)[0][1]
        
        return round(probability * 100, 1)
    except Exception as e:
        print(f"ML Error: {e}")
        return 50.0

def analyze_ticker_precision(ticker, wallet_size):
    df = fetch_data(ticker)
    if df is None: return None
    
    df = calculate_indicators(df)
    last = df.iloc[-1]
    price = last['Close']
    
    if price > wallet_size: return None
    if last['Volume'] < 1000: return None

    # Scanner Rules (Fast Check)
    uptrend = last['SMA_Fast'] > last['SMA_Slow']
    safe_rsi = last['RSI'] < BotConfig.RSI_MAX
    strong_trend = last['ADX'] > BotConfig.ADX_THRESHOLD
    
    if uptrend and safe_rsi and strong_trend:
        
        ai_score = train_and_predict(df)
        
        status = "✅ UPTREND"
        if ai_score > 65: status = "🦅 AI CONFIRMED"
        if ai_score > 80: status = "🔥 STRONG BUY"

        sparkline = df['Close'].tail(30).reset_index(drop=True)
        color = "#00FF00" if sparkline.iloc[-1] >= sparkline.iloc[0] else "#FF4B4B"

        # --- RISK MANAGEMENT CALCULATION ---
        # ATR (Average True Range) tells us how much the price moves in a day.
        # Stop Loss = 2x ATR below price (Give it room to breathe)
        # Take Profit = 4x ATR above price (Aim for 2:1 Reward ratio)
        atr = last['ATR']
        stop_loss = price - (2 * atr)
        take_profit = price + (4 * atr)

        return {
            "Ticker": ticker,
            "Price": price,
            "RSI": last['RSI'],
            "AI_Score": ai_score,
            "Status": status,
            "Chart": sparkline,
            "Color": color,
            "Shares": int(wallet_size // price),
            # New Data Points
            "Stop_Loss": stop_loss,
            "Take_Profit": take_profit
        }
        
    return None