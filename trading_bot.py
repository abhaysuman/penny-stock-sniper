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
    RSI_MAX = 80
    ADX_THRESHOLD = 15 
    VOLUME_THRESHOLD = 1.0

def fetch_data(ticker, period="1y"):
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
    # Indicators
    df['SMA_Fast'] = ta.sma(df['Close'], length=BotConfig.SMA_FAST)
    df['SMA_Slow'] = ta.sma(df['Close'], length=BotConfig.SMA_SLOW)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx['ADX_14']
    df['Vol_SMA'] = ta.sma(df['Volume'], length=20)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # ML Features
    df['SMA_Diff'] = (df['SMA_Fast'] - df['SMA_Slow']) / df['SMA_Slow']
    df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA']
    df['RSI_Norm'] = df['RSI'] / 100
    
    return df

def train_and_predict(df):
    try:
        data = df.copy().dropna()
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        features = ['SMA_Diff', 'Vol_Ratio', 'RSI_Norm', 'ADX']
        data = data.dropna()
        if len(data) < 50: return 50.0
        
        X = data[features][:-1] 
        y = data['Target'][:-1]
        X_live = data[features].tail(1)
        
        model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        model.fit(X, y)
        return round(model.predict_proba(X_live)[0][1] * 100, 1)
    except: return 50.0

def analyze_ticker_precision(ticker, wallet_size):
    # 1. Fetch
    df = fetch_data(ticker)
    if df is None: 
        return None, "❌ No Data / Download Failed"
    
    # 2. Indicators
    df = calculate_indicators(df)
    last = df.iloc[-1]
    price = last['Close']
    
    # 3. Filters with Reasons
    if price > wallet_size: 
        return None, f"💸 Too Expensive ({price:.2f})"
        
    if last['Volume'] < 1000: 
        return None, "Volume Dead (< 1k)"

    uptrend = last['SMA_Fast'] > last['SMA_Slow']
    safe_rsi = last['RSI'] < BotConfig.RSI_MAX
    
    if uptrend and safe_rsi:
        ai_score = train_and_predict(df)
        
        status = "✅ UPTREND"
        if ai_score > 60: status = "🦅 AI CONFIRMED"
        if ai_score > 70: status = "🔥 STRONG BUY" # Lowered threshold slightly

        sparkline = df['Close'].tail(30).reset_index(drop=True)
        color = "#00FF00" if sparkline.iloc[-1] >= sparkline.iloc[0] else "#FF4B4B"
        
        atr = last['ATR']
        stop_loss = price - (2 * atr)
        take_profit = price + (3 * atr)

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
        
    return None, "📉 Downtrend or High RSI"