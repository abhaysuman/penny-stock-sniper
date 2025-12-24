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
    RSI_MAX = 75       
    ADX_THRESHOLD = 20 
    VOLUME_THRESHOLD = 1.0

def fetch_data(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if len(df) < 60: return None
        return df
    except: return None

def calculate_indicators(df):
    df = df.copy()
    
    # 1. Standard Indicators
    df['SMA_Fast'] = ta.sma(df['Close'], length=BotConfig.SMA_FAST)
    df['SMA_Slow'] = ta.sma(df['Close'], length=BotConfig.SMA_SLOW)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['RSI_Slope'] = df['RSI'].diff(3) 
    
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    # Pandas TA returns 3 columns for ADX, we need the first one usually named ADX_14
    # We use iloc to be safe against naming changes
    df['ADX'] = adx.iloc[:, 0]
    
    df['Vol_SMA'] = ta.sma(df['Volume'], length=20)
    
    # 2. Bollinger Bands (THE FIX)
    bb = ta.bbands(df['Close'], length=20)
    # Instead of hardcoding 'BBU_20_2.0', we grab them by position
    # BBands returns: [Lower, Mid, Upper, Bandwidth, Percent]
    df['BB_Lower'] = bb.iloc[:, 0]
    df['BB_Upper'] = bb.iloc[:, 2]
    
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # 3. ML Features
    df['SMA_Dist'] = (df['Close'] - df['SMA_Slow']) / df['SMA_Slow']
    df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA']
    df['RSI_Norm'] = df['RSI'] / 100
    
    # Safety for BB_Pos to avoid division by zero
    bb_range = df['BB_Upper'] - df['BB_Lower']
    df['BB_Pos'] = (df['Close'] - df['BB_Lower']) / bb_range.replace(0, 1)
    
    return df

def train_and_predict(df):
    try:
        data = df.copy().dropna()
        
        # Target: Price up > 1% in 2 days
        future_return = data['Close'].shift(-2) / data['Close'] - 1
        data['Target'] = (future_return > 0.01).astype(int)
        
        features = ['SMA_Dist', 'Vol_Ratio', 'RSI_Norm', 'ADX', 'RSI_Slope', 'BB_Pos']
        data = data.dropna()
        
        if len(data) < 50: return 50.0

        X = data[features][:-2] 
        y = data['Target'][:-2]
        X_live = data[features].tail(1)
        
        model = RandomForestClassifier(n_estimators=200, min_samples_split=10, random_state=42)
        model.fit(X, y)
        
        return round(model.predict_proba(X_live)[0][1] * 100, 1)
    except: return 50.0

def analyze_ticker_precision(ticker, wallet_size):
    try:
        df = fetch_data(ticker)
        if df is None: return None, "❌ Data Error"
        
        df = calculate_indicators(df)
        last = df.iloc[-1]
        price = last['Close']
        
        if price > wallet_size: return None, "Expensive"
        if last['Volume'] < 1000: return None, "Low Volume"

        uptrend = last['SMA_Fast'] > last['SMA_Slow']
        safe_rsi = last['RSI'] < BotConfig.RSI_MAX
        
        if uptrend and safe_rsi:
            
            ai_score = train_and_predict(df)
            
            status = "✅ UPTREND"
            # Strict logic for Strong Buy
            if (ai_score > 75) and (last['ADX'] > 20) and (last['RSI_Slope'] > 0):
                status = "🔥 STRONG BUY"
            elif ai_score > 60:
                status = "🦅 AI CONFIRMED"

            sparkline = df['Close'].tail(30).reset_index(drop=True)
            color = "#00FF00" if sparkline.iloc[-1] >= sparkline.iloc[0] else "#FF4B4B"

            atr = last['ATR']
            stop_loss = price - (2 * atr)
            take_profit = price + (4 * atr)

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
    except Exception as e:
        return None, f"Error: {str(e)}"