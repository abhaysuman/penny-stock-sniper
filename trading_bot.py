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

def fetch_data(ticker, period="5d"): # Changed from 1y to 5d
    try:
        # --- THE REAL-TIME FIX ---
        # We fetch '5m' (5-minute) data instead of '1d' (Daily).
        # This makes the bot react to price changes TODAY.
        df = yf.download(ticker, period=period, interval="5m", progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.dropna()
        
        # We need at least 30 candles (30 * 5 mins = 2.5 hours of data)
        if len(df) < 30: return None
        return df
    except: return None

def calculate_indicators(df):
    df = df.copy()
    
    # These indicators now work on 5-MINUTE candles
    # SMA 21 now means "Average of the last 105 minutes" (Day Trading speed)
    df['SMA_Fast'] = ta.sma(df['Close'], length=BotConfig.SMA_FAST)
    df['SMA_Slow'] = ta.sma(df['Close'], length=BotConfig.SMA_SLOW)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['RSI_Slope'] = df['RSI'].diff(3) 
    
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx.iloc[:, 0]
    
    df['Vol_SMA'] = ta.sma(df['Volume'], length=20)
    
    # Safe Bollinger Bands
    bb = ta.bbands(df['Close'], length=20)
    df['BB_Lower'] = bb.iloc[:, 0]
    df['BB_Upper'] = bb.iloc[:, 2]
    
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # ML Features
    df['SMA_Dist'] = (df['Close'] - df['SMA_Slow']) / df['SMA_Slow']
    df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA']
    df['RSI_Norm'] = df['RSI'] / 100
    
    bb_range = df['BB_Upper'] - df['BB_Lower']
    df['BB_Pos'] = (df['Close'] - df['BB_Lower']) / bb_range.replace(0, 1)
    
    return df

def train_and_predict(df):
    try:
        data = df.copy().dropna()
        
        # Target: Price up > 0.5% in the next 30 mins (6 candles)
        # We lowered the target because we are on a faster timeframe
        future_return = data['Close'].shift(-6) / data['Close'] - 1
        data['Target'] = (future_return > 0.005).astype(int)
        
        features = ['SMA_Dist', 'Vol_Ratio', 'RSI_Norm', 'ADX', 'RSI_Slope', 'BB_Pos']
        data = data.dropna()
        
        if len(data) < 50: return 50.0

        X = data[features][:-6] 
        y = data['Target'][:-6]
        X_live = data[features].tail(1)
        
        model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        model.fit(X, y)
        
        return round(model.predict_proba(X_live)[0][1] * 100, 1)
    except: return 50.0

def analyze_ticker_precision(ticker, wallet_size):
    try:
        df = fetch_data(ticker)
        if df is None: return None, "❌ No Intraday Data"
        
        df = calculate_indicators(df)
        last = df.iloc[-1]
        price = last['Close']
        
        if price > wallet_size: return None, "Expensive"
        
        # Relax Volume check for 5-min candles (Volume is lower per candle)
        if last['Volume'] < 100: return None, "Low Vol"

        uptrend = last['SMA_Fast'] > last['SMA_Slow']
        safe_rsi = last['RSI'] < BotConfig.RSI_MAX
        
        if uptrend and safe_rsi:
            ai_score = train_and_predict(df)
            
            status = "✅ UPTREND"
            if (ai_score > 70) and (last['ADX'] > 20):
                status = "🔥 STRONG BUY"
            elif ai_score > 60:
                status = "🦅 AI CONFIRMED"

            # Chart: Last 60 candles (Last 5 hours of trading)
            sparkline = df['Close'].tail(60).reset_index(drop=True)
            color = "#00FF00" if sparkline.iloc[-1] >= sparkline.iloc[0] else "#FF4B4B"

            # Tighter Stops for Day Trading
            atr = last['ATR']
            stop_loss = price - (1.5 * atr) # Tighter Stop
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
            
        return None, "Weak Trend"
    except Exception as e:
        return None, f"Error: {str(e)}"