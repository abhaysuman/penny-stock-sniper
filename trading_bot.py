import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --- CONFIGURATION ---
class BotConfig:
    SMA_FAST = 9
    SMA_SLOW = 21
    RSI_MIN = 35       
    RSI_MAX = 75       
    ADX_THRESHOLD = 25 # Increased to avoid weak trends
    CHOP_THRESHOLD = 50 # Below 50 = Trending, Above 50 = Choppy

def fetch_data(ticker, period="5d"):
    """
    Fetches 5-minute Intraday data.
    """
    try:
        df = yf.download(ticker, period=period, interval="5m", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if len(df) < 50: return None
        return df
    except: return None

def calculate_indicators(df):
    """
    Adds Quantitative Indicators (Choppiness, Force Index, Chandelier Exit).
    """
    df = df.copy()
    
    # --- 1. REGIME DETECTION (Choppiness Index) ---
    # Values > 61.8 indicate consolidation (SIDEWAYS)
    # Values < 38.2 indicate a trend (Run!)
    # We use this to filters out "fake" breakouts.
    try:
        df['CHOP'] = ta.chop(df['High'], df['Low'], df['Close'], length=14)
    except:
        df['CHOP'] = 50 # Default if calculation fails
    
    # --- 2. FORCE INDEX (Volume + Price Change) ---
    # Measures the "Power" behind a move. 
    # Positive = Bullish Pressure, Negative = Bearish Pressure
    df['Force_Index'] = df['Close'].diff(1) * df['Volume']
    df['Force_SMA'] = ta.sma(df['Force_Index'], length=13) # Smoothed Force

    # --- 3. STANDARD INDICATORS ---
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    df['SMA_Fast'] = ta.sma(df['Close'], length=BotConfig.SMA_FAST)
    df['SMA_Slow'] = ta.sma(df['Close'], length=BotConfig.SMA_SLOW)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14).iloc[:, 0]
    
    # --- 4. VOLATILITY & EXITS ---
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # Chandelier Exit (Long) logic for calculating dynamic stops
    # Highest High of last 22 bars - 3 * ATR
    rolling_high = df['High'].rolling(22).max()
    df['Chandelier_Exit'] = rolling_high - (3 * df['ATR'])

    # --- 5. ML FEATURES (Normalized) ---
    df['VWAP_Dist'] = (df['Close'] - df['VWAP']) / df['VWAP']
    df['SMA_Dist'] = (df['Close'] - df['SMA_Slow']) / df['SMA_Slow']
    df['RSI_Norm'] = df['RSI'] / 100
    df['Force_Norm'] = df['Force_SMA'] / df['Volume'].rolling(20).mean() # Normalized Force
    
    return df

def train_and_predict(df):
    try:
        data = df.copy().dropna()
        
        # Target: Price up > 0.4% in next 20 mins (4 candles)
        future_return = data['Close'].shift(-4) / data['Close'] - 1
        data['Target'] = (future_return > 0.004).astype(int)
        
        # New Feature Set
        features = ['VWAP_Dist', 'SMA_Dist', 'RSI_Norm', 'ADX', 'CHOP', 'Force_Norm']
        data = data.dropna()
        
        if len(data) < 50: return 50.0

        X = data[features][:-4] 
        y = data['Target'][:-4]
        X_live = data[features].tail(1)
        
        model = RandomForestClassifier(n_estimators=150, min_samples_split=10, random_state=42)
        model.fit(X, y)
        
        return round(model.predict_proba(X_live)[0][1] * 100, 1)
    except: return 50.0

def analyze_ticker_precision(ticker, wallet_size):
    try:
        df = fetch_data(ticker)
        if df is None: return None, "❌ No Data"
        
        df = calculate_indicators(df)
        last = df.iloc[-1]
        price = last['Close']
        
        if price > wallet_size: return None, "Expensive"
        if last['Volume'] < 50: return None, "Low Liquidity"

        # --- QUANTITATIVE FILTERS ---
        
        # 1. Regime Check (Is the market trending?)
        # If Chop Index is > 60, market is sleeping. We skip.
        is_trending = last['CHOP'] < 60
        
        # 2. Force Check (Is there buying pressure?)
        positive_force = last['Force_SMA'] > 0
        
        # 3. Structural Check (VWAP + SMA)
        bullish_structure = (price > last['VWAP']) and (last['SMA_Fast'] > last['SMA_Slow'])
        
        if is_trending and positive_force and bullish_structure:
            
            ai_score = train_and_predict(df)
            
            status = "✅ UPTREND"
            
            # Strict "Alpha" Badge
            if (ai_score > 75) and (last['CHOP'] < 45):
                status = "🔥 STRONG BUY" # Trending HARD
            elif ai_score > 60:
                status = "🦅 AI CONFIRMED"

            sparkline = df['Close'].tail(48).reset_index(drop=True)
            color = "#00FF00" if sparkline.iloc[-1] >= sparkline.iloc[0] else "#FF4B4B"

            # Dynamic Exits
            # Stop Loss is based on Chandelier Exit (Trailing Stop)
            # If Chandelier is way below, we cap it at max 2% loss
            chan_stop = last['Chandelier_Exit']
            hard_stop = price * 0.98
            stop_loss = max(chan_stop, hard_stop) 
            
            # Target is based on volatility
            take_profit = price + (2.5 * last['ATR'])

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
            
        return None, "Choppy or Weak Force"
    except Exception as e:
        return None, f"Error: {str(e)}"