import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --- CONFIGURATION ---
class BotConfig:
    # Daily Chart Settings
    MACRO_SMA = 50     
    MACRO_RSI_MAX = 80 

    # Intraday Chart Settings
    SMA_FAST = 9
    SMA_SLOW = 21
    RSI_MIN = 35       
    RSI_MAX = 75       
    ADX_THRESHOLD = 25 
    CHOP_THRESHOLD = 50 

# --- HELPER: FETCH DAILY DATA ---
def check_macro_trend(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if len(df) < 50: return False, "No Daily Data"

        current_price = df['Close'].iloc[-1]
        sma_50 = ta.sma(df['Close'], length=BotConfig.MACRO_SMA).iloc[-1]
        daily_rsi = ta.rsi(df['Close'], length=14).iloc[-1]

        if current_price < sma_50: return False, "❌ Downtrend (Daily)"
        if daily_rsi > BotConfig.MACRO_RSI_MAX: return False, "⚠️ Overbought (Daily)"

        return True, "OK"
    except: return False, "Data Error"

# --- HELPER: FETCH INTRADAY DATA ---
def fetch_intraday_data(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if len(df) < 50: return None
        return df
    except: return None

def calculate_intraday_indicators(df):
    df = df.copy()
    
    # --- 1. PATTERN RECOGNITION (The "Eyes") ---
    # We detect specific bullish reversal patterns
    # These functions return non-zero values if a pattern is found
    
    # CDL_HAMMER: Sellers gave up
    df['CDL_HAMMER'] = ta.cdl_pattern(df['Open'], df['High'], df['Low'], df['Close'], name="hammer")['CDL_HAMMER']
    
    # CDL_ENGULFING: Buyers overwhelmed sellers
    df['CDL_ENGULFING'] = ta.cdl_pattern(df['Open'], df['High'], df['Low'], df['Close'], name="engulfing")['CDL_ENGULFING']
    
    # --- 2. QUANT INDICATORS ---
    try:
        df['CHOP'] = ta.chop(df['High'], df['Low'], df['Close'], length=14)
    except: df['CHOP'] = 50

    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    
    df['Force_Index'] = df['Close'].diff(1) * df['Volume']
    df['Force_SMA'] = ta.sma(df['Force_Index'], length=13)

    df['SMA_Fast'] = ta.sma(df['Close'], length=BotConfig.SMA_FAST)
    df['SMA_Slow'] = ta.sma(df['Close'], length=BotConfig.SMA_SLOW)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14).iloc[:, 0]
    
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    rolling_high = df['High'].rolling(22).max()
    df['Chandelier_Exit'] = rolling_high - (3 * df['ATR'])
    
    df['Vol_SMA'] = ta.sma(df['Volume'], length=20)
    df['RVOL'] = df['Volume'] / df['Vol_SMA']

    # --- 3. ML FEATURES (Including Patterns) ---
    df['VWAP_Dist'] = (df['Close'] - df['VWAP']) / df['VWAP']
    df['SMA_Dist'] = (df['Close'] - df['SMA_Slow']) / df['SMA_Slow']
    df['RSI_Norm'] = df['RSI'] / 100
    df['RVOL_Norm'] = df['RVOL']
    
    # Normalize Patterns: Convert huge numbers to just 0 or 1 for the AI
    df['Pat_Hammer'] = df['CDL_HAMMER'].apply(lambda x: 1 if x != 0 else 0)
    df['Pat_Engulfing'] = df['CDL_ENGULFING'].apply(lambda x: 1 if x != 0 else 0)
    
    return df

def train_and_predict(df):
    try:
        data = df.copy().dropna()
        
        # --- SMART TARGET (Volatility Adjusted) ---
        # Instead of fixed 0.3%, we ask: "Did price move more than 0.5x ATR?"
        # This adapts to the stock's personality.
        future_close = data['Close'].shift(-4)
        current_close = data['Close']
        target_move = 0.5 * data['ATR']
        
        data['Target'] = ((future_close - current_close) > target_move).astype(int)
        
        features = ['VWAP_Dist', 'SMA_Dist', 'RSI_Norm', 'ADX', 'CHOP', 'RVOL_Norm', 'Pat_Hammer', 'Pat_Engulfing']
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
        # MACRO CHECK
        is_healthy, reason = check_macro_trend(ticker)
        if not is_healthy: return None, reason

        # MICRO CHECK
        df = fetch_intraday_data(ticker)
        if df is None: return None, "❌ No Intraday Data"
        
        df = calculate_intraday_indicators(df)
        last = df.iloc[-1]
        price = last['Close']
        
        if price > wallet_size: return None, "Expensive"
        if last['Volume'] < 50: return None, "Low Liquidity"

        # FILTERS
        bullish_structure = (price > last['VWAP']) and (last['SMA_Fast'] > last['SMA_Slow'])
        not_choppy = last['CHOP'] < 60
        has_volume = last['RVOL'] > 1.0

        if bullish_structure and not_choppy and has_volume:
            
            ai_score = train_and_predict(df)
            
            status = "✅ UPTREND"
            
            # PATTERN BOOSTER: If we see a pattern, we trust the AI more
            has_pattern = (last['Pat_Hammer'] == 1) or (last['Pat_Engulfing'] == 1)
            
            if (ai_score > 75) and (last['RVOL'] > 1.5):
                status = "🔥 STRONG BUY"
                if has_pattern: status = "💎 DIAMOND SETUP" # New Rare Badge
                
            elif ai_score > 60:
                status = "🦅 AI CONFIRMED"

            sparkline = df['Close'].tail(48).reset_index(drop=True)
            color = "#00FF00" if sparkline.iloc[-1] >= sparkline.iloc[0] else "#FF4B4B"

            chan_stop = last['Chandelier_Exit']
            hard_stop = price * 0.985 
            stop_loss = max(chan_stop, hard_stop) 
            take_profit = price + (3.0 * last['ATR'])

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
            
        return None, "Weak Setup"
    except Exception as e:
        return None, f"Error: {str(e)}"