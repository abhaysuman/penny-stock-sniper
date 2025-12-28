import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --- CONFIGURATION ---
class BotConfig:
    # Daily Chart Settings (The Macro View)
    MACRO_SMA = 50     # Trend Filter (Must be above 50 SMA)
    MACRO_RSI_MAX = 80 # Don't buy if daily chart is overbought

    # Intraday Chart Settings (The Micro View)
    SMA_FAST = 9
    SMA_SLOW = 21
    RSI_MIN = 35       
    RSI_MAX = 75       
    ADX_THRESHOLD = 25 
    CHOP_THRESHOLD = 50 

# --- HELPER: FETCH DAILY DATA (FAST FILTER) ---
def check_macro_trend(ticker):
    """
    Checks the Daily chart first. If the big trend is bad, we reject immediately.
    This saves time and avoids 'catching falling knives'.
    """
    try:
        # Download 1 Year of Daily Data
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if len(df) < 50: return False, "No Daily Data"

        # Calculate Indicators
        current_price = df['Close'].iloc[-1]
        sma_50 = ta.sma(df['Close'], length=BotConfig.MACRO_SMA).iloc[-1]
        daily_rsi = ta.rsi(df['Close'], length=14).iloc[-1]

        # FILTER 1: Trend
        # Stock must be above its 50-Day Moving Average
        if current_price < sma_50:
            return False, "❌ Downtrend (Daily)"

        # FILTER 2: Overbought
        # Don't buy if the daily RSI is already screaming high
        if daily_rsi > BotConfig.MACRO_RSI_MAX:
            return False, "⚠️ Overbought (Daily)"

        return True, "OK"
    except:
        return False, "Data Error"

# --- HELPER: FETCH INTRADAY DATA (SNIPER ENTRY) ---
def fetch_intraday_data(ticker):
    try:
        # Download 5 Days of 5-Minute Data
        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if len(df) < 50: return None
        return df
    except: return None

def calculate_intraday_indicators(df):
    df = df.copy()
    
    # --- QUANT INDICATORS ---
    try:
        df['CHOP'] = ta.chop(df['High'], df['Low'], df['Close'], length=14)
    except: df['CHOP'] = 50

    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    
    # Force Index (Buying Pressure)
    df['Force_Index'] = df['Close'].diff(1) * df['Volume']
    df['Force_SMA'] = ta.sma(df['Force_Index'], length=13)

    # Standard
    df['SMA_Fast'] = ta.sma(df['Close'], length=BotConfig.SMA_FAST)
    df['SMA_Slow'] = ta.sma(df['Close'], length=BotConfig.SMA_SLOW)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14).iloc[:, 0]
    
    # Exits
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    rolling_high = df['High'].rolling(22).max()
    df['Chandelier_Exit'] = rolling_high - (3 * df['ATR'])
    
    # Relative Volume (RVOL)
    df['Vol_SMA'] = ta.sma(df['Volume'], length=20)
    df['RVOL'] = df['Volume'] / df['Vol_SMA']

    # ML Features
    df['VWAP_Dist'] = (df['Close'] - df['VWAP']) / df['VWAP']
    df['SMA_Dist'] = (df['Close'] - df['SMA_Slow']) / df['SMA_Slow']
    df['RSI_Norm'] = df['RSI'] / 100
    df['RVOL_Norm'] = df['RVOL']
    
    return df

def train_and_predict(df):
    try:
        data = df.copy().dropna()
        future_return = data['Close'].shift(-4) / data['Close'] - 1
        data['Target'] = (future_return > 0.003).astype(int) # 0.3% target
        
        features = ['VWAP_Dist', 'SMA_Dist', 'RSI_Norm', 'ADX', 'CHOP', 'RVOL_Norm']
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
        # STEP 1: MACRO CHECK (The Gatekeeper)
        # If the Daily Chart looks bad, we reject immediately.
        is_healthy, reason = check_macro_trend(ticker)
        if not is_healthy:
            return None, reason

        # STEP 2: MICRO CHECK (The Sniper)
        df = fetch_intraday_data(ticker)
        if df is None: return None, "❌ No Intraday Data"
        
        df = calculate_intraday_indicators(df)
        last = df.iloc[-1]
        price = last['Close']
        
        if price > wallet_size: return None, "Expensive"
        if last['Volume'] < 50: return None, "Low Liquidity"

        # --- INTRADAY FILTERS ---
        
        # 1. Structure Check
        # Price > VWAP (Institutions Buying)
        # Fast SMA > Slow SMA (Momentum Up)
        bullish_structure = (price > last['VWAP']) and (last['SMA_Fast'] > last['SMA_Slow'])
        
        # 2. Chop Check (Avoid sideways markets)
        not_choppy = last['CHOP'] < 60
        
        # 3. Volume Check
        has_volume = last['RVOL'] > 1.0

        if bullish_structure and not_choppy and has_volume:
            
            ai_score = train_and_predict(df)
            
            status = "✅ UPTREND"
            
            # Ultra Strict Badge
            if (ai_score > 80) and (last['RVOL'] > 2.0):
                status = "🔥 STRONG BUY"
            elif ai_score > 65:
                status = "🦅 AI CONFIRMED"

            sparkline = df['Close'].tail(48).reset_index(drop=True)
            color = "#00FF00" if sparkline.iloc[-1] >= sparkline.iloc[0] else "#FF4B4B"

            # Dynamic Exits
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
            
        return None, "Weak Intraday Setup"
    except Exception as e:
        return None, f"Error: {str(e)}"