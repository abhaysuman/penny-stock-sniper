import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --- CONFIGURATION (BASE) ---
class BotConfig:
    # These are now just baselines, we adjust them dynamically
    MACRO_SMA = 50     
    SMA_FAST = 9
    SMA_SLOW = 21

# --- HELPER: FETCH DAILY DATA ---
def check_macro_trend(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df is None or len(df) < 50: return False, "No Daily Data"

        current_price = df['Close'].iloc[-1]
        sma_50 = ta.sma(df['Close'], length=BotConfig.MACRO_SMA).iloc[-1]
        
        # Super safe check
        if pd.isna(current_price) or pd.isna(sma_50): return False, "Data Error"

        # Allow if price is near SMA (Loose) or above (Strict)
        if current_price < (sma_50 * 0.95): return False, "❌ Downtrend"

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
    
    # Error Handling for indicators
    try:
        df['CHOP'] = ta.chop(df['High'], df['Low'], df['Close'], length=14)
    except: df['CHOP'] = 50

    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    df['SMA_Fast'] = ta.sma(df['Close'], length=BotConfig.SMA_FAST)
    df['SMA_Slow'] = ta.sma(df['Close'], length=BotConfig.SMA_SLOW)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    if adx is not None and not adx.empty:
        df['ADX'] = adx.iloc[:, 0]
    else:
        df['ADX'] = 0
    
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    rolling_high = df['High'].rolling(22).max()
    df['Chandelier_Exit'] = rolling_high - (3 * df['ATR'])
    
    df['Vol_SMA'] = ta.sma(df['Volume'], length=20)
    df['RVOL'] = df['Volume'] / df['Vol_SMA']

    # Patterns
    df['CDL_HAMMER'] = ta.cdl_pattern(df['Open'], df['High'], df['Low'], df['Close'], name="hammer")['CDL_HAMMER']
    df['CDL_ENGULFING'] = ta.cdl_pattern(df['Open'], df['High'], df['Low'], df['Close'], name="engulfing")['CDL_ENGULFING']

    # ML Features
    df['VWAP_Dist'] = (df['Close'] - df['VWAP']) / df['VWAP']
    df['SMA_Dist'] = (df['Close'] - df['SMA_Slow']) / df['SMA_Slow']
    df['RSI_Norm'] = df['RSI'] / 100
    df['RVOL_Norm'] = df['RVOL']
    df['Pat_Hammer'] = df['CDL_HAMMER'].apply(lambda x: 1 if x != 0 else 0)
    df['Pat_Engulfing'] = df['CDL_ENGULFING'].apply(lambda x: 1 if x != 0 else 0)
    
    return df

def train_and_predict(df):
    try:
        data = df.copy().dropna()
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
        
        model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        model.fit(X, y)
        return round(model.predict_proba(X_live)[0][1] * 100, 1)
    except: return 50.0

def analyze_ticker_precision(ticker, wallet_size, strictness=5):
    """
    strictness (int): 1 (Loose) to 10 (Strict)
    """
    try:
        # 1. DYNAMIC THRESHOLDS BASED ON SLIDER
        # Strictness 1: Score > 50, RVOL > 0.5
        # Strictness 10: Score > 80, RVOL > 2.0
        min_ai_score = 50 + (strictness * 3) # 53 to 83
        min_rvol = 0.5 + (strictness * 0.15) # 0.65 to 2.0
        max_chop = 70 - (strictness * 2)     # 68 (Loose) to 50 (Strict)
        
        # MACRO CHECK
        is_healthy, reason = check_macro_trend(ticker)
        if not is_healthy and strictness > 3: return None, reason

        # MICRO CHECK
        df = fetch_intraday_data(ticker)
        if df is None: return None, "No Data"
        
        df = calculate_intraday_indicators(df)
        last = df.iloc[-1]
        price = last['Close']
        
        # Safety checks for NaNs
        if pd.isna(price) or pd.isna(last['RSI']): return None, "Bad Data"
        
        if last['Volume'] < (strictness * 10): return None, "Low Vol" # Dynamic volume filter

        # FILTERS
        # 1. Trend: Only check Moving Averages if strictness is high
        bullish_structure = True
        if strictness > 4:
            bullish_structure = (last['SMA_Fast'] > last['SMA_Slow'])
        
        # 2. Chop:
        not_choppy = last['CHOP'] < max_chop
        
        # 3. Volume:
        has_volume = last['RVOL'] > min_rvol

        if bullish_structure and not_choppy and has_volume:
            
            ai_score = train_and_predict(df)
            
            # THE DECISION MAKER
            if ai_score >= min_ai_score:
                status = "✅ UPTREND"
                
                # Badges
                if ai_score > (min_ai_score + 10): status = "🔥 STRONG BUY"
                if "STRONG" in status and (last['Pat_Hammer'] == 1 or last['Pat_Engulfing'] == 1):
                    status = "💎 DIAMOND SETUP"

                sparkline = df['Close'].tail(48).reset_index(drop=True)
                color = "#00FF00" if sparkline.iloc[-1] >= sparkline.iloc[0] else "#FF4B4B"

                # Risk Calc
                atr = last['ATR']
                chan_stop = last['Chandelier_Exit']
                hard_stop = price * 0.985 
                stop_loss = max(chan_stop, hard_stop)
                if pd.isna(stop_loss): stop_loss = price * 0.95

                take_profit = price + (2.5 * atr)
                
                # Shares
                safe_shares = int((wallet_size * 0.02) / (price - stop_loss)) if (price - stop_loss) > 0 else 1
                safe_shares = min(safe_shares, int((wallet_size * 0.20) / price))
                if safe_shares < 1: safe_shares = 1

                result = {
                    "Ticker": ticker,
                    "Price": price,
                    "RSI": last['RSI'],
                    "AI_Score": ai_score,
                    "Status": status,
                    "Chart": sparkline,
                    "Color": color,
                    "Shares": safe_shares,
                    "Stop_Loss": stop_loss,
                    "Take_Profit": take_profit
                }
                return result, "OK"
            
            return None, f"Low Score ({ai_score:.1f} < {min_ai_score})"
            
        return None, "Weak Setup"
    except Exception as e:
        return None, f"Error: {str(e)}"