import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --- CONFIGURATION ---
class BotConfig:
    # Daily Chart Settings
    MACRO_SMA = 50     
    MACRO_RSI_MAX = 85 # Relaxed from 80 to allow stronger momentum

    # Intraday Chart Settings
    SMA_FAST = 9
    SMA_SLOW = 21
    RSI_MIN = 30       
    RSI_MAX = 80       # Relaxed upper limit
    ADX_THRESHOLD = 20 # Relaxed from 25 to catch early trends
    CHOP_THRESHOLD = 60 # Relaxed from 50 (Crucial for seeing more results)

# --- HELPER: FETCH DAILY DATA ---
def check_macro_trend(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if len(df) < 50: return False, "No Daily Data"

        current_price = df['Close'].iloc[-1]
        sma_50 = ta.sma(df['Close'], length=BotConfig.MACRO_SMA).iloc[-1]
        
        # Relaxed Filter: Allow if price is > 98% of SMA (Close enough)
        # This prevents rejecting a good stock just because it dipped slightly
        if current_price < (sma_50 * 0.98): return False, "❌ Downtrend"

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
    
    # Patterns
    df['CDL_HAMMER'] = ta.cdl_pattern(df['Open'], df['High'], df['Low'], df['Close'], name="hammer")['CDL_HAMMER']
    df['CDL_ENGULFING'] = ta.cdl_pattern(df['Open'], df['High'], df['Low'], df['Close'], name="engulfing")['CDL_ENGULFING']
    
    # Indicators
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

def analyze_ticker_precision(ticker, wallet_size):
    try:
        # MACRO CHECK
        is_healthy, reason = check_macro_trend(ticker)
        if not is_healthy: return None, reason

        # MICRO CHECK
        df = fetch_intraday_data(ticker)
        if df is None: return None, "No Data"
        
        df = calculate_intraday_indicators(df)
        last = df.iloc[-1]
        price = last['Close']
        
        # Relaxed volume check to allow smaller caps
        if last['Volume'] < 100: return None, "Low Liquidity"

        # FILTERS (Relaxed for visibility)
        bullish_structure = (last['SMA_Fast'] > last['SMA_Slow'])
        
        # Relaxed Chop: Allow up to 60 (was 50)
        not_choppy = last['CHOP'] < BotConfig.CHOP_THRESHOLD
        
        # Relaxed Volume: Just needs to be active, not necessarily explosive (0.8x avg)
        has_volume = last['RVOL'] > 0.8 

        if bullish_structure and not_choppy and has_volume:
            
            ai_score = train_and_predict(df)
            
            status = "👀 WATCHLIST"
            
            # Badge Logic
            has_pattern = (last['Pat_Hammer'] == 1) or (last['Pat_Engulfing'] == 1)
            
            if (ai_score > 70) and (last['RVOL'] > 1.2):
                status = "🔥 STRONG BUY"
                if has_pattern: status = "💎 DIAMOND SETUP"
            elif ai_score > 55:
                status = "✅ UPTREND"

            sparkline = df['Close'].tail(48).reset_index(drop=True)
            color = "#00FF00" if sparkline.iloc[-1] >= sparkline.iloc[0] else "#FF4B4B"

            # Risk Calc
            atr = last['ATR']
            chan_stop = last['Chandelier_Exit']
            hard_stop = price * 0.985 
            stop_loss = max(chan_stop, hard_stop)
            
            # Position Sizing
            risk_per_trade = wallet_size * 0.01 
            risk_per_share = price - stop_loss
            if risk_per_share <= 0: risk_per_share = price * 0.01
            safe_shares = int(risk_per_trade / risk_per_share)
            max_capital_per_trade = wallet_size * 0.20
            max_shares_by_capital = int(max_capital_per_trade / price)
            final_shares = min(safe_shares, max_shares_by_capital)
            if final_shares < 1: final_shares = 1

            take_profit = price + (2.5 * atr)

            result = {
                "Ticker": ticker,
                "Price": price,
                "RSI": last['RSI'],
                "AI_Score": ai_score,
                "Status": status,
                "Chart": sparkline,
                "Color": color,
                "Shares": final_shares,
                "Stop_Loss": stop_loss,
                "Take_Profit": take_profit
            }
            return result, "OK"
            
        return None, "Weak Setup"
    except Exception as e:
        return None, f"Error: {str(e)}"