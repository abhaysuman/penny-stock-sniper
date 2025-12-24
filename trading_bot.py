import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION ---
class BotConfig:
    SMA_FAST = 9
    SMA_SLOW = 21
    RSI_MIN = 30
    RSI_MAX = 70
    ADX_THRESHOLD = 20
    VOLUME_THRESHOLD = 1.2

def fetch_data(ticker, period="2y"): # Fetches 2y data for better training
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
    # 1. Standard Indicators
    df['SMA_Fast'] = ta.sma(df['Close'], length=BotConfig.SMA_FAST)
    df['SMA_Slow'] = ta.sma(df['Close'], length=BotConfig.SMA_SLOW)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx['ADX_14']
    df['Vol_SMA'] = ta.sma(df['Volume'], length=20)
    
    # 2. ML Features (Data for the Brain)
    # The AI needs "Relative" numbers, not raw prices.
    df['SMA_Diff'] = (df['SMA_Fast'] - df['SMA_Slow']) / df['SMA_Slow'] # Trend Strength
    df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA'] # Relative Volume
    df['RSI_Norm'] = df['RSI'] / 100 # Normalized RSI
    
    return df

def train_and_predict(df):
    """
    Trains a Random Forest model on the fly for this specific stock.
    Returns: Probability (0-100%) that the price will go UP tomorrow.
    """
    try:
        data = df.copy().dropna()
        
        # 1. Create Target: Did price go up next day? (1 = Yes, 0 = No)
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        
        # 2. Define Features (What the AI looks at)
        features = ['SMA_Diff', 'Vol_Ratio', 'RSI_Norm', 'ADX']
        data = data.dropna() # Drop rows with NaNs created by shift/indicators
        
        # 3. Split Data
        # Train on everything EXCEPT the very last candle (today)
        X = data[features][:-1] 
        y = data['Target'][:-1]
        
        # The row we want to predict (Today's live data)
        X_live = data[features].tail(1)
        
        # 4. Train Model
        # n_estimators=100 means we create 100 "decision trees" and vote
        model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        model.fit(X, y)
        
        # 5. Predict
        # We get the probability of Class 1 (Price Going UP)
        probability = model.predict_proba(X_live)[0][1]
        
        return round(probability * 100, 1)
        
    except Exception as e:
        print(f"ML Error: {e}")
        return 50.0 # Default to neutral if error

def analyze_ticker_precision(ticker, wallet_size):
    # 1. Get Data
    df = fetch_data(ticker)
    if df is None: return None
    
    # 2. Add Indicators
    df = calculate_indicators(df)
    last = df.iloc[-1]
    price = last['Close']
    
    # 3. Filters
    if price > wallet_size: return None
    if last['Volume'] < 1000: return None # Dead stock check

    # 4. Strict "Scanner" Rules (Fast Check)
    uptrend = last['SMA_Fast'] > last['SMA_Slow']
    safe_rsi = last['RSI'] < BotConfig.RSI_MAX
    strong_trend = last['ADX'] > BotConfig.ADX_THRESHOLD
    
    if uptrend and safe_rsi and strong_trend:
        
        # 5. Run the AI Brain (Slow Check)
        # We only run this if the stock passes the basic filters
        ai_score = train_and_predict(df)
        
        # Logic for Badge
        status = "✅ UPTREND"
        if ai_score > 65: status = "🦅 AI CONFIRMED"
        if ai_score > 80: status = "🔥 STRONG BUY"

        # Chart Data
        sparkline = df['Close'].tail(30).reset_index(drop=True)
        color = "#00FF00" if sparkline.iloc[-1] >= sparkline.iloc[0] else "#FF4B4B"

        return {
            "Ticker": ticker,
            "Price": price,
            "RSI": last['RSI'],
            "AI_Score": ai_score, # The new magic number
            "Status": status,
            "Chart": sparkline,
            "Color": color,
            "Shares": int(wallet_size // price)
        }
        
    return None