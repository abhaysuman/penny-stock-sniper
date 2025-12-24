import yfinance as yf
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# --- 1. CONFIGURATION ---
STOCK_SYMBOL = "GOOG"  # Google Stock
START_DATE = "2020-01-01"
CASH_TO_INVEST = 10000 # Simulation money ($10k)

# --- 2. GET DATA ---
# We download the data and clean it for the backtester
def get_stock_data(symbol):
    df = yf.download(symbol, start=START_DATE, progress=False)
    # The Backtester expects columns: Open, High, Low, Close, Volume
    # yfinance sometimes returns multi-level columns, we flatten them just in case
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# --- 3. THE STRATEGY ---
class TrendFollower(Strategy):
    # Parameters we can optimize later to find "most profits"
    n1 = 50   # Short-term trend (50 days)
    n2 = 100  # Long-term trend (100 days)
    
    def init(self):
        # This runs once at the start. We calculate the indicators here.
        
        # Calculate the 50-day and 100-day Moving Averages
        self.sma1 = self.I(ta.sma, pd.Series(self.data.Close), length=self.n1)
        self.sma2 = self.I(ta.sma, pd.Series(self.data.Close), length=self.n2)
        
        # Calculate RSI (Momentum)
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), length=14)

    def next(self):
        # This runs for every single day in the data.
        # This is where the logic lives.
        
        # BUY SIGNAL:
        # If SMA 50 crosses ABOVE SMA 100 (Trend is Up)
        # AND RSI is below 70 (Not too expensive yet)
        if crossover(self.sma1, self.sma2) and self.rsi < 70:
            self.buy()

        # SELL SIGNAL:
        # If SMA 50 crosses BELOW SMA 100 (Trend is Down)
        elif crossover(self.sma2, self.sma1):
            self.position.close()

# --- 4. EXECUTION ---
import pandas as pd # Import pandas here to ensure availability

print(f"Fetching data for {STOCK_SYMBOL}...")
data = get_stock_data(STOCK_SYMBOL)

print("Running Backtest...")
bt = Backtest(data, TrendFollower, cash=CASH_TO_INVEST, commission=.002) # 0.2% fee per trade

# Run the simulation
stats = bt.run()

# Show the results
print(stats)

# OPTIONAL: This opens a browser graph (uncomment to see it)
# bt.plot()