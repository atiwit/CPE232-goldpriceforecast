import yfinance as yf
import os

os.makedirs("data/raw", exist_ok=True)

tickers = {
    "gold":  "GC=F",
    "dxy":   "DX-Y.NYB",
    "vix":   "^VIX",
    "yield": "^TNX",
    "sp500": "^GSPC",
    "oil":    "CL=F", 
    
}

for name, ticker in tickers.items():
    df = yf.download(ticker, start="2015-01-01", end="2026-03-31")
    df.to_csv(f"data/raw/{name}_raw.csv")
    print(f"Saved {name}_raw.csv — {len(df)} rows")