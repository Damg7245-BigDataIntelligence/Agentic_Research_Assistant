import re
import pandas as pd
import yfinance as yf
import snowflake.connector
from dotenv import load_dotenv
from snowflake.connector.pandas_tools import write_pandas
import os
import json
from pathlib import Path
import numpy as np
load_dotenv()

def create_daily_historical_report(ticker="NVDA", period="5y", output_file=None):
    """
    Create a report with daily historical data and technical indicators
    """
    print(f"🔍 Fetching daily historical data for {ticker} over {period}...")
    
    # Create output directory if it doesn't exist
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not output_file:
        output_file = output_dir / f"{ticker}_daily_historical.csv"
    
    try:
        # Get ticker object
        ticker_obj = yf.Ticker(ticker)
        
        # Get detailed historical data with all available columns
        hist_data = ticker_obj.history(period=period, auto_adjust=True)
        
        # Reset index to make Date a column
        df = hist_data.reset_index()
        
        # Add ticker column as the first column
        df.insert(0, 'Ticker', ticker)
        
        # Some additional calculated columns that change daily
        if 'Close' in df.columns and 'Open' in df.columns:
            df['DailyChange'] = df['Close'] - df['Open']
            df['DailyChangePercent'] = (df['Close'] / df['Open'] - 1) * 100
        
        if 'Volume' in df.columns and 'Close' in df.columns:
            df['DollarVolume'] = df['Volume'] * df['Close']
        
        # Calculate 10-day and 30-day moving averages
        # Handle NaN values for initial periods by filling with the value itself
        if 'Close' in df.columns:
            df['MA10'] = df['Close'].rolling(window=10, min_periods=1).mean()
            df['MA30'] = df['Close'].rolling(window=30, min_periods=1).mean()
        
        # Calculate volatility (standard deviation of returns over 20 days)
        if 'Close' in df.columns:
            # Calculate returns first
            df['Returns'] = df['Close'].pct_change()
            
            # Handle initial NaN value in Returns
            df['Returns'] = df['Returns'].fillna(0)
            
            # Calculate volatility with min_periods=1 to handle initial values
            df['Volatility20D'] = df['Returns'].rolling(window=20, min_periods=1).std() * (252 ** 0.5)
        
        # Calculate Relative Strength Index (RSI)
        if 'Close' in df.columns:
            delta = df['Close'].diff()
            # Handle initial NaN value
            delta = delta.fillna(0)
            
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            
            # Instead of ewm which produces initial NaNs, use rolling with expanding
            # for the initial periods
            up_mean = up.rolling(window=14, min_periods=1).mean()
            down_mean = down.rolling(window=14, min_periods=1).mean()
            
            # Avoid division by zero
            down_mean = down_mean.replace(0, np.finfo(float).eps)
            
            rs = up_mean / down_mean
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # Remove intermediate calculation columns
        if 'Returns' in df.columns:
            df = df.drop('Returns', axis=1)
        
        # Remove Dividends and Stock Splits columns if they exist
        if 'Dividends' in df.columns:
            df = df.drop('Dividends', axis=1)
        
        if 'Stock Splits' in df.columns:
            df = df.drop('Stock Splits', axis=1)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        # Summary
        print(f"✅ Created daily historical report for {ticker} with {len(df)} rows and {len(df.columns)} columns")
        print(f"📊 Report saved to: {output_file}")
        print(f"📈 Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        return str(output_file)
        
    except Exception as e:
        print(f"❌ Error creating report: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Example usage
if __name__ == "__main__":
    create_daily_historical_report("NVDA", "5y")