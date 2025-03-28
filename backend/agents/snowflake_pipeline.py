import yfinance as yf
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import io
import snowflake.connector
import os

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
        
        return df
        
    except Exception as e:
        print(f"❌ Error creating report: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def upload_csv_to_s3(df):
    # Convert DataFrame to CSV in memory (without index)
    output_buffer = io.StringIO()
    df.to_csv(output_buffer, index=False)
    output_buffer.seek(0)  # Go to the beginning of the StringIO object

    # Generate the filename with timestamp
    filename = f"nvidia_data.csv"

    # Upload the CSV to S3
    s3_utils.upload_file_to_s3(output_buffer.getvalue(), filename, "csvFile")

    print("File uploaded to s3")

def snowflake_connector():
    # Snowflake connection details
    SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")  # e.g. 'vwcoqxf-qtb83828'
    SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")  # Your Snowflake username
    SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")  # Your Snowflake password
    SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")  # Your role, e.g., 'SYSADMIN'

    # Connecting to Snowflake
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,        # This should be your username
        password=SNOWFLAKE_PASSWORD,       # This should be your password
        account=SNOWFLAKE_ACCOUNT,     # This should be your Snowflake account URL
        role=SNOWFLAKE_ROLE          # Optional, if you need to specify the role
    )

    cur = conn.cursor()
    
    # Create Warehouse (if it doesn't exist)
    cur.execute("""
        CREATE WAREHOUSE IF NOT EXISTS NVIDIA_DATA
        WAREHOUSE_SIZE = 'SMALL'
        AUTO_SUSPEND = 60
        AUTO_RESUME = TRUE;
    """)

    # Create Database (if it doesn't exist)
    cur.execute("""
        CREATE DATABASE IF NOT EXISTS NVIDIA_DB;
    """)

    cur.execute("USE DATABASE NVIDIA_DB;")  # Specify the database

    # Create Schema (if it doesn't exist)
    cur.execute("""
        CREATE SCHEMA IF NOT EXISTS NVIDIA_SCHEMA;
    """)

    cur.execute("USE SCHEMA NVIDIA_DB.NVIDIA_SCHEMA;")  # Specify the schema

    # Create Storage Integration
    def create_storage_integration(cur):
        cur.execute("""
            CREATE STORAGE INTEGRATION IF NOT EXISTS nvidia_integration
            TYPE = 'EXTERNAL_STAGE'
            STORAGE_PROVIDER = 'S3'
            ENABLED = TRUE
            STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::699475925561:role/nvidia_db_snowflake_connection'
            STORAGE_ALLOWED_LOCATIONS = ('s3://nvidia-agentic-assistant/');
        """)

    # Create CSV Format
    def create_csv_format(cur):
        cur.execute("""
            CREATE OR REPLACE FILE FORMAT NVIDIA_CSV_FORMAT
            TYPE = 'CSV'
            FIELD_OPTIONALLY_ENCLOSED_BY = '"'
            SKIP_HEADER = 1; -- Ensures the first row is treated as column headers

        """)

    # Create Stage
    def create_stage(cur):
        cur.execute("""
            CREATE STAGE IF NOT EXISTS NVIDIA_STAGE
            URL = 's3://nvidia-agentic-assistant/csvFile/'
            STORAGE_INTEGRATION = nvidia_integration
            FILE_FORMAT = (FORMAT_NAME = 'NVIDIA_CSV_FORMAT');
        """)

    # Create Table by Inferring Schema
    def create_table(cur):
        cur.execute("""
            CREATE OR REPLACE TABLE NVIDIA_FIN_DATA (
                Ticker STRING,
                Date TIMESTAMP,
                Open FLOAT,
                High FLOAT,
                Low FLOAT,
                Close FLOAT,
                Volume INT,
                DailyChange FLOAT,
                DailyChangePercent FLOAT,
                DollarVolume FLOAT,
                MA10 FLOAT,
                MA30 FLOAT,
                Volatility20D FLOAT,
                RSI FLOAT,
                Year INT, 
                Quarter INT
            );
        """)

    # Load Data into Snowflake Table from Stage
    def load_data_into_snowflake(cur):
        cur.execute("""
            COPY INTO NVIDIA_FIN_DATA
            FROM @NVIDIA_STAGE
            FILES = ('nvidia_data.csv')
            FILE_FORMAT = (FORMAT_NAME = 'NVIDIA_CSV_FORMAT')
        """)

    # Calling functions
    create_storage_integration(cur)
    create_csv_format(cur)
    create_stage(cur)
    create_table(cur)
    load_data_into_snowflake(cur)

    conn.commit()
    cur.close()
    conn.close()

    


# Example usage
if __name__ == "__main__":
    df = create_daily_historical_report("NVDA", "5y")
    # Create 'year' column by extracting the year from 'Date'
    df['Year'] = df['Date'].dt.year

    # Create 'quarter' column by extracting the quarter from 'Date'
    df['Quarter'] = df['Date'].dt.quarter
    print(len(df), type(df), df.columns)
    upload_csv_to_s3(df)
    snowflake_connector()
