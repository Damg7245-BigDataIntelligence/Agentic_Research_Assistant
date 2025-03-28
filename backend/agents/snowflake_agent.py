import snowflake.connector
from dotenv import load_dotenv
import google.generativeai as genai
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import uuid
import io
from datetime import datetime
from s3_utils import upload_visualization_to_s3
import numpy as np
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_snowflake_response(query, year_quarter_dict):

    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

    prompt = f"""

    I have a table in Snowflake that contains financial data for NVidia. This table records information for each day with different columns that represent various financial metrics.
    
    **Input Table: NVIDIA_FIN_DATA**  
    Below is a brief description of each column:
    - `DATE TIMESTAMP_NTZ`: The timestamp of the financial record, indicating the specific day.
    - `OPEN FLOAT`: The opening price of the stock on that day.
    - `DAILYCHANGE FLOAT`: The absolute change in the stock price compared to the previous day.
    - `MA10 FLOAT`: The 10-day moving average of the stock's price.
    - `HIGH FLOAT`: The highest stock price recorded on that day.
    - `CLOSE FLOAT`: The closing price of the stock on that day.
    - `RSI FLOAT`: The Relative Strength Index, a technical indicator that measures the speed and change of price movements (used for determining overbought/oversold conditions).
    - `VOLUME NUMBER`: The number of shares traded on that day.
    - `DAILYCHANGEPERCENT FLOAT`: The percentage change in the stock's price compared to the previous day.
    - `TICKER TEXT`: The stock symbol or identifier for the stock being traded.
    - `DOLLARVOLUME FLOAT`: The total dollar volume of stocks traded (calculated as the stock price multiplied by the trading volume).
    - `LOW FLOAT`: The lowest stock price recorded on that day.
    - `MA30 FLOAT`: The 30-day moving average of the stock's price.
    - `VOLATILITY20D FLOAT`: The 20-day volatility of the stock's price, indicating how much the price fluctuates over the past 20 days.
    - `Year INT`: The year of the financial record.
    - `Quarter INT`: The quarter of the financial record.

    **Important Notes for Gemini:**
    - Identify the relevant columns from the provided metadata based on the user's query.
    - Generate the appropriate SQL query that will fetch the relevant data to answer the user's query.
    - Make sure to consider: The specific financial metric(s) being asked (e.g., revenue, net income)
    - I have already added `Year` and `Quarter` as separate columns in the table.  
    - The filtering should be done **directly** using these columns, **without** needing to extract them from the `DATE` column.  
    - The user will provide a dictionary containing `Year` and `Quarter`, which should be used for filtering.  
    - Your main task is to generate the required SQL queries based on the user's request and correctly identify the relevant column(s).

    **User Query:**  
    {query}

    **Time Duration:**  
    The user will specify the time frame using a dictionary containing `Year` and `Quarter`.  
    Example: `{year_quarter_dict}`

    **Task for Gemini:**  
    Based on the user's query, generate **two separate SQL queries**:

    ### **1. Aggregated Query (Summing financial metrics like DOLLARVOLUME)**
    - This query should aggregate the specified metric (e.g., `SUM(DOLLARVOLUME)`) over the relevant time periods (specific quarters or years).
    - Use the `Year` and `Quarter` columns for filtering.

    ### **2. Raw Data Query (Without Aggregation)**
    - This query should retrieve individual record/records (financial metrics that is relevant to the user query, e.g., `(DOLLARVOLUME)`) along with date, year, quater without aggregation.
    - It should filter based on `Year` and `Quarter`.

    **Format of Response:**
    1. **Query 1: Aggregated Query**, followed by the SQL code.
    2. **Query 2: Raw Data Query**, followed by the SQL code.
    
    - Ensure proper formatting and structuring of the queries.
    - The explanation should follow after the SQL code, not between the queries.

"""
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-pro")
    response = gemini_model.generate_content(prompt)
    # Call Gemini API (example, your logic here might differ)
    #print(response.text)

    return response.text.strip()


def fetch_snowflake_df(query):
    # Snowflake connection details
    SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
    SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
    SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
    SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")

    # Connecting to Snowflake
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        role=SNOWFLAKE_ROLE
    )

    cur = conn.cursor()
    cur.execute("USE DATABASE NVIDIA_DB;")
    cur.execute("USE SCHEMA NVIDIA_DB.NVIDIA_SCHEMA;")

    # Improved regex pattern to handle more SQL query formats
    columns_pattern = re.search(r"SELECT\s+([\w\s,.*()]+?)\s+FROM", query, re.IGNORECASE | re.MULTILINE)
    
    try:
        # Execute the query first to get actual column names
        cur.execute(query)
        results = cur.fetchall()
        
        # Get column names from cursor description (this is more reliable)
        column_names = [col[0] for col in cur.description]
        
        # Create DataFrame with the correct column names
        df = pd.DataFrame(results, columns=column_names)
        return df

    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    finally:
        cur.close()
        conn.close()

def create_and_save_graph(df, query, timestamp, metadata_filters=None):
    """Create focused visualizations based on query relevance."""
    try:
        # Ensure 'Date' column is properly formatted
        df['Date'] = pd.to_datetime(df['DATE'])
        
        # Get relevant columns with fallback options
        relevant_columns = get_relevant_columns(query, df.columns, metadata_filters)
        
        # Create visualization folder path with timestamp
        viz_folder = f"visualizations/temp/query_{timestamp}"
        
        visualizations = []
        
        # 1. Create time series plot for main metrics
        plt.figure(figsize=(12, 6))
        for col in relevant_columns[:3]:  # Limit to top 3 most relevant metrics
            plt.plot(df['Date'], df[col], label=col)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f'NVIDIA Key Metrics: {", ".join(relevant_columns[:3])}')
        plt.legend()
        plt.grid(True)
        
        # Save and upload time series plot
        buffer1 = io.BytesIO()
        plt.savefig(buffer1, format='png')
        buffer1.seek(0)
        plt.close()
        
        ts_url = upload_visualization_to_s3(
            image_data=buffer1.getvalue(),
            prefix=f"{viz_folder}/time_series",
            filename="time_series.png"
        )
        
        visualizations.append({
            "url": ts_url,
            "type": "time_series",
            "title": "Key Metrics Time Series",
            "columns": relevant_columns[:3]
        })
        
        # 2. Create correlation heatmap if we have multiple relevant columns
        if len(relevant_columns) > 1:
            plt.figure(figsize=(10, 8))
            correlation = df[relevant_columns].corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation between Key Metrics')
            
            buffer2 = io.BytesIO()
            plt.savefig(buffer2, format='png')
            buffer2.seek(0)
            plt.close()
            
            corr_url = upload_visualization_to_s3(
                image_data=buffer2.getvalue(),
                prefix=f"{viz_folder}/correlation",
                filename="correlation.png"
            )
            
            visualizations.append({
                "url": corr_url,
                "type": "correlation",
                "title": "Metrics Correlation Analysis",
                "columns": relevant_columns
            })
        
        return visualizations
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return []

def get_relevant_columns(query, available_columns, metadata_filters=None):
    """Use LLM to identify relevant columns for visualization with fallback."""
    prompt = f"""
    Given this query about NVIDIA financial data:
    "{query}"
    
    And these available columns:
    {[col for col in available_columns if col not in ['DATE', 'Date', 'Year', 'Quarter']]}
    
    List only the 3-4 most relevant column names that would best answer this query.
    Return only the column names separated by commas, nothing else.
    """
    
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel("gemini-1.5-pro")
    response = gemini_model.generate_content(prompt)
    
    # Extract column names and clean them
    columns = [col.strip() for col in response.text.split(',')]
    relevant_cols = [col for col in columns if col in available_columns]
    
    # If no relevant columns found or columns don't make sense for the query,
    # use default visualization columns based on time period
    if not relevant_cols or all(col in ['Year', 'Quarter', 'DATE'] for col in relevant_cols):
        print("No specific columns identified from query, using default visualization columns")
        
        # Default visualization set 1: Price and Volume trends
        price_cols = ['HIGH', 'LOW', 'CLOSE', 'DOLLARVOLUME']
        # Default visualization set 2: Technical indicators
        tech_cols = ['MA10', 'MA30', 'RSI', 'VOLATILITY20D']
        
        # Check which columns are available and use them
        available_price_cols = [col for col in price_cols if col in available_columns]
        available_tech_cols = [col for col in tech_cols if col in available_columns]
        
        # Use price columns as primary fallback
        if available_price_cols:
            relevant_cols = available_price_cols[:3]  # Limit to 3 columns
            print(f"Using default price-based columns: {relevant_cols}")
        # Use technical indicators as secondary fallback
        elif available_tech_cols:
            relevant_cols = available_tech_cols[:3]
            print(f"Using default technical indicator columns: {relevant_cols}")
    
    return relevant_cols

year_quarter_dict = {
    "2024": ["1", "2", "3"],
    "2023": ["2", "4"]
}

query = "What is the MA10 value for a specific stock (TICKER) on a given date?"

input_string = fetch_snowflake_response(query,year_quarter_dict)

queries = re.findall(r"(SELECT[\s\S]*?);", input_string)

# Store the queries in variables
agg_query = queries[0] if len(queries) > 0 else None
raw_query = queries[1] if len(queries) > 1 else None

#fetch_snowflake_df(agg_query)
dataframe = fetch_snowflake_df(raw_query)
create_and_save_graph(dataframe, query, datetime.now().strftime('%Y%m%d%H%M%S'))

def generate_snowflake_insights(query, year_quarter_dict):
    """Main function to generate insights from Snowflake data."""
    try:
        # Get SQL queries based on user question
        llm_response = fetch_snowflake_response(query, year_quarter_dict)
        queries = re.findall(r"(SELECT[\s\S]*?);", llm_response)
        
        # Extract queries
        agg_query = queries[0] if len(queries) > 0 else None
        raw_query = queries[1] if len(queries) > 1 else None
        
        # Execute queries and get data
        agg_df = fetch_snowflake_df(agg_query) if agg_query else None
        raw_df = fetch_snowflake_df(raw_query) if raw_query else None
        
        # Generate visualization if we have raw data
        visualizations = []
        if raw_df is not None and not raw_df.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            visualizations = create_and_save_graph(raw_df, query, timestamp)
            
            print("\n" + "="*80)
            print("🖼️ VISUALIZATION DEBUG")
            print("="*80)
            print(f"Number of visualizations generated: {len(visualizations)}")
            for idx, viz in enumerate(visualizations, 1):
                print(f"\nVisualization {idx}:")
                print(f"Type: {viz['type']}")
                print(f"Title: {viz['title']}")
                print(f"Columns: {viz['columns']}")
                print(f"URL: {viz['url']}")
            print("="*80 + "\n")
        
        # Generate summary of the data
        summary = generate_data_summary(query, agg_df, raw_df)
        
        result = {
            "summary": summary,
            "visualizations": visualizations,
            "raw_data": raw_df.head(5).to_dict(orient="records") if raw_df is not None else []
        }
        
        print("\n" + "="*80)
        print("📊 SNOWFLAKE INSIGHTS RESULT")
        print("="*80)
        print("Summary preview:", summary[:200], "...")
        print(f"\nVisualization count: {len(visualizations)}")
        print(f"Raw data samples: {len(result['raw_data'])} rows")
        print("="*80 + "\n")
        
        return result
    except Exception as e:
        print(f"Error in generate_snowflake_insights: {e}")
        return {
            "summary": f"Error generating insights: {str(e)}",
            "visualizations": [],
            "raw_data": []
        }

def create_intelligent_visualizations(df, query):
    """
    Create appropriate visualizations based on data characteristics 
    and user query, handling scaling issues intelligently.
    """
    viz_urls = []
    
    # Determine which columns to visualize based on the query
    # Extract key terms from query
    query_terms = set(query.lower().split())
    
    # Identify numeric columns that aren't DATE, Year, or Quarter
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['Year', 'Quarter']]
    
    # Ensure 'DATE' column is datetime
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Find columns that match query terms (prioritize these)
    relevant_cols = []
    for col in numeric_cols:
        if col.lower() in query_terms or any(term in col.lower() for term in query_terms):
            relevant_cols.append(col)
    
    # If no columns matched query terms, use all numeric columns
    if not relevant_cols:
        relevant_cols = numeric_cols
    
    # Group columns by scale to avoid plotting issues
    if len(relevant_cols) > 1:
        # Calculate mean values for each column to determine scale
        col_means = {col: df[col].mean() for col in relevant_cols}
        
        # Group columns by order of magnitude
        scale_groups = {}
        for col, mean in col_means.items():
            if mean == 0:
                magnitude = 0
            else:
                magnitude = int(np.log10(abs(mean)))
            
            if magnitude not in scale_groups:
                scale_groups[magnitude] = []
            scale_groups[magnitude].append(col)
        
        # Create separate visualizations for each scale group
        for magnitude, cols in scale_groups.items():
            if cols:
                viz_url = create_and_upload_visualization(df, cols, f"magnitude_{magnitude}")
                viz_urls.append({
                    "url": viz_url,
                    "type": "line_chart",
                    "title": f"Metrics with similar scale (10^{magnitude})",
                    "columns": cols
                })
    else:
        # Just one column, create a single visualization
        viz_url = create_and_upload_visualization(df, relevant_cols, "single_metric")
        viz_urls.append({
            "url": viz_url,
            "type": "line_chart",
            "title": f"Time series for {relevant_cols[0]}",
            "columns": relevant_cols
        })
    
    return viz_urls

def create_and_upload_visualization(df, columns, chart_type):
    """Create visualization and upload to S3, returning the URL."""
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # If we have date column, use it for x-axis
    if 'DATE' in df.columns:
        x_column = 'DATE'
    else:
        # Use the first available index
        x_column = df.index
    
    # Plot each selected column
    for column in columns:
        plt.plot(df[x_column], df[column], label=column)
    
    # Add chart elements
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'NVIDIA Financial Metrics: {", ".join(columns)}')
    plt.legend()
    plt.grid(True)
    
    # Save to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    # Use the specialized function to upload and get URL
    viz_prefix = f"nvidia_{chart_type}"
    presigned_url = upload_visualization_to_s3(
        image_data=buffer.getvalue(),
        prefix=viz_prefix
    )
    
    return presigned_url

def generate_data_summary(query, agg_df, raw_df):
    """Generate a text summary of the data based on query and results."""
    # Prepare data for summarization
    agg_data = agg_df.to_dict(orient="records") if agg_df is not None else []
    sample_data = raw_df.head(5).to_dict(orient="records") if raw_df is not None else []
    
    # Describe the data's statistical properties
    stats = {}
    if raw_df is not None:
        numeric_cols = raw_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            stats[col] = {
                "min": raw_df[col].min(),
                "max": raw_df[col].max(),
                "avg": raw_df[col].mean(),
                "median": raw_df[col].median()
            }
    
    # Create a prompt for the LLM to summarize the data
    prompt = f"""
    As a financial analyst, summarize the following NVIDIA data in response to this query:
    
    QUERY: {query}
    
    AGGREGATED DATA: {agg_data}
    
    SAMPLE RAW DATA: {sample_data}
    
    STATISTICS: {stats}
    
    Provide a clear, concise summary focusing on key insights related to the query.
    Include notable trends, patterns, or outliers in the data.
    Use specific numbers from the data to support your analysis.
    """
    
    # Get summary from Gemini
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel("gemini-1.5-pro")
    response = gemini_model.generate_content(prompt)
    
    return response.text.strip()
