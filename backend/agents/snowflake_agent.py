import snowflake.connector
from dotenv import load_dotenv
import google.generativeai as genai
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Load environment variables
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", ".env"))
load_dotenv(dotenv_path)

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
    - Generate the appropriate SQL query that will fetch the relevant data to answer the user’s query.
    - Make sure to consider: The specific financial metric(s) being asked (e.g., revenue, net income)
    - I have already added `Year` and `Quarter` as separate columns in the table.  
    - The filtering should be done **directly** using these columns, **without** needing to extract them from the `DATE` column.  
    - The user will provide a dictionary containing `Year` and `Quarter`, which should be used for filtering.  
    - Your main task is to generate the required SQL queries based on the user’s request and correctly identify the relevant column(s).

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
    gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = gemini_model.generate_content(prompt)
    # Call Gemini API (example, your logic here might differ)
    #print(response.text)

    return response.text.strip()


def fetch_snowflake_df(query):
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
    cur.execute("USE DATABASE NVIDIA_DB;")  # Specify the database
    cur.execute("USE SCHEMA NVIDIA_DB.NVIDIA_SCHEMA;")  # Specify the schema

    #print(agg_query)
    columns = re.search(r"SELECT\s+(.*?)\s+FROM", query, re.IGNORECASE)

    # Split the extracted column names by commas and strip whitespace
    if columns:
        column_names = [col.strip() for col in columns.group(1).split(",")]
        print(column_names)
    else:
        print("No columns found.")

    cur.execute(query)
    results = cur.fetchall()  # Fetch all rows
    #print(results)
    print(type(results))
    df = pd.DataFrame(results, columns=column_names)
    print(df)
    return df

def create_and_save_graph(df, filename='graph.png'):
    # Ensure 'Date' column is properly formatted
    print(type(df['DATE']))
    df['Date'] = pd.to_datetime(df['DATE'])
    print("HI")
    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Plot all numeric columns except 'Year' and 'Quarter'
    numeric_columns = [col for col in df.columns if col not in ['DATE', 'Year', 'Quarter']]
    print(numeric_columns)
    # Plot each numeric column against 'Date'
    for column in numeric_columns:
        print(df['DATE'], df[column])
        plt.plot(df['DATE'], df[column], label=column)
    
    # Add labels, title, and legend
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title('Dynamic Graph')
    plt.legend()
    plt.grid()
    
    # Save the graph as a PNG file
    plt.savefig(filename)
    print(f"Graph saved as {filename}")


year_quarter_dict = {
    "2024": ["1", "2", "3"],
    "2023": ["2", "4"]
}

query = "What is the MA10 value for a specific stock (TICKER) on a given date?"

input_string = fetch_snowflake_response(query,year_quarter_dict)
print(input_string)

queries = re.findall(r"(SELECT[\s\S]*?);", input_string)

# Store the queries in variables
agg_query = queries[0] if len(queries) > 0 else None
raw_query = queries[1] if len(queries) > 1 else None

# Print the extracted queries
print("Aggregated Query:")
print(agg_query)

print("\nRaw Data Query:")
print(raw_query)

#fetch_snowflake_df(agg_query)
dataframe = fetch_snowflake_df(raw_query)
create_and_save_graph(dataframe)
