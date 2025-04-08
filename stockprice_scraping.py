import pandas as pd
import yfinance as yf
from datetime import datetime

def get_stock_data(tickers, start_date, end_date):
    """
    Fetches historical stock data including open, high, low, close, adjusted close, and volume
    for the specified tickers within a date range.

    Parameters:
    tickers (list): List of stock ticker symbols
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
    dict: Dictionary containing DataFrames for each data type (Open, High, Low, Close, Adj Close, Volume)
    """
    # Dictionary to map company names to their ticker symbols
    ticker_mapping = {
        'Meta': 'META',
        'Apple': 'AAPL',
        'Amazon': 'AMZN',
        'Netflix': 'NFLX',
        'Google': 'GOOGL'
    }

    # If tickers are provided as company names, convert them to ticker symbols
    ticker_symbols = []
    for ticker in tickers:
        if ticker in ticker_mapping:
            ticker_symbols.append(ticker_mapping[ticker])
        else:
            ticker_symbols.append(ticker)

    # Download data
    data = yf.download(ticker_symbols, start=start_date, end=end_date)

    # Create a dictionary with all available data types
    stock_data = {}
    for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if column in data:
            stock_data[column] = data[column]

    # Handle 'Adj Close' separately due to the space in the name
    if 'Adj Close' in data:
        stock_data['Adj_Close'] = data['Adj Close']

    return stock_data

def save_to_csv(data_dict, base_filename="stock_data"):
    """
    Saves each DataFrame in the data dictionary to a separate CSV file.

    Parameters:
    data_dict (dict): Dictionary of DataFrames to save
    base_filename (str): Base name for output files
    """
    for data_type, df in data_dict.items():
        filename = f"{base_filename}_{data_type.lower()}.csv"
        df.to_csv(filename)
        print(f"{data_type} data saved to {filename}")

def save_combined_data(data_dict, filename="all_stock_data.csv"):
    """
    Saves all data in a single multi-level CSV file.

    Parameters:
    data_dict (dict): Dictionary of DataFrames to save
    filename (str): Name of the output file
    """
    # Create a multi-level DataFrame
    combined_data = pd.concat(data_dict, axis=1)
    combined_data.to_csv(filename)
    print(f"All data saved to {filename}")

def main():
    # List of companies to fetch data for
    companies = ['Meta', 'Apple', 'Amazon', 'Netflix', 'Google']

    # Get user input for date range
    print("Enter the date range for stock data (YYYY-MM-DD format):")
    start_date = input("Start date: ")
    end_date = input("End date (leave empty for today's date): ")

    # Use today's date if end_date is not provided
    if not end_date:
        end_date = datetime.today().strftime('%Y-%m-%d')

    try:
        # Validate dates
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')

        # Get stock data
        stock_data = get_stock_data(companies, start_date, end_date)

        # Print summary of closing prices if available
        if 'Close' in stock_data:
            print("\nStock Closing Prices Summary (Last 5 days):")
            print(stock_data['Close'].tail())

        # Save options
        print("\nHow would you like to save the data?")
        print("1. Separate CSV files for each data type (Open, High, Low, etc.)")
        print("2. Single combined CSV file with all data")
        print("3. Both options")

        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            save_to_csv(stock_data)
        elif choice == '2':
            save_combined_data(stock_data)
        elif choice == '3':
            save_to_csv(stock_data)
            save_combined_data(stock_data)
        else:
            print("Invalid choice. Saving as separate files by default.")
            save_to_csv(stock_data)

        # Optional: Plot the data
        if 'Close' in stock_data:
            print("\nWould you like to plot the closing prices? (y/n)")
            if input().lower() == 'y':
                stock_data['Close'].plot(figsize=(12, 6), title="Stock Closing Prices")
                import matplotlib.pyplot as plt
                plt.ylabel("Price ($)")
                plt.grid(True)
                plt.show()

    except ValueError as e:
        print(f"Error: {e}. Please ensure dates are in YYYY-MM-DD format.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()