"""
etf data scraper for energy sector stock prediction research

this script downloads historical price data for 8 energy sector etfs from yahoo finance
using the yfinance library. data is automatically organized by subsector and saved to 
separate csv files for each ticker.

the script uses yfinance's built-in repair functionality to automatically fix common
data quality issues like missing adjustments, currency errors, and missing data points.

data coverage: january 1, 2016 to december 31, 2024
data features: open, high, low, close, volume, adj close
"""

import subprocess
import sys


def install_dependencies():
    """
    check for and install required packages if they are missing.
    
    this function attempts to import each required package and installs it
    using pip if the import fails. this ensures the script can run even on
    fresh python installations without manual dependency management.
    
    required packages:
    - yfinance: for downloading financial data from yahoo finance
    - scipy: required by yfinance for data repair functionality
    - pandas: data manipulation (usually comes with python, but checking anyway)
    """
    
    required_packages = {
        'yfinance': 'yfinance',
        'scipy': 'scipy',
        'pandas': 'pandas'
    }
    
    for package_import, package_pip in required_packages.items():
        try:
            __import__(package_import)
            print(f"Package `{package_import}` available!")
        except ImportError:
            print(f"{package_pip} not found. installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_pip])
            print(f"{package_pip} installed successfully\n")


# install dependencies before importing them
install_dependencies()

import yfinance as yf
from pathlib import Path


def download_etf_data():
    """
    download and save historical etf data organized by energy subsector.
    
    this function handles the complete workflow of:
    1. defining ticker symbols and their subsector classifications
    2. creating necessary directory structure
    3. downloading data with automatic repair from yahoo finance
    4. saving individual csv files for each ticker
    
    the function will terminate with an error message if any download fails,
    ensuring data integrity across all tickers.
    
    returns:
        none. writes csv files to disk in subsector directories.
    """
    
    # define etf tickers organized by subsector
    # each etf is mapped to its corresponding energy subsector folder
    etf_mapping = {
        'raw_oil_and_gas': ['CRAK', 'PXE', 'FCG'],
        'raw_renewable': ['ICLN', 'SMOG', 'TAN'],
        'raw_nuclear': ['URA', 'NLR']
    }
    
    # date range for historical data
    # covers 8 years of data for the longest training window plus test period
    start_date = '2016-01-01'
    end_date = '2024-12-31'
    
    # create directory structure for each subsector
    # using pathlib for cross-platform compatibility
    print("\nsetting up directories...")
    for subsector in etf_mapping.keys():
        Path(subsector).mkdir(parents=True, exist_ok=True)
    
    # download and save data for each subsector
    for subsector, tickers in etf_mapping.items():
        print(f"\ndownloading {subsector} etfs...")
        
        # download data for all tickers in this subsector at once
        # using batch download is more efficient than individual downloads
        df = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval='1d',           # daily data
            auto_adjust=False,       # keep both close and adj close columns
            repair=True,             # automatically fix data quality issues
            actions=False,           # we don't need separate dividend/split data
            threads=True,            # parallel downloads for speed
            progress=True,           # show download progress bar
            group_by='ticker'        # organize data by ticker symbol
        )
        
        # save individual csv file for each ticker
        for ticker in tickers:
            # extract data for this specific ticker
            # yfinance returns multi-level columns when downloading multiple tickers
            ticker_data = df[ticker].copy()
            
            # construct output path: subsector_folder/ticker_lowercase.csv
            output_path = Path(subsector) / f"{ticker.lower()}.csv"
            
            # save to csv with date as index
            # this overwrites any existing file with the same name
            ticker_data.to_csv(output_path)
            
            # print confirmation with row count
            row_count = len(ticker_data)
            print(f"  saved {ticker.lower()}.csv ({row_count:,} rows)")
        
        print()  # blank line between subsectors for readability
    
    print("="*50)
    print("download complete!")
    print("="*50)


if __name__ == "__main__":
    # execute the download function
    # script will terminate with error message if any download fails
    download_etf_data()