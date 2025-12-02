"""
random forest preprocessing script

this script provides functions to preprocess energy sector etf data for
random forest regression models. the preprocessing transforms raw csv
time series data into 2d tabular format with engineered lag features.

output format: (samples, features)
- samples: number of training/testing examples (one per trading day)
- features: engineered features including current values and historical lags

the script handles:
- loading raw csv data from subsector directories
- filtering data by training window (5, 6, 7, or 8 years)
- creating lag features (historical values from previous days)
- generating technical indicators (moving averages, volatility measures)
- train/test split (2016-2023 for training, 2024 for testing)
- removing rows with nan values introduced by lag/rolling calculations

note: random forest is less sensitive to feature scaling than neural networks
or svm, so normalization is optional but can still improve performance
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_and_preprocess_random_forest(ticker_path, training_years=5, lag_periods=[1, 5, 10, 20]):
    """
    load csv data and preprocess into 2d tabular format for random forest.
    
    this function performs the complete preprocessing pipeline:
    1. loads raw csv from relative path
    2. extracts relevant features (ohlcv + adj close)
    3. creates lag features (previous day values)
    4. generates technical indicators (moving averages, returns, volatility)
    5. filters data based on training window size
    6. splits into training and testing sets
    
    parameters:
        ticker_path (str): relative path to csv file (e.g., 'raw_oil_and_gas/crak.csv')
        training_years (int): number of years for training window (5, 6, 7, or 8)
                              5 = 2019-2023, 6 = 2018-2023, 7 = 2017-2023, 8 = 2016-2023
        lag_periods (list): list of lag periods to create (e.g., [1, 5, 10, 20] means
                            create features for 1-day ago, 5-days ago, etc.)
    
    returns:
        tuple: (x_train, y_train, x_test, y_test, feature_names)
            x_train (pandas.DataFrame): training features, shape (n_train_samples, n_features)
            y_train (pandas.Series): training targets (adj close prices)
            x_test (pandas.DataFrame): testing features, shape (n_test_samples, n_features)
            y_test (pandas.Series): testing targets (adj close prices)
            feature_names (list): list of feature column names for interpretation
    
    example:
        x_train, y_train, x_test, y_test, features = load_and_preprocess_random_forest(
            'raw_oil_and_gas/crak.csv', 
            training_years=5,
            lag_periods=[1, 5, 10, 20]
        )
    """
    
    # load csv data with date as index
    df = pd.read_csv(ticker_path, index_col='Date', parse_dates=True)
    
    # select base features for model input
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df = df[feature_columns].copy()
    
    # create lag features for each base feature
    # this allows the model to use historical values as predictors
    for lag in lag_periods:
        for col in feature_columns:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # create technical indicators
    # these capture trends and patterns in the data
    
    # moving averages (5, 10, 20 day windows)
    for window in [5, 10, 20]:
        df[f'close_ma_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'volume_ma_{window}'] = df['Volume'].rolling(window=window).mean()
    
    # daily return (percent change from previous day)
    df['daily_return'] = df['Adj Close'].pct_change()
    
    # volatility (standard deviation of returns over 20 day window)
    df['volatility_20'] = df['daily_return'].rolling(window=20).std()
    
    # high-low range (measure of daily price movement)
    df['high_low_range'] = df['High'] - df['Low']
    
    # close relative to high-low range (where close falls in daily range)
    df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    # remove rows with nan values introduced by lag and rolling calculations
    df = df.dropna()
    
    # determine training window start date based on training_years parameter
    training_start_dates = {
        5: '2019-01-01',  # 2019-2023
        6: '2018-01-01',  # 2018-2023
        7: '2017-01-01',  # 2017-2023
        8: '2016-01-01'   # 2016-2023
    }
    
    training_start = training_start_dates[training_years]
    training_end = '2023-12-31'
    test_start = '2024-01-01'
    test_end = '2024-12-31'
    
    # split data into training and testing periods
    train_df = df.loc[training_start:training_end].copy()
    test_df = df.loc[test_start:test_end].copy()
    
    # separate features (x) from target (y)
    # target is the adjusted close price we want to predict
    target_col = 'Adj Close'
    
    # create feature matrix by dropping the target column
    x_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    x_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    # get feature names for later interpretation and feature importance analysis
    feature_names = x_train.columns.tolist()
    
    return x_train, y_train, x_test, y_test, feature_names


def load_all_tickers_random_forest(training_years=5, lag_periods=[1, 5, 10, 20]):
    """
    load and preprocess all etf tickers for random forest training.
    
    this convenience function processes all 8 etf tickers across the three
    subsectors, returning a dictionary organized by subsector and ticker.
    
    parameters:
        training_years (int): number of years for training window (5, 6, 7, or 8)
        lag_periods (list): list of lag periods to create for features
    
    returns:
        dict: nested dictionary structure:
              {
                  'oil_and_gas': {
                      'crak': (x_train, y_train, x_test, y_test, feature_names),
                      'pxe': (x_train, y_train, x_test, y_test, feature_names),
                      'fcg': (x_train, y_train, x_test, y_test, feature_names)
                  },
                  'renewable': {...},
                  'nuclear': {...}
              }
    """
    
    # define directory structure and tickers
    subsectors = {
        'oil_and_gas': ['crak', 'pxe', 'fcg'],
        'renewable': ['icln', 'smog', 'tan'],
        'nuclear': ['ura', 'nlr']
    }
    
    # dictionary to store preprocessed data for all tickers
    all_data = {}
    
    # iterate through each subsector
    for subsector, tickers in subsectors.items():
        all_data[subsector] = {}
        
        # process each ticker in the subsector
        for ticker in tickers:
            # construct relative path to csv file
            ticker_path = f'raw_{subsector}/{ticker}.csv'
            
            # preprocess this ticker's data
            x_train, y_train, x_test, y_test, features = load_and_preprocess_random_forest(
                ticker_path,
                training_years=training_years,
                lag_periods=lag_periods
            )
            
            # store in nested dictionary
            all_data[subsector][ticker] = (x_train, y_train, x_test, y_test, features)
    
    return all_data
