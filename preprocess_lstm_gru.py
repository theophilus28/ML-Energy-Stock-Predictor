"""
lstm and gru preprocessing script

this script provides functions to preprocess energy sector etf data for 
recurrent neural network architectures (lstm and gru). the preprocessing
transforms raw csv time series data into 3d sequences suitable for temporal
pattern learning.

output format: (samples, timesteps, features)
- samples: number of training/testing examples
- timesteps: number of historical days used to predict next day
- features: number of input variables per day (6: open, high, low, close, adj close, volume)

the script handles:
- loading raw csv data from subsector directories
- filtering data by training window (5, 6, 7, or 8 years)
- feature scaling using min-max normalization
- sequence generation with sliding window approach
- train/test split (2016-2023 for training, 2024 for testing)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_lstm_gru(ticker_path, training_years=5, sequence_length=60):
    """
    load csv data and preprocess into 3d sequences for lstm/gru models.
    
    this function performs the complete preprocessing pipeline:
    1. loads raw csv from relative path
    2. extracts relevant features (ohlcv + adj close)
    3. filters data based on training window size
    4. scales features to [0, 1] range for neural network training
    5. creates sequences using sliding window approach
    6. splits into training and testing sets
    
    parameters:
        ticker_path (str): relative path to csv file (e.g., 'raw_oil_and_gas/crak.csv')
        training_years (int): number of years for training window (5, 6, 7, or 8)
                              5 = 2019-2023, 6 = 2018-2023, 7 = 2017-2023, 8 = 2016-2023
        sequence_length (int): number of historical days to use for prediction (default: 60)
                               each sample will contain 60 consecutive days to predict day 61
    
    returns:
        tuple: (x_train, y_train, x_test, y_test, scaler)
            x_train (numpy.ndarray): training sequences, shape (n_train_samples, sequence_length, 6)
            y_train (numpy.ndarray): training targets, shape (n_train_samples, 1)
            x_test (numpy.ndarray): testing sequences, shape (n_test_samples, sequence_length, 6)
            y_test (numpy.ndarray): testing targets, shape (n_test_samples, 1)
            scaler (MinMaxScaler): fitted scaler object for potential inverse transformation
    
    example:
        x_train, y_train, x_test, y_test, scaler = load_and_preprocess_lstm_gru(
            'raw_oil_and_gas/crak.csv', 
            training_years=5, 
            sequence_length=60
        )
    """
    
    # load csv data with date as index
    df = pd.read_csv(ticker_path, index_col='Date', parse_dates=True)
    
    # select features for model input
    # using all available price and volume data
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df = df[feature_columns]
    
    # remove any rows with missing values
    df = df.dropna()
    
    # determine training window start date based on training_years parameter
    # all windows end at 2023-12-31, test period is 2024
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
    train_data = df.loc[training_start:training_end].copy()
    test_data = df.loc[test_start:test_end].copy()
    
    # initialize and fit scaler on training data only
    # this prevents data leakage from test set
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # create sequences using sliding window approach
    # for each sequence, use sequence_length days to predict the next day's adjusted close
    x_train, y_train = create_sequences(train_scaled, sequence_length, target_col_index=4)
    x_test, y_test = create_sequences(test_scaled, sequence_length, target_col_index=4)
    
    return x_train, y_train, x_test, y_test, scaler


def create_sequences(data, sequence_length, target_col_index):
    """
    generate sequences for time series prediction using sliding window.
    
    this function transforms scaled data into overlapping sequences where
    each sequence contains sequence_length consecutive days of data and
    the target is the adjusted close price of the following day.
    
    sliding window example (sequence_length=3):
        day 0, day 1, day 2 -> predict day 3
        day 1, day 2, day 3 -> predict day 4
        day 2, day 3, day 4 -> predict day 5
    
    parameters:
        data (numpy.ndarray): scaled feature data, shape (n_days, n_features)
        sequence_length (int): number of historical days per sequence
        target_col_index (int): column index for target variable (4 = adj close)
    
    returns:
        tuple: (x, y)
            x (numpy.ndarray): input sequences, shape (n_sequences, sequence_length, n_features)
            y (numpy.ndarray): target values, shape (n_sequences, 1)
    """
    
    x = []
    y = []
    
    # iterate through data creating overlapping sequences
    # stop at len(data) - sequence_length to ensure we always have a target
    for i in range(len(data) - sequence_length):
        # extract sequence_length consecutive days
        sequence = data[i:i + sequence_length]
        
        # extract target (next day's adjusted close price)
        target = data[i + sequence_length, target_col_index]
        
        x.append(sequence)
        y.append(target)
    
    # convert lists to numpy arrays with proper shapes
    x = np.array(x)
    y = np.array(y).reshape(-1, 1)
    
    return x, y


def load_all_tickers_lstm_gru(training_years=5, sequence_length=60):
    """
    load and preprocess all etf tickers for lstm/gru training.
    
    this convenience function processes all 8 etf tickers across the three
    subsectors, returning a dictionary organized by subsector and ticker.
    useful for batch processing or training separate models per ticker.
    
    parameters:
        training_years (int): number of years for training window (5, 6, 7, or 8)
        sequence_length (int): number of historical days per sequence (default: 60)
    
    returns:
        dict: nested dictionary structure:
              {
                  'oil_and_gas': {
                      'crak': (x_train, y_train, x_test, y_test, scaler),
                      'pxe': (x_train, y_train, x_test, y_test, scaler),
                      'fcg': (x_train, y_train, x_test, y_test, scaler)
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
            x_train, y_train, x_test, y_test, scaler = load_and_preprocess_lstm_gru(
                ticker_path,
                training_years=training_years,
                sequence_length=sequence_length
            )
            
            # store in nested dictionary
            all_data[subsector][ticker] = (x_train, y_train, x_test, y_test, scaler)
    
    return all_data
