import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from os.path import isfile, join
import joblib
import master.util as util

def scale_and_shape(source, unwantedFeatures, LOOKBACK_SIZE=10):
    scale = joblib.load('scaler_remove.pkl')
    datasets = []
    short = 0

    # Locate all files
    files = [f for f in os.listdir(source) if isfile(join(source, f))]
    
    # Load and preprocess files
    for f in range(0,len(files)):
        df = pd.read_csv(join(source, files[f]))
        if len(df) <= LOOKBACK_SIZE:
            short += 1
            continue
        df = util.transform_features(df)
        df = util.drop_features(df, unwantedFeatures)
        df = util.scale_data(df, scale)
        
        _data_ = df.to_numpy(copy=True)
        X = np.zeros(shape=(df.shape[0]-LOOKBACK_SIZE,LOOKBACK_SIZE,df.shape[1]))
        Y = np.zeros(shape=(df.shape[0]-LOOKBACK_SIZE,df.shape[1]))
        for i in range(LOOKBACK_SIZE-1, df.shape[0]-1):
            Y[i-LOOKBACK_SIZE+1] = _data_[i+1]
            for j in range(i-LOOKBACK_SIZE+1, i+1):
                X[i-LOOKBACK_SIZE+1][LOOKBACK_SIZE-1-i+j] = _data_[j]
        if len(X) >= 10:
            datasets.append((X,Y))
        
    print("Number of files shorter than lookback size: " + str(short))
    return datasets


def scale_and_shape_test(datasets, unwantedFeatures, LOOKBACK_SIZE=10):
    scale = joblib.load('scaler_remove.pkl')
    files = []
    short = 0

    # Load and preprocess files
    for dataset in datasets:
        if len(dataset) <= LOOKBACK_SIZE:
            short += 1
            continue
        df = util.transform_features(dataset)
        df = util.drop_features(df, unwantedFeatures)
        df = util.scale_data(df, scale)
        
        _data_ = df.to_numpy(copy=True)
        X = np.zeros(shape=(df.shape[0]-LOOKBACK_SIZE,LOOKBACK_SIZE,df.shape[1]))
        Y = np.zeros(shape=(df.shape[0]-LOOKBACK_SIZE,df.shape[1]))
        for i in range(LOOKBACK_SIZE-1, df.shape[0]-1):
            Y[i-LOOKBACK_SIZE+1] = _data_[i+1]
            for j in range(i-LOOKBACK_SIZE+1, i+1):
                X[i-LOOKBACK_SIZE+1][LOOKBACK_SIZE-1-i+j] = _data_[j]
        files.append((X,Y))
        
    print("Number of files shorter than lookback size: " + str(short))
    return files