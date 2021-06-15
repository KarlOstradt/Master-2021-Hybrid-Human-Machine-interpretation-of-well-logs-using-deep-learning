import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import math
import os
from os.path import join, isfile, isdir
from sklearn.preprocessing import MinMaxScaler
import master.util as util 
    
def reshape(df, nLines):
    if nLines == 1:
        return df
    
    # Remove excess rows
    excess = len(df)%nLines
    
    if excess > 0:
        df = df[:-excess]
    
    # Original labels
    labels = list(df)
    org_len = len(labels)
    
    # Amount of columns in reshaped dataset
    new_dim = org_len * nLines
    
    # Assign arbitrary names to new columns
    for i in range(org_len,new_dim):
        labels.append(labels[i%org_len]+str(math.floor(i/org_len)+1))
    
    # Reshape dataframe
    df = df.values.reshape(-1,new_dim)
    df = pd.DataFrame(data=df,columns=labels)
    
    return df
    
    
def scale_and_shape(source, unwantedFeatures, missing, nLines=1):
    scale = joblib.load( f'scaler_{missing}.pkl')
    
    # Locate all files
    files = [f for f in os.listdir(source) if isfile(join(source, f))]
    
    # Create dataframe from first file
    df = pd.read_csv(join(source,files[0]))
    df = util.drop_features(df, unwantedFeatures)
    df = util.transform_features(df)
    df = util.scale_data(df, scale)
    df = reshape(df, nLines)
    df, val_df, _, _ = train_test_split(df, df, shuffle=True, test_size=0.1, random_state=1)
    
    # Process all train files
    for f in range(1,len(files)):
        file = pd.read_csv(source + files[f])
        if len(file) == 0:
            print(f"{f}, {files[f]}")
            continue
        if len(file) < 10:
            continue
        file = util.drop_features(file, unwantedFeatures)
        file = util.transform_features(file)
        file = util.scale_data(file, scale)
        file = reshape(file, nLines)
        file, val_file, _, _ = train_test_split(file, file, shuffle=True, test_size=0.1, random_state=1)
        df = df.append(file)
        val_df = val_df.append(val_file)
        
    return df, val_df
    

def scale_and_shape_test(datasets, unwantedFeatures, missing, nLines=1):
    scale = joblib.load( f'scaler_{missing}.pkl')
    
    # Locate all files
    files = []
    
    # Process all train files
    for dataset in datasets:
        file = util.drop_features(dataset, unwantedFeatures)
        file = util.transform_features(file)
        file = util.scale_data(file, scale)
        file = reshape(file, nLines)
        files.append(file)
        
    return files
    
    
    
    