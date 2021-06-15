import numpy as np
import pandas as pd
import joblib
import os
from os.path import join, isfile, isdir
from sklearn.preprocessing import MinMaxScaler

def prepare_data(trainPath, testPath, destPath, unwantedFeatures, missing='remove'): 
    # Find and store scaler
    scale = find_scale(trainPath, unwantedFeatures, missing)
    # Scaler is identical for all models because we use min-max scaling.
    joblib.dump(scale, f'scaler_{missing}.pkl')
    
    # Split, scale and save datasets into subfiles
    save_files(trainPath, destPath, missing, True)
    save_files(testPath, destPath, missing, False)


def save_files(source, destPath, missing, isTraining):
    folder = 'test/'
    if isTraining:
        folder = 'training/'
    destPath = join(destPath, missing, folder)    
    ensure_path(destPath)
    empty_folder(destPath)
    
    # Locate all files
    files = [f for f in os.listdir(source) if isfile(join(source, f))]
    
    # Process all train files
    for f in files:
        file = pd.read_csv(source + f)
        
        # Split file into subfiles
        if missing == 'remove':
            grps = file.isna().any(axis=1).cumsum()
            file = file.dropna()
            subfiles = file.groupby(grps)
            subfiles = [subfile for _, subfile in subfiles]
        else:
            subfiles = [replace_missing(file, missing)]
        
        fileName = join(destPath, f.split('.')[0] + '_')
        for i, subfile in enumerate(subfiles):
            subfile.to_csv(fileName + str(i) + '.csv', index=False)

            
def replace_missing(df, arg):
    """Replace missing values in a dataset.
    
    Args:
        df (pandas.DataFrame): Dataset containing missing values.
        arg (str): How to replace missing values.

    Returns:
        pandas.DataFrame: Dataset without missing values
    """
    if arg == 'remove':
        df = df.dropna()
    elif arg == 'drop':
        df = df.dropna()
    elif arg == 'zero':
        df = df.replace(np.nan, 0)       
    elif arg == 'median':
        for col in df:
            df[col] = df[col].replace(np.nan, df[col].median())
    elif arg == 'mean':
        for col in df:
            df[col] = df[col].replace(np.nan, df[col].mean())
    elif arg == 'interpolation':
        df = df.interpolate(limit_area='inside')
        df = df.dropna()

    return df


def find_scale(trainPath, unwantedFeatures, missing='remove', verbose=False):
    # Locate all files
    trainFiles = [f for f in os.listdir(trainPath) if isfile(join(trainPath, f))]
    
    # Create dataframe from first file
    df = pd.read_csv(trainPath+'/'+trainFiles[0])
    df = drop_features(df, unwantedFeatures)
    df = replace_missing(df, missing)
    df = transform_features(df)
    
    # Append other files to dataframe 
    for f in range(1,len(trainFiles)):
        file = pd.read_csv(trainPath+'/'+trainFiles[f])
        file = drop_features(file, unwantedFeatures)
        file = replace_missing(file, missing)
        file = transform_features(file)
        df = df.append(file)
    
    # Find scaler from entire dataframe
    scale = MinMaxScaler().fit(df.values)

    if verbose:
        print(df.quantile([0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.999]))
        for column in df:
            print(column + ":")
            print('\t min: ' + '{:1.2f}'.format(min(df[column])))
            print('\t avg: ' + '{:1.2f}'.format(np.mean(df[column])))
            print('\t max: ' + '{:1.2f}'.format(max(df[column])))
    
    return scale


def scale_data(df, scale):
    scaled_features = scale.transform(df.values)
    df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
    return df


def drop_features(dataset, features):
    df = dataset.copy()
    for feature in features:
        if feature in df:
            del df[feature]
    return df


def transform_features(dataset):
    df = dataset.copy()
    
    if 'RMED' in df:
        df['RMED'] = np.log10(df['RMED'])
    
    if 'RDEP' in df:
        df['RDEP'] = np.log10(df['RDEP'])
        
    if 'BS' in df and 'CALI' in df:
        df['CALI-BS'] = df['CALI'] - df['BS']
        del df['CALI']
        del df['BS']
    
    return df
    
    
def ensure_path(source):
    if not isdir(source):
        os.makedirs(source)

        
def empty_folder(source):
    if not isdir(source):
        return
    
    files = [join(source,f) for f in os.listdir(source) if isfile(join(source, f))]
    for file in files:
        if isfile(file):
            os.remove(file)