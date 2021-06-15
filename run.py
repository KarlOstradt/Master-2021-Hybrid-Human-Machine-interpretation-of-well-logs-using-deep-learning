import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Comment out to enable GPU if available
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
from os.path import join, isfile, isdir
import time
import datetime
from collections import defaultdict
from pprint import pprint
import socket
import json
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix

from multiprocessing import Process

import joblib

import master.util as util
import master.models as models
import master.metrics as metrics
import master.deepant_processing as d
import master.vae_processing as v

import traceback

res_file = 'results.csv'
res_file_short = 'results_short.csv'

paths = {
    'Hostname': "path/to/source",
}

# DEPTH,AC,ACS,BS,CALI,DEN,DENC,GR,NEU,PEF,RDEP,RMED
# m,us/ft,us/ft,in,in,g/cm3,g/cm3,gAPI,m,,ohm.m,ohm.m,,
units = {
    'DEPTH': "m",
    'AC': "us/ft",
    'ACS': "us/ft",
    'BS': "in",
    'CALI': "in",
    'DEN': "g/cm3",
    'DENC': "g/cm3",
    'GR': "gAPI",
    'NEU': "m",
    'PEF': "",
    'RDEP': "ohm.m",
    'RMED': "ohm.m",
    'ed_deepant': "",
    'der_deepant': "",
    'poi': "",
    'ed_vae': "",
    'der_vae': "",
    'ed_lstm': "",
    'der_lstm': "",
    'ed_ae': "",
    'der_ae': "",
    'percentile': "",
    'labels': "",
}

def limit_gpu_memory():
    tf.config.list_physical_devices('GPU')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def get_path():
    hostname = socket.gethostname()
    return paths[hostname]


def get_config(training=True):
    source = 'config/'
    files = [join(source, f) for f in os.listdir(source) if isfile(join(source, f))]
    configs = []
    for file in files:
        try:
            with open(file, 'r') as f:
                conf = json.loads(f.read())
            configs.append(conf)
        except Exception as e:
            print(f'Could not read {file}')
            traceback.print_tb(e.__traceback__)
            traceback.print_stack()
    return configs


def map_datasets(source_datasets, source_org):
    org_names = [f for f in os.listdir(source_org) if isfile(join(source_org, f))]
    subfiles = [f for f in os.listdir(source_datasets) if isfile(join(source_datasets, f))]
    d = dict()
    for name in org_names:
        d[name] = list()
        for subfile in subfiles:
            if name.split('.')[0] == subfile.split('.')[0][:-2]:
                df = pd.read_csv(join(source_datasets, subfile))
                d[name].append(df)
            elif name.split('.')[0] == subfile.split('.')[0][:-3]:
                df = pd.read_csv(join(source_datasets, subfile))
                d[name].append(df)
        
    return d 


def save_figure(X, Y, xlab, ylab, title, figPath):
    fig = plt.figure()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if type(Y) == list:
        for y in Y:
            plt.plot(X, y)
    else: 
        plt.plot(X, Y)
        plt.legend(list(Y))
    plt.savefig(figPath + '.png', bbox_inches='tight')
    plt.savefig(figPath + '.pdf', bbox_inches='tight')


def save_figure_f1_mcc(X, Y, xlab, ylab, title, figPath):
    fig = plt.figure()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.plot(X, Y['F1 Lenient'], '.-', c='tab:blue')
    plt.plot(X, Y['MCC Lenient'], '.-', c='tab:orange')
    plt.plot(X, Y['F1 Downsampled'], '.--', c='tab:blue')
    plt.plot(X, Y['MCC Downsampled'], '.--', c='tab:orange')

    plt.legend(list(Y))
    plt.savefig(figPath + '.png', bbox_inches='tight')
    plt.savefig(figPath + '.pdf', bbox_inches='tight')

        
def get_labels(org_labels, unwanted_features):
    labels = org_labels.copy()
    for feature in unwanted_features:
        if feature in org_labels:
            labels.remove(feature)          
    return labels


def add_cols(df_org, columns):
    df = df_org.copy()
    for col in list(columns):
        df[col] = columns[col]
    return df


def add_poi(df_org, der_lists, p = 99.0):
    # Add points of interest
    temp = []
    for der in der_lists:
        x = np.array(der)
        x = x[~np.isnan(x)]
        temp.append(x)

    key = f'p{p}'
    stats = metrics.stats(temp, percentiles=[p])

    poi = []
    for i, der in enumerate(der_lists):
        x = np.array(der)

        # Local percentile (per subfile)
        x[x<stats['files'][i][key]] = 0
        x[x>=stats['files'][i][key]] = 0.5

        poi.append(x)

    output_file = pd.DataFrame(data={'poi': [np.nan]*len(df_org)})
    for p in poi:
        output_file = output_file.combine_first(pd.DataFrame(data={'poi': p}))

    return add_cols(df_org, output_file)


def add_percentiles(df_org, der_lists):
    temp = []
    for der in der_lists:
        x = np.array(der)
        x = x[~np.isnan(x)]
        temp.append(x)

    stats = metrics.stats(temp, percentiles=range(0, 101))

    percentiles = []
    for i, der in enumerate(der_lists):
        x = np.array(der)

        for p in range(0,100):
            key = f'p{p}'
            next_key = f'p{p+1}'
            lower = stats['files'][i][key]
            upper = stats['files'][i][next_key]

            # Local percentiles
            mask = (lower <= der) & (der < upper)

            # Overwrite values in a percentile bracket to the percentile p
            x[mask] = p

        x[der>=stats['files'][i]['p99']] = 99
        percentiles.append(x)

    output_file = pd.DataFrame(data={'percentile': [np.nan]*len(df_org)})
    for p in percentiles:
        output_file = output_file.combine_first(pd.DataFrame(data={'percentile': p}))

    return add_cols(df_org, output_file)


def add_labels(df_org, l=0.6, u=0.3):
    ranges = get_ranges(l,u)
    depth = df_org['DEPTH']
    labels = np.array([0]*len(depth))
    data = {}
    for r in ranges:
        mask = (r[0] <= depth) & (depth <= r[1])
        labels[mask] = 1
    data['labels'] = labels

    cols = pd.DataFrame(data=data)

    return add_cols(df_org, cols)


def find_units(col_names):
    units_list = []
    for col in col_names:
        units_list.append(units[col])
    return units_list


def add_units(file):
    with open(file, 'r') as f:
        data = f.readlines()
    with open(file, 'w') as f:
        f.write(data[0])
        col_names = find_units(data[0].rstrip().split(','))
        f.write(','.join(col_names)+'\n')
        f.writelines(data[1:])


def get_ranges(lower=1.0, upper=0.5):
    ranges = []

    points = [
        2587.3,
        2589.7,
        2591.5,
        2592.8,
        2600.3,
        2600.8,
        2602.5,
        2603.5,
        2604.4,
        2608.1,
        2617.7,
        2618.2,
        2624.7,
        2641.2,
        2645.0,
        2645.6,
        2659.3,
        2685.9,
        2688.5,
        2728.1,
        2728.7,
        2732.6,
        2733.3,
        2742.1,
        2744.0,
        2744.2,
        2749.6,
        2760.0,
        2768.0,
        2770.7,
        2774.3,
        2778.9,
        2780.0,
        2787.6,
        2788.7,
        2821.3,
        2822.9, # Skifer til kalk
        2853.8,
        3324.5, # sand blandt kalk, sjekk completion log
        3986.6, # Kalk til skifer
        4102.8,
        4105.1,
        4393.8,
        4411.6,
        4412.4,
        4413.2,
        4415.4,
        4419.8,
        4420.2,
        4422.0,
        4423.3,
        4431.5,
        4433.3,
        4507.4,
    ]
    for p in points:
        ranges.append((p-lower, p+upper))

    ranges.append((2574.0,2575.7))
    ranges.append((2575.7,2577.5))
    ranges.append((4158.3,4159.5))
    ranges.append((4200.0,4201.0))
    ranges.append((4201.8,4202.5))
    ranges.append((4215.6,4216.5))
    ranges.append((4220.3,4221.2))
    ranges.append((4228.9,4229.5))
    ranges.append((4231.4,4231.9))
    ranges.append((4232.3,4233.7))
    ranges.append((4235.5,4236.4))
    ranges.append((4257.5,4258.5))
    ranges.append((4265.1,4266.0))
    ranges.append((4269.6,4270.7))
    ranges.append((4282.8,4283.6))
    ranges.append((4288.3,4289.4))
    ranges.append((4355.0,4357.0))
    ranges.append((4360.1,4361.6))
    ranges.append((4367.4,4369.1))
    ranges.append((4370.3,4371.6))
    ranges.append((4394.0,4395.7))
    ranges.append((4397.6,4398.3))
    ranges.append((4415.9,4417.9))
    ranges.append((4446.0,4449.3))
    ranges.append((4453.2,4460.1))
    ranges.append((4461.2,4463.6))
    ranges.append((4497.8,4499.0))
    ranges.append((4506.4,4507.3))

    ranges.append((2613.6,2616.7))
    ranges.append((2725.7,2726.0))
    ranges.append((2751.8,2754.0))
    ranges.append((3580.8,3582.3))
    ranges.append((3924.0,3927.5))
    ranges.append((3967.1,3968.0))
    ranges.append((3992.6,3993.5))
    ranges.append((4013.3,4014.0))
    ranges.append((4016.4,4016.6))
    ranges.append((4146.7,4148.5))
    ranges.append((4210.9,4212.9))

    return ranges


def new_empty_row():
    rows = defaultdict(list)
    rows['filename'] = []
    rows['conf_no'] = []
    rows['model'] = []
    rows['LOOKBACK_SIZE'] = []
    rows['kernel_size'] = []
    rows['n_filters'] = []
    rows['padding'] = []
    rows['pool_size'] = []
    rows['dense_layer_size'] = []
    rows['dropout'] = []
    rows['optimizer'] = []
    rows['loss'] = []
    rows['epochs'] = []
    rows['intermediate_dims'] = []
    rows['latent_dim'] = []
    rows['lstm_neurons'] = []
    rows['batch_size'] = []

    return rows


def compute_metrics(conf_no, conf, org, org_filename, subfiles, LOOKBACK_SIZE, t=94, l=0.6, u=0.3):
    c_matrix = np.array([0,0,0,0])
    rows = new_empty_row()
    n_rows = len(subfiles) + 1
    for _ in range(n_rows):
        rows['filename'].append(org_filename)

        # Model configuration
        rows['conf_no'].append(conf_no)
        for arg, value in conf.items():
            rows[arg].append(value)

        # Validation configuration
        rows['threshold'].append(t)
        rows['lower_boundary'].append(l)
        rows['upper_boundary'].append(u)
        rows['downsample'].append(False)
    del rows['unwanted_features']

    for subfile_no, subfile in enumerate(subfiles):
        depth = np.array(subfile['DEPTH'])
        subfile_range = (depth[0],depth[-1])
        subset = org[(org['DEPTH'] >= subfile_range[0]) & (org['DEPTH'] <= subfile_range[1])]

        labels = np.array([0]*len(depth))
        ranges = get_ranges(l,u)
        for r in ranges:
            mask = (r[0] <= depth) & (depth <= r[1])
            labels[mask] = 1
        labels = np.delete(labels, 0) # Delete first 

        percentiles = subset['percentile']
        percentiles = percentiles.dropna()
        preds = np.array([0]*len(percentiles))
        preds[percentiles >= t] = 1

        # Subfile info
        rows['subfile'].append(subfile_no)
        rows['bottom_depth'].append(subfile_range[0])
        rows['top_depth'].append(subfile_range[1])

        # Subfile metrics
        tn, fp, fn, tp = confusion_matrix(labels[LOOKBACK_SIZE:], preds).ravel()
        c_matrix += [tn, fp, fn, tp]
        m = metrics.evaluation_metrics(tp,tn,fp,fn)
        for metric, value in m.items():
            rows[metric].append(value)
        rows['tp'].append(tp)
        rows['tn'].append(tn)
        rows['fp'].append(fp)
        rows['fn'].append(fn)

    # Entire file info
    rows['subfile'].append(np.nan)
    rows['bottom_depth'].append(rows['bottom_depth'][0])
    rows['top_depth'].append(rows['top_depth'][-1])

    # Metrics based on all subfiles combined
    tn, fp, fn, tp = c_matrix
    m = metrics.evaluation_metrics(tp,tn,fp,fn)
    for metric, value in m.items():
        rows[metric].append(value)
    rows['tp'].append(tp)
    rows['tn'].append(tn)
    rows['fp'].append(fp)
    rows['fn'].append(fn)

    # Expand non-present features in this configuration
    for feature, value in rows.items():
        if len(value) == 0:
            rows[feature] = [np.nan]*n_rows

    # Write to file
    results = write_results(f'results/{conf["model"]}/{res_file}', rows)
    _ = write_results_short(f'results/{conf["model"]}/{res_file_short}', rows)
    return results


def compute_metrics_downsample(conf_no, conf, org, org_filename, subfiles, LOOKBACK_SIZE, t=94):
    c_matrix = np.array([0,0,0,0])
    rows = new_empty_row()
    n_rows = len(subfiles) + 1
    for _ in range(n_rows):
        rows['filename'].append(org_filename)

        # Model configuration
        rows['conf_no'].append(conf_no)
        for arg, value in conf.items():
            rows[arg].append(value)

        # Validation configuration
        rows['threshold'].append(t)
        rows['lower_boundary'].append(np.nan)
        rows['upper_boundary'].append(np.nan)
        rows['downsample'].append(True)
    del rows['unwanted_features']

    for subfile_no, subfile in enumerate(subfiles):
        depth = np.array(subfile['DEPTH'])
        subfile_range = (depth[0],depth[-1])
        subset = org[(org['DEPTH'] >= subfile_range[0]) & (org['DEPTH'] <= subfile_range[1])]

        labels = np.array([0]*len(depth))
        ranges = get_ranges(0,0.999)
        for r in ranges:
            mask = (r[0] <= depth) & (depth <= r[1])
            labels[mask] = 1
        labels = np.delete(labels, 0) # Delete first

        percentiles = subset['percentile']
        percentiles = percentiles.dropna()
        preds = np.array([0]*len(percentiles))
        preds[percentiles >= t] = 1

        # Downsample
        labels, preds = downsample(depth[LOOKBACK_SIZE+1:], labels[LOOKBACK_SIZE:], preds)
        # Subfile info
        rows['subfile'].append(subfile_no)
        rows['bottom_depth'].append(subfile_range[0])
        rows['top_depth'].append(subfile_range[1])

        # Subfile metrics
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        c_matrix += [tn, fp, fn, tp]
        m = metrics.evaluation_metrics(tp,tn,fp,fn)
        for metric, value in m.items():
            rows[metric].append(value)
        rows['tp'].append(tp)
        rows['tn'].append(tn)
        rows['fp'].append(fp)
        rows['fn'].append(fn)

    # Entire file info
    rows['subfile'].append(np.nan)
    rows['bottom_depth'].append(rows['bottom_depth'][0])
    rows['top_depth'].append(rows['top_depth'][-1])

    # Metrics based on all subfiles combined
    tn, fp, fn, tp = c_matrix
    m = metrics.evaluation_metrics(tp,tn,fp,fn)
    for metric, value in m.items():
        rows[metric].append(value)
    rows['tp'].append(tp)
    rows['tn'].append(tn)
    rows['fp'].append(fp)
    rows['fn'].append(fn)

    # Expand non-present features in this configuration
    for feature, value in rows.items():
        if len(value) == 0:
            rows[feature] = [np.nan]*n_rows

    # Write to file
    results = write_results(f'results/{conf["model"]}/{res_file}', rows)
    _ = write_results_short(f'results/{conf["model"]}/{res_file_short}', rows)
    return results


def write_results(file, rows):
    results = None
    # pprint(rows)
    rows = pd.DataFrame(data=rows)
    if isfile(file):
        results = pd.read_csv(file)
        results = pd.concat([results, rows], ignore_index=True)
    else:
        results = rows

    results.to_csv(file, index=False)
    return results


def write_results_short(file, rows):
    results = None
    columns = ['conf_no','subfile','threshold','downsample','accuracy', 'precision','recall','f1_score', 'mcc','prevalence']
    rows = pd.DataFrame(data=rows)[columns]

    if isfile(file):
        results = pd.read_csv(file)
        results = pd.concat([results, rows], ignore_index=True)
    else:
        results = rows

    results.to_csv(file, index=False)
    return results


def downsample(depth, labels, preds):
    # This method max pools values within a meter
    # Starts from a round number
    depth = np.floor(depth)
    labels_downsampled = np.array([0]*int(depth[-1]-depth[0]+1))
    preds_downsampled = np.array([0]*int(depth[-1]-depth[0]+1))
    for i, val in enumerate(range(int(depth[0]), int(depth[-1])+1)):
        mask = (val == depth)
        try:
            labels_downsampled[i] = max(labels[mask])
        except:
            labels_downsampled[i] = 0
        try:
            preds_downsampled[i] = max(preds[mask])
        except:
            preds_downsampled[i] = 0

    return labels_downsampled, preds_downsampled


def run_ae(i, conf, source, file, df_org, org_subfiles):
    print('-----Training AE model-----')
    print('Loading datasets...')
    trainPath = join(source, 'datasets/remove/training/')
    unwanted_features = conf['unwanted_features']

    missing = 'remove'
    nLines = 1

    before = time.time()
    dataset, val_dataset = v.scale_and_shape(trainPath, unwanted_features, missing, nLines=nLines)
    after = time.time()  

    print('Time to shape dataset: ' + '{:1.4f}'.format(after-before) + ' seconds\n')

    # Network Architecture Parameters
    batch_size_train = conf['batch_size_train']
    batch_size_test = conf['batch_size_test']
    original_dim = len(list(dataset))
    input_shape = (original_dim, )
    intermediate_dims = conf['intermediate_dims']
    latent_dim = conf['latent_dim']
    n_epochs = conf['epochs']

    ae = models.AE(original_dim, intermediate_dims, latent_dim)
    
    ae.compile(optimizer=conf['optimizer'], loss=conf['loss'])
    ae.build(input_shape=(None,original_dim))
    
    before = time.time()
    history = ae.fit(dataset, dataset,
                     shuffle=True,
                     epochs=n_epochs,
                     batch_size=batch_size_train,
                     validation_data=(val_dataset, val_dataset),
                     verbose=2)
    after = time.time()

    minutes = int((after-before)/60)
    seconds = max((after-before)-(minutes*60), 0)
    hours = int(minutes/60)
    print('Time to fit: ' + str(hours) +'h:'+ str(minutes) + 'm:' + '{:1.2f}'.format(seconds) + 's')

    # Save weights
    path = f"results/{conf['model']}/{i}/" 
    util.ensure_path(path)
    ae.save_weights(join(path, "ae.h5"))

    with open(join(path,'config.json'), 'w') as f:
        json.dump(conf, f, indent=4)
    print(f'Model has been saved as: {join(path, "ae.h5")}')
    del dataset
    ##############

    print('\n-----Obtaining results-----')
    # Scale and shape test files
    ae_test = v.scale_and_shape_test(org_subfiles, unwanted_features, missing, nLines=nLines)

    # Predict subfiles
    pred_subfiles = []
    for test in ae_test:
        pred_subfiles.append(ae.predict(test, batch_size=batch_size_test))
    
    ed_lists = []
    der_lists = []
    output_file = pd.DataFrame(data={
                                     'ed_ae': [np.nan]*len(df_org), 
                                     'der_ae': [np.nan]*len(df_org)})
    for j, subfile in enumerate(org_subfiles):
        index = df_org.index[df_org['DEPTH'] == subfile['DEPTH'][0]][0]
        ed_subfile = np.linalg.norm(ae_test[j]-pred_subfiles[j], axis=1)
        der_subfile = np.insert(np.abs(np.diff(ed_subfile)), 0, np.nan)
        
        ed = [np.nan]*len(df_org)
        der = [np.nan]*len(df_org)
        ed[index:index+len(subfile)] = ed_subfile
        der[index:index+len(subfile)] = der_subfile
        ed_lists.append(ed)
        der_lists.append(der)

        output_file = output_file.combine_first(pd.DataFrame(data={
                                                                   'ed_ae': ed, 
                                                                   'der_ae': der}))

    # Add metrics
    output_file = add_cols(df_org, output_file)
    output_file = add_poi(output_file, der_lists)
    output_file = add_percentiles(output_file, der_lists)
    output_file = add_labels(output_file)
    output_file.to_csv(join(path, file), index=False)
    add_units(join(path, file))
    print(f'Output file saved as: {join(path, file)}')

    print('Saving figures...')

    # Save loss plot
    loss = np.transpose([history.history['loss'], history.history['val_loss']])
    loss = pd.DataFrame(loss, columns=['Training Loss', 'Validation Loss'])
    save_figure(range(1,n_epochs+1), loss, 'Epoch', 'Loss', 'Training & Validation Loss', join(path, 'loss'))

    # Save ed/der figures
    x = df_org['DEPTH']
    xlab = "DEPTH"
    save_figure(x, ed_lists, xlab, "Euclidean Distance", file, join(path, 'euclid'))           
    save_figure(x, der_lists, xlab, "Change in Euclidean Distance (absolute value)", file, join(path, 'derivative'))

    # Save original figure
    for j, subfile in enumerate(ae_test):
        x = org_subfiles[j]['DEPTH']
        
        labels = ['AC', 'ACS', 'DEN', 'DENC', 'GR', 'NEU', 'PEF', 'RDEP', 'RMED', 'CALI-BS']
        data = pd.DataFrame(subfile,  columns=labels)      
        save_figure(x, data, xlab, "Original Value (scaled)", file, join(path, f'orginal{j}'))

    # Save prediction figure
    for j, subfile in enumerate(pred_subfiles):
        x = org_subfiles[j]['DEPTH']
        labels = ['AC', 'ACS', 'DEN', 'DENC', 'GR', 'NEU', 'PEF', 'RDEP', 'RMED', 'CALI-BS']
        preds = pd.DataFrame(subfile,  columns=labels)      
        save_figure(x, preds, xlab, "Predicted Value (scaled)", file, join(path, f'prediction{j}'))

    print('Computing metrics...')
    f1_mcc = []
    f1_mcc_d = []
    f1_mcc_combined = []
    threshold = []
    for t in range(80,100):
        metrics = compute_metrics(i, conf, output_file, file, org_subfiles, 0, t=t)
        metrics_d = compute_metrics_downsample(i, conf, output_file, file, org_subfiles, 0, t=t)

        f1 = metrics['f1_score'][len(metrics)-1]
        mcc = metrics['mcc'][len(metrics)-1]
        threshold.append(metrics['threshold'][len(metrics)-1])
        f1_mcc.append([f1,mcc])

        f1_d = metrics_d['f1_score'][len(metrics_d)-1]
        mcc_d = metrics_d['mcc'][len(metrics_d)-1]
        f1_mcc_d.append([f1_d,mcc_d])

        f1_mcc_combined.append([f1,mcc,f1_d,mcc_d])

    f1_mcc_combined = pd.DataFrame(f1_mcc_combined, columns=['F1 Lenient', 'MCC Lenient','F1 Downsampled', 'MCC Downsampled'])
    save_figure_f1_mcc(threshold,f1_mcc_combined, 'Threshold (percentile)', "F1 & MCC", 'F1 & MCC vs Threshold', join(path,f'ae_f1_mcc_{i}'))

    print(f'Metrics saved in: {res_file}')


def run_vae(i, conf, source, file, df_org, org_subfiles):
    print('-----Training VAE model-----')
    print('Loading datasets...')
    trainPath = join(source, 'datasets/remove/training/')
    unwanted_features = conf['unwanted_features']

    missing = 'remove'
    nLines = 1

    before = time.time()
    df, val_df = v.scale_and_shape(trainPath, unwanted_features, missing, nLines=nLines)
    after = time.time()  

    print('Time to shape dataset: ' + '{:1.4f}'.format(after-before) + ' seconds\n')

    # Network Architecture Parameters
    batch_size_train = conf['batch_size_train']
    batch_size_test = conf['batch_size_test']
    original_dim = len(list(df))
    input_shape = (original_dim, )
    intermediate_dims = conf['intermediate_dims']
    latent_dim = conf['latent_dim']
    n_epochs = conf['epochs']

    vae, encoder, decoder = models.vae_model(original_dim, intermediate_dims, latent_dim, 'kl-mse')
    vae.compile(optimizer=conf['optimizer'])
    before = time.time()
    history = vae.fit(df,
                      shuffle=True,
                      epochs=n_epochs,
                      batch_size=batch_size_train,
                      validation_data=(val_df,val_df),
                      verbose=2)
    after = time.time()

    minutes = int((after-before)/60)
    seconds = max((after-before)-(minutes*60), 0)
    hours = int(minutes/60)
    print('Time to fit: ' + str(hours) +'h:'+ str(minutes) + 'm:' + '{:1.2f}'.format(seconds) + 's')

    # Save weights
    path = f"results/{conf['model']}/{i}/"
    util.ensure_path(path)
    vae.save_weights(join(path, "vae.h5"))
    encoder.save_weights(join(path, "encoder.h5"))
    decoder.save_weights(join(path, "decoder.h5"))

    with open(join(path,'config.json'), 'w') as f:
        json.dump(conf, f, indent=4)
    print(f'Model has been saved as: {join(path, "vae.h5")}')
    del df
    ##############

    print('\n-----Obtaining results-----')
    # Scale and shape test files
    vae_test = v.scale_and_shape_test(org_subfiles, unwanted_features, missing, nLines=nLines)

    # Predict subfiles
    pred_subfiles = []
    for test in vae_test:
        pred_subfiles.append(vae.predict(test, batch_size=batch_size_test))
    
    ed_lists = []
    der_lists = []
    output_file = pd.DataFrame(data={
                                     'ed_vae': [np.nan]*len(df_org), 
                                     'der_vae': [np.nan]*len(df_org)})
    for j, subfile in enumerate(org_subfiles):
        index = df_org.index[df_org['DEPTH'] == subfile['DEPTH'][0]][0]
        ed_subfile = np.linalg.norm(vae_test[j]-pred_subfiles[j], axis=1)
        der_subfile = np.insert(np.abs(np.diff(ed_subfile)), 0, np.nan)
        
        ed = [np.nan]*len(df_org)
        der = [np.nan]*len(df_org)
        ed[index:index+len(subfile)] = ed_subfile
        der[index:index+len(subfile)] = der_subfile
        ed_lists.append(ed)
        der_lists.append(der)

        output_file = output_file.combine_first(pd.DataFrame(data={
                                                                   'ed_vae': ed, 
                                                                   'der_vae': der}))

    # Add metrics
    output_file = add_cols(df_org, output_file)
    output_file = add_poi(output_file, der_lists)
    output_file = add_percentiles(output_file, der_lists)
    output_file = add_labels(output_file)
    output_file.to_csv(join(path, file), index=False)
    add_units(join(path, file))
    print(f'Output file saved as: {join(path, file)}')

    print('Saving figures...')

    # Save loss plot
    loss = np.transpose([history.history['loss'], history.history['val_loss']])
    loss = pd.DataFrame(loss, columns=['Training Loss', 'Validation Loss'])
    save_figure(range(1,n_epochs+1), loss, 'Epoch', 'Loss', 'Training & Validation Loss', join(path, 'loss'))

    # Save ed/der figures
    x = df_org['DEPTH']
    xlab = "DEPTH"
    save_figure(x, ed_lists, xlab, "Euclidean Distance", file, join(path, 'euclid'))           
    save_figure(x, der_lists, xlab, "Change in Euclidean Distance (absolute value)", file, join(path, 'derivative'))

    # Save original figure
    for j, subfile in enumerate(vae_test):
        x = org_subfiles[j]['DEPTH']
        
        labels = ['AC', 'ACS', 'DEN', 'DENC', 'GR', 'NEU', 'PEF', 'RDEP', 'RMED', 'CALI-BS']
        data = pd.DataFrame(subfile,  columns=labels)      
        save_figure(x, data, xlab, "Original Value (scaled)", file, join(path, f'orginal{j}'))

    # Save prediction figure
    for j, subfile in enumerate(pred_subfiles):
        x = org_subfiles[j]['DEPTH']
        labels = ['AC', 'ACS', 'DEN', 'DENC', 'GR', 'NEU', 'PEF', 'RDEP', 'RMED', 'CALI-BS']
        preds = pd.DataFrame(subfile,  columns=labels)      
        save_figure(x, preds, xlab, "Predicted Value (scaled)", file, join(path, f'prediction{j}'))

    print('Computing metrics...')
    f1_mcc = []
    f1_mcc_d = []
    f1_mcc_combined = []
    threshold = []
    for t in range(80,100):
        metrics = compute_metrics(i, conf, output_file, file, org_subfiles, 0, t=t)
        metrics_d = compute_metrics_downsample(i, conf, output_file, file, org_subfiles, 0, t=t)

        f1 = metrics['f1_score'][len(metrics)-1]
        mcc = metrics['mcc'][len(metrics)-1]
        threshold.append(metrics['threshold'][len(metrics)-1])
        f1_mcc.append([f1,mcc])

        f1_d = metrics_d['f1_score'][len(metrics_d)-1]
        mcc_d = metrics_d['mcc'][len(metrics_d)-1]
        f1_mcc_d.append([f1_d,mcc_d])

        f1_mcc_combined.append([f1,mcc,f1_d,mcc_d])

    f1_mcc_combined = pd.DataFrame(f1_mcc_combined, columns=['F1 Lenient', 'MCC Lenient','F1 Downsampled', 'MCC Downsampled'])
    save_figure_f1_mcc(threshold,f1_mcc_combined, 'Threshold (percentile)', "F1 & MCC", 'F1 & MCC vs Threshold', join(path,f'vae_f1_mcc_{i}'))
    print(f'Metrics saved in: {res_file}')


def run_deepant(i, conf, source, file, df_org, org_subfiles):
    print('-----Training DeepAnT model-----')
    print('Loading datasets...')
    trainPath = join(source, 'datasets/remove/training/')
    unwanted_features = conf['unwanted_features']
    LOOKBACK_SIZE = conf['LOOKBACK_SIZE']
    original_dim = 13 - len(unwanted_features) -1

    before = time.time()
    datasets = d.scale_and_shape(trainPath, unwanted_features, LOOKBACK_SIZE=LOOKBACK_SIZE)
    after = time.time()

    print("Number of training datasets: " + str(len(datasets)))
    print('Time to shape datasets: ' + '{:1.4f}'.format(after-before) + ' seconds\n')

    deepant = models.DeepAnt(LOOKBACK_SIZE, original_dim, 
                             kernel_size=conf['kernel_size'], 
                             n_filters=conf['n_filters'], 
                             padding=conf['padding'], # 'same', 'valid', 'causal'
                             pool_size=conf['pool_size'], 
                             dense_layer_size=conf['dense_layer_size'], 
                             dropout=conf['dropout'])
    deepant.compile(optimizer=conf['optimizer'], loss=conf['loss'])
    deepant.build(input_shape=(None,LOOKBACK_SIZE,original_dim))

    before = time.time()
    n_datasets = len(datasets)
    n_epochs = conf['epochs']
    losses = [0]*n_epochs
    val_losses = [0]*n_epochs
    for epoch in range(n_epochs):
        t = time.time()
        c = 1
        loss = 0
        val_loss = 0
        for X, Y in datasets:
            history = deepant.fit(x=X,
                                  y=Y,
                                  shuffle=False,
                                  validation_split=0.1,
                                  batch_size=len(X),
                                  verbose=0)
            loss += history.history['loss'][0]
            val_loss += history.history['val_loss'][0]
            print(f'Epoch {epoch+1}/{n_epochs}, Dataset {c}/{n_datasets}', end='\r')
            c += 1
        print('Epoch ' + str(epoch+1) + ' took ' + '{:1.4f}'.format(time.time()-t) + ' seconds to fit')
        losses[epoch] = loss
        val_losses[epoch] = val_loss
    after = time.time()

    minutes = int((after-before)/60)
    seconds = max((after-before)-(minutes*60), 0)
    hours = int(minutes/60)
    print('Time to fit: ' + str(hours) +'h:'+ str(minutes) + 'm:' + '{:1.2f}'.format(seconds) + 's')

    # Save weights
    path = f"results/{conf['model']}/{i}/"
    util.ensure_path(path)
    deepant.save_weights(join(path, f"deepant_script.h5"))
    with open(join(path,'config.json'), 'w') as f:
        json.dump(conf, f, indent=4)
    print(f'Model has been saved as: {join(path, "deepant_script.h5")}')
    del datasets
    ##############

    print('\n-----Obtaining results-----')
    # Scale and shape test files
    deepant_test = d.scale_and_shape_test(org_subfiles, unwanted_features, LOOKBACK_SIZE=LOOKBACK_SIZE)

    # Predict subfiles
    pred_subfiles = []
    for test in deepant_test:
        pred_subfiles.append(deepant.predict(test[0]))
    
    # Remove datasets that are too small
    org_subfiles_deepant = []
    for df in org_subfiles:
        if len(df) > LOOKBACK_SIZE:
            org_subfiles_deepant.append(df)

    ed_lists = []
    der_lists = []
    output_file = pd.DataFrame(data={
                                     'ed_deepant': [np.nan]*len(df_org), 
                                     'der_deepant': [np.nan]*len(df_org)})
    for j, subfile in enumerate(org_subfiles_deepant):
        index = df_org.index[df_org['DEPTH'] == subfile['DEPTH'][0]][0]
        index += LOOKBACK_SIZE
        ed_subfile = np.linalg.norm(deepant_test[j][1]-pred_subfiles[j], axis=1)
        der_subfile = np.insert(np.abs(np.diff(ed_subfile)), 0, np.nan)

        ed = [np.nan]*len(df_org)
        der = [np.nan]*len(df_org)
        ed[index:index+len(ed_subfile)] = ed_subfile
        der[index:index+len(der_subfile)] = der_subfile
        ed_lists.append(ed)
        der_lists.append(der)
        
        output_file = output_file.combine_first(pd.DataFrame(data={'ed_deepant': ed, 'der_deepant': der}))

    # Add metrics
    output_file = add_cols(df_org, output_file)
    output_file = add_poi(output_file, der_lists)
    output_file = add_percentiles(output_file, der_lists)
    output_file = add_labels(output_file)
    output_file.to_csv(join(path, file), index=False)
    add_units(join(path, file))
    print(f'Output file saved as: {join(path, file)}')

    print('Saving figures...')
    # Save losses figure
    loss = np.transpose([losses,val_losses])
    loss = pd.DataFrame(loss, columns=['Training Loss', 'Validation Loss'])
    save_figure(range(1,n_epochs+1), loss, 'Epoch', 'Loss', 'Training & Validation Loss', join(path, 'loss'))

    # Save ed/der figures
    x = df_org['DEPTH']
    xlab = "DEPTH"
    save_figure(x, ed_lists, xlab, "Euclidean Distance", file, join(path, 'euclid'))           
    save_figure(x, der_lists, xlab, "Change in Euclidean Distance (absolute value)", file, join(path, 'derivative'))
    
    # Save original figure
    for j, subfile in enumerate(deepant_test):
        x = org_subfiles_deepant[j]['DEPTH'][LOOKBACK_SIZE:]
        
        labels = ['AC', 'ACS', 'DEN', 'DENC', 'GR', 'NEU', 'PEF', 'RDEP', 'RMED', 'CALI-BS']
        data = pd.DataFrame(subfile[1],  columns=labels)      
        save_figure(x, data, xlab, "Original Value (scaled)", file, join(path, f'orginal{j}'))

    # Save prediction figure
    for j, subfile in enumerate(pred_subfiles):
        x = org_subfiles_deepant[j]['DEPTH'][LOOKBACK_SIZE:]
        labels = ['AC', 'ACS', 'DEN', 'DENC', 'GR', 'NEU', 'PEF', 'RDEP', 'RMED', 'CALI-BS']
        preds = pd.DataFrame(subfile,  columns=labels)      
        save_figure(x, preds, xlab, "Predicted Value (scaled)", file, join(path, f'prediction{j}'))

    print('Computing metrics...')
    f1_mcc = []
    f1_mcc_d = []
    f1_mcc_combined = []
    threshold = []
    for t in range(80,100):
        metrics = compute_metrics(i, conf, output_file, file, org_subfiles_deepant, LOOKBACK_SIZE, t=t)
        metrics_d = compute_metrics_downsample(i, conf, output_file, file, org_subfiles_deepant, LOOKBACK_SIZE, t=t)

        f1 = metrics['f1_score'][len(metrics)-1]
        mcc = metrics['mcc'][len(metrics)-1]
        threshold.append(metrics['threshold'][len(metrics)-1])
        f1_mcc.append([f1,mcc])

        f1_d = metrics_d['f1_score'][len(metrics_d)-1]
        mcc_d = metrics_d['mcc'][len(metrics_d)-1]
        f1_mcc_d.append([f1_d,mcc_d])

        f1_mcc_combined.append([f1,mcc,f1_d,mcc_d])

    f1_mcc_combined = pd.DataFrame(f1_mcc_combined, columns=['F1 Lenient', 'MCC Lenient','F1 Downsampled', 'MCC Downsampled'])
    save_figure_f1_mcc(threshold,f1_mcc_combined, 'Threshold (percentile)', "F1 & MCC", 'F1 & MCC vs Threshold', join(path,f'deepant_f1_mcc_{i}'))

    print(f'Metrics saved in: {res_file}')


def run_lstm(i, conf, source, file, df_org, org_subfiles):
    print('-----Training LSTM model-----')
    print('Loading datasets...')
    trainPath = join(source, 'datasets/remove/training/')
    unwanted_features = conf['unwanted_features']
    LOOKBACK_SIZE = conf['LOOKBACK_SIZE']
    neurons = conf['lstm_neurons']
    original_dim = 13 - len(unwanted_features) -1
    
    before = time.time()
    datasets = d.scale_and_shape(trainPath, unwanted_features, LOOKBACK_SIZE=LOOKBACK_SIZE)
    after = time.time()

    print("Number of training datasets: " + str(len(datasets)))
    print('Time to shape datasets: ' + '{:1.4f}'.format(after-before) + ' seconds\n')

    lstm = models.LSTM(neurons, LOOKBACK_SIZE, original_dim, dropout=conf['dropout'])
    lstm.compile(optimizer=conf['optimizer'], loss=conf['loss'])
    lstm.build(input_shape=(None,LOOKBACK_SIZE,original_dim))
    
    before = time.time()
    n_datasets = len(datasets)
    n_epochs = conf['epochs']
    losses = [0]*n_epochs
    val_losses = [0]*n_epochs
    for epoch in range(n_epochs):
        t = time.time()
        c = 1
        loss = 0
        val_loss = 0
        for X, Y in datasets:
            history = lstm.fit(x=X,
                        y=Y,
                        shuffle=False,
                        validation_split=0.1,
                        batch_size=len(X),
                        verbose=0)
            loss += history.history['loss'][0]
            val_loss += history.history['val_loss'][0]
            lstm.reset_states() 
            print(f'Epoch {epoch+1}/{n_epochs}, Dataset {c}/{n_datasets}', end='\r')
            c += 1
        print('Epoch ' + str(epoch+1) + ' took ' + '{:1.4f}'.format(time.time()-t) + ' seconds to fit')
        losses[epoch] = loss
        val_losses[epoch] = val_loss
    after = time.time()

    minutes = int((after-before)/60)
    seconds = max((after-before)-(minutes*60), 0)
    hours = int(minutes/60)
    print('Time to fit: ' + str(hours) +'h:'+ str(minutes) + 'm:' + '{:1.2f}'.format(seconds) + 's')
    
    # Save weights
    path = f"results/{conf['model']}/{i}/"
    util.ensure_path(path)
    lstm.save_weights(join(path, f"lstm_script.h5"))
    with open(join(path,'config.json'), 'w') as f:
        json.dump(conf, f, indent=4)
    print(f'Model has been saved as: {join(path, "lstm_script.h5")}')
    del datasets
    ##############

    print('\n-----Obtaining results-----')
    # Scale and shape test files
    lstm_test = d.scale_and_shape_test(org_subfiles, unwanted_features, LOOKBACK_SIZE=LOOKBACK_SIZE)

    # Predict subfiles
    pred_subfiles = []
    for test in lstm_test:
        pred_subfiles.append(lstm.predict(test[0]))
    
    # Remove datasets that are too small
    org_subfiles_lstm = []
    for df in org_subfiles:
        if len(df) > LOOKBACK_SIZE:
            org_subfiles_lstm.append(df)

    ed_lists = []
    der_lists = []
    output_file = pd.DataFrame(data={
                                     'ed_lstm': [np.nan]*len(df_org),
                                     'der_lstm': [np.nan]*len(df_org)})
    for j, subfile in enumerate(org_subfiles_lstm):
        index = df_org.index[df_org['DEPTH'] == subfile['DEPTH'][0]][0]
        index += LOOKBACK_SIZE
        ed_subfile = np.linalg.norm(lstm_test[j][1]-pred_subfiles[j], axis=1)
        der_subfile = np.insert(np.abs(np.diff(ed_subfile)), 0, np.nan)

        ed = [np.nan]*len(df_org)
        der = [np.nan]*len(df_org)
        ed[index:index+len(ed_subfile)] = ed_subfile
        der[index:index+len(der_subfile)] = der_subfile
        ed_lists.append(ed)
        der_lists.append(der)
        
        output_file = output_file.combine_first(pd.DataFrame(data={'ed_lstm': ed, 'der_lstm': der}))

    # Add metrics
    output_file = add_cols(df_org, output_file)
    output_file = add_poi(output_file, der_lists)
    output_file = add_percentiles(output_file, der_lists)
    output_file = add_labels(output_file)
    output_file.to_csv(join(path, file), index=False)
    add_units(join(path, file))
    print(f'Output file saved as: {join(path, file)}')

    print('Saving figures...')
    # Save losses figure
    loss = np.transpose([losses,val_losses])
    loss = pd.DataFrame(loss, columns=['Training Loss', 'Validation Loss'])
    save_figure(range(1,len(loss)+1), loss, 'Epoch', 'Loss', 'Training & Validation Loss', join(path, 'loss'))
    
    # Save ed/der figures
    x = df_org['DEPTH']
    xlab = "DEPTH"
    save_figure(x, ed_lists, xlab, "Euclidean Distance", file, join(path, 'euclid'))           
    save_figure(x, der_lists, xlab, "Change in Euclidean Distance (absolute value)", file, join(path, 'derivative'))
    
    # Save original figure
    for j, subfile in enumerate(lstm_test):
        x = org_subfiles_lstm[j]['DEPTH'][LOOKBACK_SIZE:]
        
        labels = ['AC', 'ACS', 'DEN', 'DENC', 'GR', 'NEU', 'PEF', 'RDEP', 'RMED', 'CALI-BS']
        data = pd.DataFrame(subfile[1],  columns=labels)      
        save_figure(x, data, xlab, "Original Value (scaled)", file, join(path, f'orginal{j}'))

    # Save prediction figure
    for j, subfile in enumerate(pred_subfiles):
        x = org_subfiles_lstm[j]['DEPTH'][LOOKBACK_SIZE:]
        labels = ['AC', 'ACS', 'DEN', 'DENC', 'GR', 'NEU', 'PEF', 'RDEP', 'RMED', 'CALI-BS']
        preds = pd.DataFrame(subfile,  columns=labels)      
        save_figure(x, preds, xlab, "Predicted Value (scaled)", file, join(path, f'prediction{j}'))

    print('Computing metrics...')
    f1_mcc = []
    f1_mcc_d = []
    f1_mcc_combined = []
    threshold = []
    for t in range(80,100):
        metrics = compute_metrics(i, conf, output_file, file, org_subfiles_lstm, LOOKBACK_SIZE, t=t)
        metrics_d = compute_metrics_downsample(i, conf, output_file, file, org_subfiles_lstm, LOOKBACK_SIZE, t=t)

        f1 = metrics['f1_score'][len(metrics)-1]
        mcc = metrics['mcc'][len(metrics)-1]
        threshold.append(metrics['threshold'][len(metrics)-1])
        f1_mcc.append([f1,mcc])

        f1_d = metrics_d['f1_score'][len(metrics_d)-1]
        mcc_d = metrics_d['mcc'][len(metrics_d)-1]
        f1_mcc_d.append([f1_d,mcc_d])

        f1_mcc_combined.append([f1,mcc,f1_d,mcc_d])

    f1_mcc_combined = pd.DataFrame(f1_mcc_combined, columns=['F1 Lenient', 'MCC Lenient','F1 Downsampled', 'MCC Downsampled'])
    save_figure_f1_mcc(threshold,f1_mcc_combined, 'Threshold (percentile)', "F1 & MCC", 'F1 & MCC vs Threshold', join(path,f'lstm_f1_mcc_{i}'))

    print(f'Metrics saved in: {res_file}')
    
    
def run(i, conf, source, file, df_org, org_subfiles):
    limit_gpu_memory()
    print('\n----------------------------------')
    print(f'-----Configuration #{i}-----')
    pprint(conf)
    print() # Empty line

    if conf['model'] == 'deepant':
        run_deepant(i, conf, source, file, df_org, org_subfiles)
    if conf['model'] == 'vae':
        run_vae(i, conf, source, file, df_org, org_subfiles)
    if conf['model'] == 'ae':
        run_ae(i, conf, source, file, df_org, org_subfiles)
    if conf['model'] == 'lstm':
        run_lstm(i, conf, source, file, df_org, org_subfiles)

if __name__ == '__main__':
    if isfile(res_file):
        print(f'Deleting {res_file}')
        os.remove(res_file)
    if isfile(res_file_short):
        print(f'Deleting {res_file_short}')
        os.remove(res_file_short)

    source = get_path()
    testPath = join(source, 'path/to/preprocessed/test/file')
    orgPath = join(source, 'path/to/original/test/file')
    
    # Load subfiles from original test file 
    file_number = 0
    files = map_datasets(testPath, orgPath)
    file = list(files.keys())[file_number]
    print(f'Test file: {file}')
    df_org = pd.read_csv(join(orgPath, file))
    org_subfiles = files[file]
    
    for i, conf in enumerate(get_config()):
        p = Process(target=run, args=(i,conf,source,file,df_org,org_subfiles))
        p.start()
        p.join()
        p.close()
        time.sleep(2.0) # Give time to release resources
        