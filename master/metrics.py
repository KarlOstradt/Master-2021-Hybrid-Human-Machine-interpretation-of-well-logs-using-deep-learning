import numpy as np
import math
import matplotlib.pyplot as plt
from prettytable import PrettyTable

import master.util as util

def evaluation_metrics(truePos, trueNeg, falsePos, falseNeg):    
    eval_metrics = {
        "accuracy" : accuracy(truePos,trueNeg,falsePos,falseNeg),
        "error_rate": error_rate(truePos, trueNeg, falsePos, falseNeg),
        "prevalence" : prevalence(truePos, trueNeg, falsePos, falseNeg),
        "null_error_rate" : null_error_rate(truePos, trueNeg, falsePos, falseNeg),
        "precision" : precision(truePos, falsePos),
        "recall" : recall(truePos, falseNeg),
        "specificity" : specificity(trueNeg, falsePos),
        "fallout" : fallout(trueNeg, falsePos),
        "miss_rate" : miss_rate(truePos, falseNeg),
        "f1_score" : f_score(truePos, falsePos, falseNeg, beta=1),
        "f2_score" : f_score(truePos, falsePos, falseNeg, beta=2),
        "false_discovery_rate" : false_discovery_rate(truePos,falsePos),
        "false_omission_rate" : false_omission_rate(trueNeg, falseNeg),
        "mcc" : matthews_correlation_coefficient(truePos, trueNeg, falsePos, falseNeg)
    }
    
    return eval_metrics 


def accuracy(truePos, trueNeg, falsePos, falseNeg): 
    # Proportion of correct labels
    if truePos + trueNeg + falsePos + falseNeg == 0: return np.nan
    return (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg)


def error_rate(truePos, trueNeg, falsePos, falseNeg):
    # Proportion of incorrect labels (1 - accuracy)
    if truePos + trueNeg + falsePos + falseNeg == 0: return np.nan
    return (falsePos + falseNeg) / (truePos + trueNeg + falsePos + falseNeg)


def prevalence(truePos, trueNeg, falsePos, falseNeg):
    # Proportion of positive labels
    if truePos + trueNeg + falsePos + falseNeg == 0: return np.nan
    return (truePos + falseNeg) / (truePos + trueNeg + falsePos + falseNeg)


def null_error_rate(truePos, trueNeg, falsePos, falseNeg):
    # Proportion of negative labels
    if truePos + trueNeg + falsePos + falseNeg == 0: return np.nan
    return (trueNeg + falsePos) / (truePos + trueNeg + falsePos + falseNeg)


def precision(truePos, falsePos):
    # Alias: Positive predictive value (PPV)
    if truePos + falsePos == 0: return np.nan
    return truePos / (truePos + falsePos)


def recall(truePos, falseNeg):
    # Alias: Sensitivity / True positive rate (TPR)
    if truePos + falseNeg == 0: return np.nan
    return truePos / (truePos + falseNeg)
    

def specificity(trueNeg, falsePos):
    # Alias: Selectivity / True negative rate (TNR)
    if trueNeg + falsePos == 0: return np.nan
    return trueNeg / (trueNeg + falsePos)


def fallout(trueNeg, falsePos):
    # Alias: False positive rate (FPR)
    if falsePos + trueNeg == 0: return np.nan
    return falsePos / (falsePos + trueNeg)


def miss_rate(truePos, falseNeg):
    # Alias: False negative rate (FNR)
    if falseNeg + truePos == 0: return np.nan
    return  falseNeg / (falseNeg + truePos)


def f_score(truePos, falsePos, falseNeg, beta=1):
    prec = precision(truePos, falsePos)
    rec = recall(truePos, falseNeg)
    if prec == np.nan: return np.nan
    if rec == np.nan: return np.nan
    return (1+beta**2) * ((prec * rec) / ((beta**2 * prec) + rec))


def false_discovery_rate(truePos,falsePos):
    if falsePos + truePos == 0: return np.nan
    return falsePos / (falsePos + truePos)


def false_omission_rate(trueNeg, falseNeg):
    if falseNeg + trueNeg == 0: return np.nan
    return falseNeg / (falseNeg + trueNeg)


def matthews_correlation_coefficient(truePos, trueNeg, falsePos, falseNeg):
    # Testing for cases when only one value is non-zero
    n_cases = truePos + trueNeg + falsePos + falseNeg
    if truePos == n_cases or trueNeg == n_cases:
        return 1
    elif falsePos == n_cases or falseNeg == n_cases:
        return -1

    # Testing for rows or columns in confusion matrix with sum of zero
    p = truePos + falseNeg # Actual positives
    n = falsePos + trueNeg # Actual negatives
    pp = truePos + falsePos # Predicted positives
    pn = falseNeg + trueNeg # Predicted negatives
    if p == 0 or n == 0 or pp == 0 or pn == 0:
        return 0

    nominator = truePos*trueNeg - falsePos*falseNeg
    denominator = math.sqrt(math.prod(np.array([p,n,pp,pn], dtype=np.float64)))
    return nominator / denominator


def stats(lists, percentiles=[5,50,95]):
    """Find min, max, avg and percentiles for test files.

    Args:
        lists : Tuple, list or numpy array containing lists (of numbers).

    Returns:
        dict: Statistics for test files, or None if any lists are empty.
    """
    if len(lists) == 0:
        return None
    for l in lists:
        if len(l) == 0:
            return None
    z = np.concatenate((lists))
    stats =  {
        'min': min(z),          # Global minimum
        'max': max(z),          # Global maximum
        'avg': sum(z)/len(z) ,  # Global average
        'files': list()         # Statistics of all individual test files
    }
    for p in percentiles:
        key = f'p{p}'
        stats[key] = 0


    for l in lists:
        percentile = np.percentile(l, percentiles)
        e = {
            'min': min(l),
            'max': max(l),
            'avg': sum(l)/len(l),
        }
        for i, p in enumerate(percentiles):
            key = f'p{p}'
            e[key] = percentile[i]
        stats['files'].append(e)

    median = lambda p : np.median(list(map(lambda i : stats['files'][i][p] , range(len(stats['files'])))))
    for p in percentiles:
        key = f'p{p}'
        stats[key] = median(key)

    return stats







