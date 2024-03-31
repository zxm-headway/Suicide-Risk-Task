#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import csv
import numpy as np
import pandas as pd
import datetime
from sklearn import metrics


def load_df(dataset_name):
    if dataset_name in ['reddit_500', 'reddit']:
        file_name = './data/reddit.csv'
        df = pd.read_csv(file_name)
        return df
    else:
        raise ValueError("Error: unrecognized dataset")



# 找到适合的阈值
def find_threshold(fpr, tpr, threshold):
    rate = np.array(tpr) + np.array(fpr)
    return threshold[np.argmax(rate)]


def evaluate_prediction(y_test, y_pred, model_name, dataset_name):
    fpr, tpr, th = metrics.roc_curve(y_test, y_pred)

    roc_auc = metrics.auc(fpr, tpr)

    df_results = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}, index=None)

    df_results.to_csv('./prediction_{}_{}.csv'.format(dataset_name, model_name), index=False)

    o_threshold = 0.5
    for i in range(len(y_pred)):
        if y_pred[i] >= o_threshold:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    acc = metrics.accuracy_score(y_test, y_pred)
    pre = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    dict_eval = {'date': datetime.date.today(),
                 'model': model_name,
                 'accuracy': acc,
                 'precision': pre,
                 'recall': rec,
                 'f-score': f1,
                 'roc': roc_auc,
                #  'note': '{}_th fold'.format(k_th),
                 'dataset': dataset_name
                 }
    
    with open('./{}.csv'.format(dataset_name), 'a') as f:
        field_names = ['date', 'model', 'accuracy', 'precision', 'recall', 'f-score', 'roc', 'note', 'dataset']
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writerow(dict_eval)
    return acc, pre, rec, f1, roc_auc

from tools import utils
reddit =  utils.load_df('reddit_500')
print(reddit['Post'][0])