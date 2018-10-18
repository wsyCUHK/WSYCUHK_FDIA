# -*- coding: utf-8 -*-
"""
Created on Tue May 15 11:34:42 2018

@author: Wang Shuoyao
"""

import warnings
import sys
warnings.filterwarnings("ignore")
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer  
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import scipy.io as sio



def test2train(save_name):
    y_pred = sio.loadmat(save_name)['output_mode_pred']
    return y_pred
        # sio.savemat(save_name, {'input_h': X_test/10000000,'output_mode':Y_test,'output_mode_pred': y_pred})
        # # Compute the error for iteration t.
           
def cal_acc(a,b):
    n=a.shape[0]
    m=a.shape[1]
    tterr=0
    r_err=0
    for i in range(n):
        cuerr=0
        for j in range(m):
            if a[i][j]!= b[i][j]:
               tterr+=1
               cuerr+=1
        if cuerr>0:
            r_err+=1
            
    return 1-r_err/n, 1-tterr/(n*m)

KK = 1                     # number of users
K= 1

# Load data
measurement = sio.loadmat('./data/20_train%d' %K)['X_t']
attack = sio.loadmat('./data/20_train%d' %K)['label_t']
measurement1 = sio.loadmat('./data/20_train%d' %K)['X_s']
attack1 = sio.loadmat('./data/20_train%d' %K)['label_s']
#measurement1 = sio.loadmat('./data/test%d' %KK)['newStateRe']
#attack1 = sio.loadmat('./data/test%d' %KK)['newlabel']


# online
def LGB_predict(train_x,train_y,test_x):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=10000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc', verbose=100, early_stopping_rounds=2000)
#    res['score'] = clf.predict_proba(test_x)[:,1]
   # res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    #res.to_csv('submission.csv', index=False)
    return clf.predict_proba(test_x)
pred_y=LGB_predict(measurement,attack,measurement1)
print(cal_acc(pred_y,attack1))