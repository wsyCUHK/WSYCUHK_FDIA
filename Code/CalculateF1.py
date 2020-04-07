# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:33:36 2018

@author: Wang Shuoyao
"""


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

import scipy.io as sio 
# Load data

for i in range (5):
    j=i+1
    x_train= sio.loadmat('../data/14case_CNN%d'%j)['output_mode']
    y_train= sio.loadmat('../data/14case_CNN%d'%j)['output_mode_pred']



    n1=x_train.shape[0]
    n2=x_train.shape[1]
    wahaha=0
    waha=0
    wh=0
    whh=0
    for i in range(n1):
        for j in range (n2):
            if y_train>0.5:
                if x_train[i][i]*y_train[i][j]>0.5:
                    wahaha+=1
                    waha+=1
                else:
                    waha+=1
            else:
                if x_train[i][i]+y_train[i][j]>0.5:
                    wh+=1
                else:
                    whh+=1
                    wh+=1	
    #row,acca=cal_acc(pred_y,y_test)
    
    print('Precise:',wahaha/waha,'Recall:',whh/wh,'F1:',2*((wahaha*whh/(wh*waha))/(wahaha/waha+whh/wh)))
#sio.savemat('../data/14case_CNN4_N5', {'output_mode':pred_y,'output_mode_pred': y_test})