# -*- coding: utf-8 -*-
"""
Created on Tue May  1 21:16:25 2018

@author: Wang Shuoyao
"""

import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy
import dnn_kernal_wangshuoyao as dnn     # import our function file
import tensorflow as tf


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


classifiers = []
netarray=[]






# =============================================================================
# model_location3 = "./DNNmodel/model_demo3.ckpt"
# save_name3="./data/3Prediction_%d" % K
# net=[50,80]
# netarray.append(net)
# training_epochs=200  #training rate
# regularizer=0.0005
# batch_size=100
# LR = 0.001   #learning rate
# in_keep=1
# hi_keep=1
# LRdecay=1
# print('Case: K=%d, Total Samples: %d, Total Iterations: %d, layers:%d\n'%(K, len(measurement), training_epochs,len(net)))
# 
# # Train the deep neural network
# print('train DNN3 ...')
# dnn.DNN_train(net,measurement, attack, measurement1, attack1,model_location3,export_weight_biase_sw,regularizer,training_epochs,batch_size,LR,in_keep,hi_keep,LRdecay)
# 
# 
# =============================================================================




# # Test Deep Neural Networks
# dnntime, Y_pred = dnn.DNN_test(net,measurement1, attack1,  model_location,save_name,binary=1)
# print('Testing Time: %0.3f s' % (dnntime))



# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#          iterations - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights

n_pts, n_labels = np.shape(attack1)

 # append new classifiers to this list
alphas = []  # append the vote weight of the classifiers to this list

    # The weights for the first iteration


# setup and initialize the DNN network structure
current_weights = np.ones((n_pts, 1)) / float(n_pts)

iterations=3
for t in range(iterations):
        # A new classifier can be trained like this, given the current weights.
        #classifiers.append(base_classifier.train_classifier(X, labels, current_weights))

        # Do classification for each point.
        #vote_t = classifiers[-1].classify(X)
        

        
        name="./data/"+str(t+1)+"33Prediction_1"
        yy=[]
        
        y_pred=test2train(name)
        for i in range(n_pts):
            y=[]
            for j in range(n_labels):
                if y_pred[i][j]>0:
                    y.append(1)
                else:
                    y.append(0)
            yy.append(y)
        error_t = 0.00001  # If we have 0 here, some alphas will be inf.
        for i in range(n_pts):
            if  np.any(yy[i] != attack1[i]):
                    error_t += current_weights[i]

        # Compute alpha for iteration t.
        alpha_t = 0.5 * (np.log(1 - error_t) - np.log(error_t))
        alphas.append(alpha_t)

        # Update the weights for next iteration, t + 1.
        for i in range(n_pts):
                factor = np.exp(-alpha_t) if np.any(y_pred[i] == attack[i]) else np.exp(alpha_t)
                current_weights[i] *= factor
        current_weights / np.sum(current_weights)
        #print(current_weights)


# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    n_labels - the number of different classes
# out:  y_pred - N vector of class predictions for test points

    #n_pts = X.shape[0]
    #n_classifiers = len(classifiers)
n_classifiers=3
    # If we only have one classifier, we may just classify directly.
votes = np.zeros((n_pts, n_labels)) #the vote result?

for classindex in range(3):
        name="./data/"+str(classindex+1)+"33Prediction_1"
        
        yy=[]
        
        y_pred=test2train(name)

        for i in range(n_pts):
            y=[]
            for j in range(n_labels):
                if y_pred[i][j]>0.1:
                    y.append(1)
                else:
                    y.append(0)
            yy.append(y)
        e1,e2=cal_acc(np.array(yy),attack1)            
        print('Classifier',str(classindex+1),'|Accuracy:',e2,'Row Accuracy:',e1,'\n')
        votes += alphas[classindex]*yy
        vv=votes
        
        
for thr in range(16):        
    for i in range(n_pts):
        for j in range(n_labels):
            if votes[i][j]>(thr+1)/2:
                votes[i][j]=1
            else:
                votes[i][j]=0

    e1,e2=cal_acc(votes,attack1)            
    print('Boosting', (thr+1)/2,'|Accuracy:',e2,'Row Accuracy:',e1,'\n')
        # sio.savemat(save_name, {'input_h': X_test/10000000,'output_mode':Y_test,'output_mode_pred': y_pred})
        # # Compute the error for iteration t.
        # error_t = 0.00001  # If we have 0 here, some alphas will be inf.
        # for i in range(n_pts):
        #     if y_pred[i] != attack[i]:
        #         error_t += current_weights[i]

        # # Compute alpha for iteration t.
        # alpha_t = 0.5 * (np.log(1 - error_t) - np.log(error_t))
        # alphas.append(alpha_t)

        # # Update the weights for next iteration, t + 1.
        # for i in range(n_pts):
        #     factor = np.exp(-alpha_t) if y_pred[i] == attack[i] else np.exp(alpha_t)
        #     current_weights[i] *= factor
        # current_weights / np.sum(current_weights)



        # for t in range(n_classifiers):
        #     h = classifiers[t].classify(X) #the y-pred result of this classifier
        #     for i, xi_label in enumerate(h):
        #         votes[i][xi_label] += alphas[t] #alphas is the weight, the upper function learn the weights

        # return np.argmax(votes, axis=1)


