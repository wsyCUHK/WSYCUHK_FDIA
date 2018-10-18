# -*- coding: utf-8 -*-
"""
Created on Tue May  1 13:07:21 2018

@author: Wang Shuoyao
"""

import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy
import dnn_kernal_wangshuoyao as dnn     # import our function file
import tensorflow as tf


def test2train(net,thelocation,attack,measurement):
    n_pts, n_labels = np.shape(attack)
    
     # append new classifiers to this list
    #alphas = []  # append the vote weight of the classifiers to this list
    
        # The weights for the first iteration
    tf.reset_default_graph()
    
    # setup and initialize the DNN network structure
    n_input = measurement.shape[1]                          # input size
    n_output = attack.shape[1]                         # output size
    with tf.name_scope('inputs'):
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_output])
    is_train = tf.placeholder("bool")
    
    input_keep_prob = tf.placeholder(tf.float32)
    hidden_keep_prob = tf.placeholder(tf.float32)
    
    # A new classifier can be trained like this, given the current weights.
    #classifiers.append(base_classifier.train_classifier(X, labels, current_weights))
    
            # Do classification for each point.
            #vote_t = classifiers[-1].classify(X
    weights=[tf.Variable(tf.truncated_normal([n_input, net[0]]) / np.sqrt(n_input))]
    for i in range(len(net)-1):
        weights.append(tf.Variable(tf.truncated_normal([net[i], net[i+1]]) / np.sqrt(net[i])))
    weights.append(tf.Variable(tf.truncated_normal([net[len(net)-1], n_output]) / net[i]))
    
    biases=[tf.Variable(tf.ones([net[0]]) * 0.05)]
    for i in range(len(net)-1):
        biases.append(tf.Variable(tf.ones([net[i+1]]) * 0.05))
    biases.append(tf.Variable(tf.ones([n_output]) * 0.05))
    
    x1 = tf.nn.dropout(x, input_keep_prob)
    for i in range(len(net)):
        x1 = tf.add(tf.matmul(x1, weights[i]), biases[i])   # x = wx+b
        x1 = tf.nn.relu(x1)                                 # x = max(0, x)
        x1 = tf.nn.dropout(x1, hidden_keep_prob)            # dropout layer
    pred = tf.matmul(x1, weights[len(net)]) + biases[len(net)]
    
    prediction = tf.sigmoid(pred)
    predicted_class = tf.greater(prediction, 0.5)
    correct = tf.equal(predicted_class, tf.equal(y,1.0))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
    accuracy_symbol=0;
    for i in range (n_input):
        prediction1 = tf.sigmoid(pred[i,:])
        predicted_class1 = tf.greater(prediction1, 0.5)
        correct1 = tf.equal(predicted_class1, tf.equal(y[i,:],1.0))
        accuracy1 = tf.reduce_mean(tf.cast(correct1, 'float'))
        #printvalue=tf.greater(accuracy1, 0.99)
    		       # print(printvalue.eval())
    if tf.greater(accuracy1, 0.99) is not None:
            accuracy_symbol+=1 
    		           # print(tf.greater(accuracy1, 0.99))
    		   # accuracy_symbol=accuracy_symbol/X_test.shape[0]
    print('Test Row Accuracy:', accuracy_symbol)
    binary=0
    saver = tf.train.Saver()
    with tf.Session() as sess:
    
            saver.restore(sess, thelocation)
            y_pred = sess.run(pred, feed_dict={x: measurement, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False})
    
            if binary==1:
                print('Test Accuracy:', accuracy.eval({x: measurement, y: attack, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False}))
               
            y_pred = tf.sigmoid(y_pred)
            y_pred = tf.greater(y_pred, 0.5)
            y_pred = tf.cast(y_pred, tf.int32)
            y_pred = y_pred.eval()
            sess.close()
    return y_pred
        # sio.savemat(save_name, {'input_h': X_test/10000000,'output_mode':Y_test,'output_mode_pred': y_pred})
        # # Compute the error for iteration t.
           


KK = 1                     # number of users
K= 2

# Load data
measurement = sio.loadmat('./data/embed_train%d' %K)['X_t']
attack = sio.loadmat('./data/embed_train%d' %K)['label_t']
measurement1 = sio.loadmat('./data/embed_train%d' %K)['X_s']
attack1 = sio.loadmat('./data/embed_train%d' %K)['label_s']
#measurement1 = sio.loadmat('./data/test%d' %KK)['newStateRe']
#attack1 = sio.loadmat('./data/test%d' %KK)['newlabel']



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

n_pts, n_labels = np.shape(attack)

 # append new classifiers to this list
alphas = []  # append the vote weight of the classifiers to this list

    # The weights for the first iteration


# setup and initialize the DNN network structure
current_weights = np.ones((n_pts, 1)) / float(n_pts)

iterations=2
for t in range(iterations):
        # A new classifier can be trained like this, given the current weights.
        #classifiers.append(base_classifier.train_classifier(X, labels, current_weights))

        # Do classification for each point.
        #vote_t = classifiers[-1].classify(X)
        

        
        thelocation="./DNNmodel/model_demo"+str(t+1)+".ckpt"
        net=netarray[t]
        print("The model is:", thelocation,"The net size is:",net)
        y_pred=test2train(net,thelocation,attack,measurement)
        error_t = 0.00001  # If we have 0 here, some alphas will be inf.
        for i in range(n_pts):
            if  np.any(y_pred[i] != attack[i]):
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
n_classifiers=2
    # If we only have one classifier, we may just classify directly.
votes = np.zeros((n_pts, n_labels)) #the vote result?

for classindex in range(3):
        thelocation="./DNNmodel/model_demo"+str(classindex+1)+".ckpt"
        net=netarray[t]
        y_pred=test2train(net,thelocation,attack1,measurement1)
        votes += alphas[classindex]*y_pred
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


