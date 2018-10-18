import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy
import dnn_kernal_wangshuoyao as dnn     # import our function file
import tensorflow as tf


def test2train(save_name):
    y_pred = sio.loadmat(save_name)['output_mode_pred']
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


classifiers = []
netarray=[]

# Save & Load model from this path
model_location = "./DNNmodel/model_demo1.ckpt"
save_name="./data/1Prediction_%d" % K
#export the weights and biases or not
export_weight_biase_sw=1

#hyper-parameter
net=[120,130]
netarray.append(net)
training_epochs=100  #training rate
regularizer=0.0005
batch_size=500
LR = 0.001   #learning rate
in_keep=1
hi_keep=1
LRdecay=1



print('Case: K=%d, Total Samples: %d, Total Iterations: %d, layers:%d\n'%(K, len(measurement), training_epochs,len(net)))

# Train the deep neural network
print('train DNN1 ...')
dnn.DNN_train(net,measurement, attack, measurement1, attack1,model_location,export_weight_biase_sw,regularizer,training_epochs,batch_size,LR,in_keep,hi_keep,LRdecay)
dnn.DNN_test(net,measurement1, attack1,  model_location, save_name)


# Save & Load model from this path
model_location2 = "./DNNmodel/model_demo2.ckpt"
save_name2="./data/2Prediction_%d" % K
#export the weights and biases or not
export_weight_biase_sw=1

#hyper-parameter
net=[120,80]
netarray.append(net)
training_epochs=100  #training rate
regularizer=0.0005
batch_size=400
LR = 0.001   #learning rate
in_keep=1
hi_keep=1
LRdecay=1
print('Case: K=%d, Total Samples: %d, Total Iterations: %d, layers:%d\n'%(K, len(measurement), training_epochs,len(net)))

# Train the deep neural network
print('train DNN2 ...')
dnn.DNN_train(net,measurement, attack, measurement1, attack1,model_location2,export_weight_biase_sw,regularizer,training_epochs,batch_size,LR,in_keep,hi_keep,LRdecay)
dnn.DNN_test(net,measurement1, attack1,  model_location2, save_name2)





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


