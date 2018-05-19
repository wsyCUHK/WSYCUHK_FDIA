from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
from collections import Counter
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import coverage_error, label_ranking_loss, hamming_loss, accuracy_score
class DNN(object)
    def _init_(self)
        self.trained=False

    def train_clasifier(self,X,labels,weights=None)
        rtn=DNN()



    def DNN_train(net,X_train,Y_train,X_valid,Y_valid,model_location,export_weight_biase_sw=0,regularizer=0.001,training_epochs=100, batch_size=100, LR= 0.001,in_keep=0.95,hi_keep=0.95,LRdecay=0):

            n_input = X_train.shape[1]                          # input size
            n_output = Y_train.shape[1]                         # output size
            num_train = X_train.shape[0]
            num_valid = X_valid.shape[0]
            print('train: %d ' % num_train, 'validation: %d ' % num_valid)

        # setup and initialize the DNN network structure
            x = tf.placeholder("float", [None, n_input])
            y = tf.placeholder("float", [None, n_output])
            is_train = tf.placeholder("bool")

            learning_rate = tf.placeholder(tf.float32, shape=[])
            total_batch = int((X_train.shape[0]+X_valid.shape[0]) / batch_size)
            input_keep_prob = tf.placeholder(tf.float32)
            hidden_keep_prob = tf.placeholder(tf.float32)

            weights=[tf.Variable(tf.truncated_normal([n_input, net[0]]) / np.sqrt(n_input))] 
            #net is input parameter of the size of the nerual network
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

        # train the DNN
            if regularizer==0:
                cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels =y, logits = pred))+0.5*tf.abs((tf.nn.zero_fraction(y)-tf.nn.zero_fraction(pred)))
            else:
                cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels =y, logits = pred))+0.5*tf.abs((tf.nn.zero_fraction(y)-tf.nn.zero_fraction(pred)))
                for i in range(len(weights)):
                    cost=cost+tf.contrib.layers.l2_regularizer(regularizer)(weights[i])

            optimizer = tf.train.AdamOptimizer(learning_rate, 0.09).minimize(cost)

            prediction = tf.sigmoid(pred)
            predicted_class = tf.greater(prediction, 0.5)
            correct = tf.equal(predicted_class, tf.equal(y,1.0))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(init)
                start_time = time.time()

                for epoch in range(training_epochs):
                    for i in range(total_batch):
                        idx = np.random.randint(num_train,size=batch_size)
                        if LRdecay==1:
                            _, c = sess.run([optimizer, cost], feed_dict={x: X_train[idx, :], y: Y_train[idx, :],
                                                                          input_keep_prob: in_keep, hidden_keep_prob: hi_keep,
                                                                          learning_rate: LR/(epoch+1), is_train: True})
                        elif LRdecay==0:
                            _, c = sess.run([optimizer, cost], feed_dict={x: X_train[idx, :], y: Y_train[idx, :],
                                                                              input_keep_prob: in_keep, hidden_keep_prob: hi_keep,
                                                                              learning_rate: LR, is_train: True})


                    if epoch%10==0:
                        accu, y_valid = sess.run([accuracy, pred], feed_dict={x: X_valid, y: Y_valid, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False})

                        print('epoch:%d, '%epoch, 'train:%0.4f'%accuracy.eval({x: X_train, y: Y_train, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False}), 'tcost:%0.4f'%cost.eval({x: X_train, y: Y_train, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False}),'validation:%0.4f:'%accuracy.eval({x: X_valid, y: Y_valid, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False}),'vcost:%0.4f:'%cost.eval({x: X_valid, y: Y_valid, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False}))
                print('epoch:%d, '%training_epochs, 'train:%0.4f'%accuracy.eval({x: X_train, y: Y_train, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False}), 'tcost:%0.4f'%cost.eval({x: X_train, y: Y_train, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False}),'validation:%0.4f:'%accuracy.eval({x: X_valid, y: Y_valid, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False}),'vcost:%0.4f:'%cost.eval({x: X_valid, y: Y_valid, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False}))

        # export the parameters of the trained DNN, which is further reproduced in MATLAB
                if export_weight_biase_sw==1:
                    WB={}
                    WB.setdefault('weight_1',sess.run(tf.Print(weights[0],[n_input ,net[0]])))
                    for a in range(len(net)-1):
                        WB.setdefault('weight_%d'%(a+2),sess.run(tf.Print(weights[a+1],[net[a] ,net[a+1]])))
                    WB.setdefault('weight_%d'%(len(net)+1),sess.run(tf.Print(weights[len(net)],[net[(len(net)-1)] ,n_output])))

                    WB.setdefault('biase_1',sess.run(tf.Print(biases[0],[net[0]])))
                    for a in range(len(net)-1):
                        WB.setdefault('biase_%d'%(a+2),sess.run(tf.Print(biases[a+1],[net[a+1]])))
                    WB.setdefault('biase_%d'%(len(net)+1),sess.run(tf.Print(biases[len(net)],[n_output])))


                    sio.savemat('./data/weights_biases',WB)

                print("Training Time: %0.2f s" % (time.time() - start_time))

                saver.save(sess, model_location)

            return 0
    

# Functions for deep neural network testing
def DNN_test(net,X_test, Y_test,  model_location, save_name, binary=0):
    tf.reset_default_graph()

# setup and initialize the DNN network structure
    n_input = X_test.shape[1]                          # input size
    n_output = Y_test.shape[1]                         # output size
    with tf.name_scope('inputs'):
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_output])
    is_train = tf.placeholder("bool")

    input_keep_prob = tf.placeholder(tf.float32)
    hidden_keep_prob = tf.placeholder(tf.float32)

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

   #  accuracy_symbol=0;
   #  for i in range (n_input):
   #      prediction1 = tf.sigmoid(pred[i,:])
   #      predicted_class1 = tf.greater(prediction1, 0.5)
   #      correct1 = tf.equal(predicted_class1, tf.equal(y[i,:],1.0))
   #      accuracy1 = tf.reduce_mean(tf.cast(correct1, 'float'))
   #      printvalue=tf.greater(accuracy1, 0.99)
   #      # print(printvalue.eval(session=sess))
   #      if tf.greater(accuracy1, 0.99) is not None:
   #          accuracy_symbol+=1 
   #         # print(tf.greater(accuracy1, 0.99))
    average_acc=[]
    arrayacc=predicted_class.eval()
    for i in range(n_input):
          average_acc.append(accuracy_score(y[i],arrayacc[i]))
    average_acc_total= np.average(average_acc)
    average_acc_row=average_acc.count(1)/predictions.shape[1]

    print('Test Row Accuracy:', average_acc_row, '\n Test Total Average Accuracy:',average_acc_total)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_location) # restore the saved DNN network

        start_time = time.time()
        y_pred = sess.run(pred, feed_dict={x: X_test, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False})

        testtime = time.time() - start_time

        if binary==1:
            print('Test Accuracy:', accuracy.eval({x: X_test, y: Y_test, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False}))
           

 
            y_pred = tf.sigmoid(y_pred)
            y_pred = tf.greater(y_pred, 0.5)
            y_pred = tf.cast(y_pred, tf.int32)
            y_pred = y_pred.eval()

        sio.savemat(save_name, {'input_h': X_test/10000000,'output_mode':Y_test,'output_mode_pred': y_pred})
    return testtime, y_pred




# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#          iterations - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def train_boost(base_classifier, X, labels, iterations=10):
    n_pts, n_dims = np.shape(X)

    classifiers = []  # append new classifiers to this list
    alphas = []  # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    current_weights = np.ones((n_pts, 1)) / float(n_pts)
    
    for t in range(0, iterations):
        # A new classifier can be trained like this, given the current weights.
        classifiers.append(base_classifier.train_classifier(X, labels, current_weights))

        # Do classification for each point.
        vote_t = classifiers[-1].classify(X)

        # Compute the error for iteration t.
        error_t = 0.00001  # If we have 0 here, some alphas will be inf.
        for i in range(n_pts):
            if vote_t[i] != labels[i]:
                error_t += current_weights[i]

        # Compute alpha for iteration t.
        alpha_t = 0.5 * (np.log(1 - error_t) - np.log(error_t))
        alphas.append(alpha_t)

        # Update the weights for next iteration, t + 1.
        for i in range(n_pts):
            factor = np.exp(-alpha_t) if vote_t[i] == labels[i] else np.exp(alpha_t)
            current_weights[i] *= factor
        current_weights / np.sum(current_weights)
        #print(current_weights)
    return classifiers, alphas


# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    n_labels - the number of different classes
# out:  y_pred - N vector of class predictions for test points
def classify_boost(X, classifiers, alphas, n_labels):
    n_pts = X.shape[0]
    n_classifiers = len(classifiers)

    # If we only have one classifier, we may just classify directly.
    if n_classifiers == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((n_pts, n_labels)) #the vote result?

        for t in range(n_classifiers):
            h = classifiers[t].classify(X) #the y-pred result of this classifier
            for i, xi_label in enumerate(h):
                votes[i][xi_label] += alphas[t] #alphas is the weight, the upper function learn the weights

        return np.argmax(votes, axis=1)


class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def train_classifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = train_boost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classify_boost(X, self.classifiers, self.alphas, self.nbr_classes)



class DNNClassifier(object)
    def init_Fx_variables(self):
        W1 = self.weight_variable([self.input_dim, self.net[0]], "weight_x1") # see above definition, one more feature level than Fe and Fd
        W2 = self.weight_variable([self.net[0], self.net[1]], "weight_x2")
        W3 = self.weight_variable([self.net[1], self.output_dim], "weight_x3")
        b1 = self.bias_variable([self.net[0]], "bias_x1")
        b2 = self.bias_variable([self.net[1]], "bias_x2")
        b3 = self.bias_variable([self.output_dim], "bias_x3")
        return W1, W2, W3, b1, b2, b3

    # def init_Fe_variables(self):
    #     W1 = self.weight_variable([self.config.labels_dim, self.config.solver.hidden_dim], "weight_e1")
    #     W2 = self.weight_variable([self.config.solver.hidden_dim, self.config.solver.latent_embedding_dim], "weight_e2")
    #     b1 = self.bias_variable([self.config.solver.hidden_dim], "bias_e1")
    #     b2 = self.bias_variable([self.config.solver.latent_embedding_dim], "bias_e2") 
    #     return W1, W2, b1, b2

    # def init_Fd_variables(self):
    #     W1 = self.weight_variable([self.config.solver.latent_embedding_dim, self.config.solver.hidden_dim], "weight_d1")
    #     W2 = self.weight_variable([self.config.solver.hidden_dim, self.config.labels_dim], "weight_d2")
    #     b1 = self.bias_variable([self.config.solver.hidden_dim], "bias_d1")
    #     b2 = self.bias_variable([self.config.labels_dim], "bias_d2")   # labels dim is different from latent embedding dim
    #     return W1, W2, b1, b2

    def accuracy(self, y_pred, y):
        return tf.reduce_mean(tf.cast(tf.equal(tf.round(y_pred), y), tf.float32))

    def Fx(self, X, keep_prob): #feature mapping function
        hidden1 = tf.nn.dropout(utils.leaky_relu(tf.matmul(X, self.Wx1) + self.bx1), keep_prob) # x input, 2 hidden, just like normal
        hidden2 = tf.nn.dropout(utils.leaky_relu(tf.matmul(hidden1, self.Wx2) + self.bx2), keep_prob)
        hidden3 = tf.nn.dropout(utils.leaky_relu(tf.matmul(hidden2, self.Wx3) + self.bx3), keep_prob)
        return hidden3

    # def Fe(self, Y, keep_prob): #encoding function
    #     hidden1 = tf.nn.dropout(utils.leaky_relu(tf.matmul(Y, self.We1)) + self.be1, keep_prob) # y input, 1 hidden, inverse
    #     pred = tf.nn.dropout(utils.leaky_relu(tf.matmul(hidden1, self.We2) + self.be2), keep_prob)   #pred the latent value (embedded label) ??????
    #     return pred

    # def Fd(self, input, keep_prob): #decoding function
    #     hidden1 = tf.nn.dropout(utils.leaky_relu(tf.matmul(input, self.Wd1) + self.bd1), keep_prob) #the encoded value as input, 1 hidden, re-predect the label value
    #     y_pred = tf.matmul(hidden1, self.Wd2) + self.bd2
    #     return y_pred

    def prediction(self, X, keep_prob):
        Fx = self.Fx(X, keep_prob)
        return self.Fd(Fx, keep_prob)

   # def embedding_loss(self, Fx, Fe):
      # Ix, Ie = tf.eye(tf.shape(Fx)[0]), tf.eye(tf.shape(Fe)[0])
        #C1, C2, C3 = tf.abs(Fx - Fe), tf.matmul(Fx, tf.transpose(Fx)) - Ix, tf.matmul(Fe, tf.transpose(Fe)) - Ie
    #    return tf.reduce_mean(tf.square(Fx - Fe)) #tf.trace(tf.matmul(C1, tf.transpose(C1))) + self.config.solver.lagrange_const * tf.trace(tf.matmul(C2, tf.transpose(C2))) + self.config.solver.lagrange_const * tf.trace(tf.matmul(C3, tf.transpose(C3)))

    # My blood, sweat and tears were also embedded into the emebedding.
    def output_loss(self, predictions, labels):
        Ei = 0.0
        i, cond = 0, 1
        while cond == 1:
            cond = tf.cond(i >= tf.shape(labels)[0] - 1, lambda: 0, lambda: 1)
            prediction_, Y_ = tf.slice(predictions, [i, 0], [1, self.config.labels_dim]), tf.slice(labels, [i, 0], [1, self.config.labels_dim])
            zero, one = tf.constant(0, dtype=tf.float32), tf.constant(1, dtype=tf.float32)
            ones, zeros = tf.gather_nd(prediction_, tf.where(tf.equal(Y_, one))), tf.gather_nd(prediction_, tf.where(tf.equal(Y_, zero)))
            y1 = tf.reduce_sum(Y_)
            y0 = Y_.get_shape().as_list()[1] - y1
            temp = (1/y1 * y0) * tf.reduce_sum(tf.exp(-(tf.reduce_sum(ones) / tf.cast(tf.shape(ones)[0], tf.float32) - tf.reduce_sum(zeros) / tf.cast(tf.shape(zeros)[0], tf.float32))))
            Ei += tf.cond(tf.logical_or(tf.is_inf(temp), tf.is_nan(temp)), lambda : tf.constant(0.0), lambda : temp)
            i += 1
        return Ei 

    def cross_loss(self, features, labels, keep_prob):
        predictions = self.prediction(features, keep_prob)
        Fx = self.Fx(features, keep_prob)
        Fe = self.Fe(labels, keep_prob)
        cross_loss = tf.add(tf.log(1e-10 + tf.nn.sigmoid(predictions)) * labels, tf.log(1e-10 + (1 - tf.nn.sigmoid(predictions))) * (1 - labels))
        cross_entropy_label = -1 * tf.reduce_mean(tf.reduce_sum(cross_loss, 1))
        return cross_entropy_label

    def loss(self, features, labels, keep_prob):
        lamda = 0.00
        prediction = tf.nn.sigmoid(self.prediction(features, keep_prob))
        #Fx = self.Fx(features, keep_prob)
        #Fe = self.Fe(labels, keep_prob)
        l2_norm = tf.reduce_sum(tf.square(self.Wx1)) + tf.reduce_sum(tf.square(self.Wx2)) + tf.reduce_sum(tf.square(self.Wx3)) +))
        return self.output_loss(prediction, labels) + lamda * l2_norm # self.cross_loss(features, labels, keep_prob) 

    def train_step(self, loss):
        optimizer = self.config.solver.optimizer
        return optimizer(self.config.solver.learning_rate).minimize(loss)




testClassifier(BoostClassifier())