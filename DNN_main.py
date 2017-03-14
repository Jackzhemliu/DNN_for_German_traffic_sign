# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:37:12 2017

@author: zliu1
"""
import time 
import csv
import cv2
import os 
import scipy as sp
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import confusion_matrix, accuracy_score 

##### for training and predicting 
def plot_learning_curve(estimator, title, X, y, cv=None,
                            train_sizes=np.linspace(0.1,
                            1.0,5 )):
        plt.figure() 
        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, 
                                                               y, cv=cv, n_jobs=1,
                                                               train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std= np.std(train_scores, axis=1)
        test_scores_mean=np.mean(test_scores, axis=1)
        test_scores_std=np.std(test_scores, axis=1)
        plt.grid() 
        
        plt.fill_between(train_sizes, train_scores_mean-train_scores_std, 
                        train_scores_mean+train_scores_std, alpha=0.1, 
                         color='r')
        plt.fill_between(train_sizes, test_scores_mean-test_scores_std, 
                        test_scores_mean+test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', 
                label='Training score')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
                label="testing score")
        plt.legend(loc="best")
        return plt 
        
        
#################### read data       
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    locations = []
    roi_images = [] 
    database = [] 
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        gtReader.next() # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            full = cv2.imread(prefix + row[0], 0)
            images.append(full) # the 1th column is the filename
            regions = [int(row[3]), int(row[4]), int(row[5]), int(row[6])]
            locations.append(regions)
            imag_roi = full[regions[0]:regions[2], regions[1]:regions[3]] 
            temp = cv2.resize(imag_roi,(32,32))
            roi_images.append(temp)
            labels.append(int(row[7])) # the 8th column is the label
            database.append(temp.flatten())
        gtFile.close()
    return images, labels, roi_images, locations, database
        
        
###########################################################
def training(X_all, y_all, testsize= 0.3):
    X = np.array(X_all)
    y = 1.0*np.array(y_all)
    
    X = X/255.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testsize)
    
    hidden1_neurons = 800 
    hidden2_neurons = 400
    hidden3_neurons = 100
    mlp=MLPClassifier(hidden_layer_sizes=(hidden1_neurons,hidden2_neurons, hidden3_neurons,),
                    activation=('logistic'), max_iter=10000,alpha=0.0001, solver='lbfgs', 
                    verbose=0,tol=0.000000001, momentum=0.1, 
                    learning_rate_init=0.01)
    
    t0 = time.time() 
    mlp.fit(X_train, y_train)
    t1 = time.time() 
    delta_t = (t1 - t0)/3600.0
    y_pred = mlp.predict(X_test)
    print delta_t 
    print mlp.score(X_train, y_train)
    print mlp.score(X_test, y_test) 
    return y_pred, mlp, delta_t, X_test, y_test

##################################################################################

def store_wrong_prediction(y_pred, y_test, X_test): 
    errors = 0 
    wrong_label = []  
    error_instances = [] 
    ids = [] 
    for i in range(len(y_pred)): 
        if (abs(y_pred[i] - y_test[i]) > 0.1):
            errors = errors + 1
            ids.append(i)
            wrong_label.append(i)
            temp = (X_test[i, :]).reshape(-1,32)*255.
            error_instances.append(temp)
    
    path =  os.getcwd() 
    f = open('error_predict/error_label.csv', 'wb')
    for i in range(len(error_instances)):
        image_id = ids[i]
        cv2.imwrite(path+'/error_predict/img_'+str(image_id)+'.jpg', error_instances[i])
        f.write(str(y_pred[image_id]) + ',' + str(y_test[image_id]) + '\n')
    
    f.close() 
    return errors, error_instances, ids
    
#################################################################################
def main():
    rootpath = os.getcwd() 
    images, y_all, roi_images, locations, X_all = readTrafficSigns(rootpath)
    y_pred, mlp, delta_t, X_test, y_test = training(X_all, y_all)
    errors, error_instances, ids = store_wrong_prediction(y_pred,y_test, X_test)
    
if __name__ == "__main__":
    main() 

    


