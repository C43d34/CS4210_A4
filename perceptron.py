#-------------------------------------------------------------------------
# AUTHOR: Cleighton Greer
# FILENAME: perception.py
# SPECIFICATION: Use Grid search for hyperparameters of Single-layer and Multi-layer Perception classifiers to find best accuracy
# FOR: CS 4210- Assignment #4
# TIME SPENT:  30min
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

                        #accuracy, learning rate, shuffle bool
cur_best_perceptron_itr = (0, 0, 0)
cur_best_MLP_itr = (0, 0, 0)
for learning_rate in n: #iterates over n

    for shuffle_flag in r: #iterates over r
        #iterates over both algorithms
        perception_classifiers = ["Single-Layer", "Multi-Layer"]

        for classifier in perception_classifiers: #iterates over the algorithms

            #Create a Neural Network classifier
            if (classifier == "Single-Layer"):
              clf = Perceptron(eta0 = learning_rate, shuffle = shuffle_flag, max_iter = 1000)    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            else:
              clf = MLPClassifier(activation = "logistic", learning_rate_init = learning_rate, hidden_layer_sizes = (25,), shuffle = shuffle_flag, max_iter = 1000) #use those hyperparameters: activation='logistic', learning_rate_init = learning rate, hidden_layer_sizes = number of neurons in the ith hidden layer,
                                    #  shuffle = shuffle the training data, max_iter=1000


            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            #for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            cur_best_accuracy = cur_best_perceptron_itr[0] if classifier == "Single-Layer" else cur_best_MLP_itr[0]
            correct_classifications = 0
            total_classifications = 0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                classification = clf.predict([x_testSample])[0]
                if (classification == y_testSample):
                  correct_classifications += 1
                total_classifications += 1
            
            accuracy = correct_classifications/total_classifications
            # print(f"Compare Accuracy for {classifier}, ({accuracy} vs {cur_best_accuracy})")
            if (accuracy > cur_best_accuracy):
                if (classifier == "Single-Layer"):
                    cur_best_perceptron_itr = (accuracy, learning_rate, shuffle_flag)
                else:
                    cur_best_MLP_itr = (accuracy, learning_rate, shuffle_flag)

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
        print(f"Highest Perception accuracy so far: {cur_best_perceptron_itr[0]:.3f}, Parameters: learning rate={cur_best_perceptron_itr[1]}, shuffle={cur_best_perceptron_itr[2]}")
        print(f"Highest MLP accuracy so far: {cur_best_MLP_itr[0]:.3f}, Parameters: learning rate={cur_best_MLP_itr[1]}, shuffle={cur_best_MLP_itr[2]}")
        print("")
        print("Test Next Configuration~~~~~~~~~~~~~~~~~~~~~~~~~~~")











