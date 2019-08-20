import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import exp,log


#..... PREPROCESSING ........#

dataset  = pd.read_csv("Social_Network_Ads.csv")

X = dataset.iloc[:,1:4].values
Y = dataset.iloc[:,-1].values

# Taking care of Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x = LabelEncoder()
X[:,0] = labelencoder_x.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(X)

X = X[:,1:]

X = np.append(arr = np.ones((400,1)), values = X, axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.25, random_state = 0)
Y_train,Y_test = Y_train.reshape(-1,1), Y_test.reshape(-1,1)

#........................................................#


#...... LOGISTIC REGRESSION CODE ...........#

alpha = 0.0001
params = np.array([1,1,1,1])
m = len(X_train)

def sigmoid(a):
      return  1/(1+np.exp(-a))

def hypothesis(features_i, parameters):
      result = features_i*parameters
      result = result.sum(axis = 0)
      return sigmoid(result)


def cost_fn(): # y = list of all results , hypo = list of all hypothesis due to all fatures h = g(theta * X)
      global m
      global X_train
      global params
      result = 0
      for i in range(0,m):
            hypo = hypothesis(X_train[i],params)
            result += (Y_train[i]*np.log(hypo) + (1-Y_train[i])*np.log(1-hypo)).item(0)
      result = (-result)/m
      return result


def gradient_descent():  # single gradient descent
      global alpha
      global params
      global m
      alpha = 0.01
      summation = 0
      global X_train
      no_of_params = len(params)
      temp_params = []
      for j in range(no_of_params):
            for i in range(m):
                 hypo = hypothesis(X_train[i], params).item(0)
                 summation += ((hypo - Y_train[i])*X_train[i][j]).item()
            temp = params[j].item(0) - alpha*summation
            temp_params.append(temp)
      temp_params = np.array(temp_params)
      params = temp_params[:]
     
      
      
def operation():
      
      prev_1 = cost_fn()
      gradient_descent()
      prev_2 = cost_fn()
      count = 0
      
      while(prev_2 < prev_1):
            prev_1 = cost_fn()
            gradient_descent()
            prev_2 = cost_fn()
            count += 1

def classify(hypo):
      if hypo>=0.5:
            return 1
      else:
            return 0

def predict():
      global X_test
      global params
      temp_list = []
      for x in range(len(X_test)):
            temp = (X_test[x]*params).reshape(1,-1).sum(axis = 1).item(0)
            temp_list.append(classify(temp))
      temp_array = np.array(temp_list)
      return temp_array


#......TESTING TEST VARAIBLES...........................#

operation() # Training the data
Y_pred = predict() #Testing th model
#..........................................................#

#........... SKLEARN LOGISTIC REGRESSOR ....................#

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(random_state = 0)
regressor.fit(X_train, Y_train)
YY_pred = regressor.predict(X_test)

#............COMPARING THE RESULTS........................#

from sklearn.metrics import confusion_matrix
cm_1 = confusion_matrix(Y_test, Y_pred)
#cm_2 = confusion_matrix(YY_pred, Y_pred)
cm_3 = confusion_matrix(Y_test, YY_pred)

#................. ACCURACY FROM COMFUSION MATRIX................#
def accuracy(cm):
    return (cm[0][0] + cm[1][1])/sum(sum(cm))

print(f"My model Results: {accuracy(cm_1)*100} %")
print(f"sklearn Results: {accuracy(cm_3)*100} %")




