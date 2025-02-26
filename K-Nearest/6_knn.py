import pandas as pd
import numpy as np
import statistics
import math

####### EUCLDEAN DISTANCE FORMULA #######
def distance_euc(s1, s2):
    d=0
    for i in range(len(s1)):
        d=d+(s1[i]-s2[i])**2
    d=math.sqrt(d)
    return d

####### MANHATTAN DISTANCE FORMULA #######
def distance_mnhtn(s1, s2):
    d=0
    for i in range(len(s1)):
        d=d+math.fabs(s1[i]-s2[i])
    return d

####### PREDICTION #######
def predict(X_train, Y_train, testpoint, k):
    dist_list = []
    for i in range(len(X_train)):
        dist = distance_euc(X_train[i], testpoint)  # Access X_train directly by index
        dist_list.append(dist)
    dist_indices = np.argsort(dist_list)[:k]        # argsort returns indices for sorting
    labels = Y_train[dist_indices]
    return statistics.mode(labels)


####### CSV TSHIRTS #######
data = pd.read_csv("D:/WORKSPACE/ML/Tshirts.csv")
X=data.iloc[:,[0,1]].values
Y=data.iloc[:,[2]].values.flatten() #FLATTENING Y
print('X = ', X)
print('Y = ', Y)

####### DATA TO NEURAL NETWORKS #######
from sklearn.model_selection import train_test_split

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# print('X Train = ', X_train.shape)
# Y_train = np.array(Y_train).flatten()
# print('Y Train = ', Y_train.shape)

# print('X Test = ', X_test.shape)
# Y_test = np.array(Y_test).flatten()
# print('Y Test = ', Y_test.shape)

# Ensure arrays are numpy arrays after splitting
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_train = np.array(X_train)  # Convert to numpy array
X_test = np.array(X_test)    # Convert to numpy array
Y_train = np.array(Y_train)  # Convert to numpy array
Y_test = np.array(Y_test)    # Convert to numpy array

print('X Train = ', X_train.shape)
print('Y Train = ', Y_train.shape)
print('X Test = ', X_test.shape)
print('Y Test = ', Y_test.shape)

####### K NEAREST #######
k = 3
Y_pred = []
for onesample in X_test:
    Y_pred.append(predict(X_train, Y_train, onesample, k))

# Accuracy calculation
correct = sum(1 for i in range(len(Y_test)) if Y_test[i] == Y_pred[i])
print("Actual Label =", Y_test)
print("Prediction of Model =", Y_pred)
acc = correct / len(Y_test) * 100
print("Accuracy of Model =", acc)