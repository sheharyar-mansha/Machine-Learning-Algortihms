import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

data = pd.read_csv("D:\\WORKSPACE\\ML\\weather.csv")

print(data)

le = LabelEncoder()
label_outlook = le.fit_transform(data['Outlook'])
data.drop('Outlook', axis = 1, inplace = True)
data['Outlook'] = label_outlook

# print(data)

label_temp = le.fit_transform(data['Temp'])
data.drop('Temp', axis = 1, inplace = True)
data['Temp'] = label_temp

label_humidity = le.fit_transform(data['Humidity'])
data.drop('Humidity', axis = 1, inplace = True)
data['Humidity'] = label_humidity

label_windy = le.fit_transform(data['Windy'])
data.drop('Windy', axis = 1, inplace = True)
data['Windy'] = label_windy

label_play = le.fit_transform(data['Play'])
data.drop('Play', axis = 1, inplace = True)
data['Play'] = label_play

print(data)

X = data.iloc[:,[0,1,2,3]].values
Y = data.iloc[:,[4]].values

Y = np.ndarray.flatten(Y)

############ MODEL ############
model = GaussianNB()
cv = LeaveOneOut()

Y_pred = []
Y_actual = []

for train_idx, test_idx in cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    model.fit(X_train, Y_train)
    Y_pred.append(model.predict(X_test))
    Y_actual.append(Y_test[0])

print("ACCURACY: ", round(accuracy_score(Y_actual, Y_pred), 3))