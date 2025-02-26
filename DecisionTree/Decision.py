import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('D:\\WORKSPACE\\ML\\diabetes_data.csv')
X = data.iloc[:,[0,1,2,3,4,5,6,7]].values
Y = data.iloc[:,[8]].values

Y = np.ndarray.flatten(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

clf = tree.DecisionTreeClassifier(criterion = 'entropy') #gini
clf.fit(X_train, Y_train)
pred = clf.predict(X_test)

print("Accuracy of Test: ", accuracy_score(Y_test, pred))

import matplotlib.pyplot as plt

f = ['Surgeries', 'glucose', 'BP', 'Skin Thickness', 'Insulin', 'BMI', 'DPF', 'Age']
plt.figure(figsize=(30,10))
t = tree.plot_tree(clf, feature_names = f, fontsize = 8)
plt.show()
