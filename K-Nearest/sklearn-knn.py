# import pandas as pd
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score

# # Load dataset
# data = pd.read_csv("iris_data.csv")
# x = data.iloc[:,[0,1,2,3]].values
# y = data.iloc[:,[4]].values
# y=np.ndarray.flatten(y)
# print(y)
# kf=KFold(n_splits=5, random_state = 40, shuffle = True)
# clf=KNeighborsClassifier(n_neighbors = 6)
# a = []
# for train_indexes, test_indexes in kf.split(x):
#     x_train, x_test = x[train_indexes], x[test_indexes]
#     y_train, y_test = y[train_indexes], y[test_indexes]
#     clf.fit(x_train, y_train)
#     y_pred = clf.predict(x_test)
#     # acc = accuracy_score(y_test, y_pred)
#     a.append(accuracy_score(y_test, y_pred))

# print("Accuracy: ", np.mean(a))




import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("iris_data.csv")
x = data.iloc[:,[0,1,2,3]].values
y = data.iloc[:,[4]].values
y = np.ndarray.flatten(y)

kf = KFold(n_splits=5, random_state=40, shuffle=True)

k_values = [1,2,3,4,5,6,7,8,9,10]
accuracies = []

for k in k_values:
    clf = KNeighborsClassifier(n_neighbors=k)
    fold_accuracies = []
    
    for train_indexes, test_indexes in kf.split(x):
        x_train, x_test = x[train_indexes], x[test_indexes]
        y_train, y_test = y[train_indexes], y[test_indexes]
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        fold_accuracies.append(accuracy_score(y_test, y_pred))

    mean_accuracy = np.mean(fold_accuracies)
    accuracies.append(mean_accuracy)
    print(f"Accuracy for k={k}: {mean_accuracy:.3f}")

plt.plot(k_values, accuracies, marker='o')
plt.title('KNN Accuracy vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()
