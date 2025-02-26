import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('diamonds.csv')
print(data)
print()
print()
print("=========================")
print("DETAILED INFO ABOUT DATA: ")
print("=========================")
print(data.info())
print()
print()
print("=====================")
print("CHECK EMPTY RECORDS: ")
print("=====================")
print(data.isnull().sum())
print()
print()
print("====================")
print("DESCRIPTION OF DATA: ")
print("====================")
print(data.describe())

data=data.drop(data[data["x"]==0].index)
data=data.drop(data[data["y"]==0].index)
data=data.drop(data[data["z"]==0].index)

s=(data.dtypes=='object')
print()
print()
print("===================")
print("DATA IN s VARIABLE: ")
print("===================")
print(s)

object_cols=list(s[s].index)
print()
print()
print("=================")
print("CATEGORICAL DATA: ")
print("=================")
print(object_cols)

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()

for c in object_cols:
    data[c]=lb.fit_transform(data[c])

X=data.drop(['price'], axis=1)
Y=data['price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
lr=LinearRegression()
lr.fit(X_train, Y_train)
print()
print()
print("==============")
print("TRAINING DATA: ")
print("==============")
pred_train = lr.predict(X_train)
print("MSE: ", mean_squared_error(Y_train,pred_train))
print("MAE: ", mean_absolute_error(Y_train,pred_train))
print("R2 Score: ", r2_score(Y_train,pred_train))

print()
print()
print("=============")
print("TESTING DATA: ")
print("=============")
pred_test=lr.predict(X_test)
print("MSE: ", mean_squared_error(Y_test,pred_test))
print("MAE: ", mean_absolute_error(Y_test,pred_test))
print("R2 Score: ", r2_score(Y_test,pred_test))
