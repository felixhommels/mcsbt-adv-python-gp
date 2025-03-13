import os
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Loading Data into the Model and dropping the Order_ID column
data = pd.read_csv("data/Food_Delivery_times.csv")

data.drop(columns=["Order_ID"], inplace=True)
data.dropna(inplace=True)

#We have some categorical variables - for linear regression we need to convert them to numerical variables
data = pd.get_dummies(data)

X = data.drop(columns=["Delivery_Time_min"])
Y = data["Delivery_Time_min"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

model = LinearRegression()

model.fit(X_train, Y_train)

#During development wanted to see the coefficients of the different features
coefficients = model.coef_
feature_names = X.columns

for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef}")

#Predicting the test set results
Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}") #During testing was approx .83 which is acceptable

#Saving the model to pickle file
pickle.dump(model, open("food_delivery_model.pkl", "wb"))











