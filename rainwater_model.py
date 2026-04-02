# Rainwater Harvesting Prediction Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# loading dataset
data = pd.read_csv("rainwater_dataset.csv")

print("First 5 rows of dataset:")
print(data.head())

# separating input and output
X = data[['rainfall_mm', 'area_m2', 'runoff_coeff']]
y = data['harvested_water_liters']

# splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# creating models
lr_model = LinearRegression()
rf_model = RandomForestRegressor()

# training models
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# predictions
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# checking error
lr_error = np.sqrt(mean_squared_error(y_test, lr_pred))
rf_error = np.sqrt(mean_squared_error(y_test, rf_pred))

print("\nLinear Regression Error:", lr_error)
print("Random Forest Error:", rf_error)

# choosing better model
if rf_error < lr_error:
    model = rf_model
    print("Using Random Forest")
else:
    model = lr_model
    print("Using Linear Regression")

# user input prediction
rain = float(input("\nEnter rainfall (mm): "))
area = float(input("Enter area (m^2): "))
runoff = float(input("Enter runoff coefficient (0-1): "))

result = model.predict([[rain, area, runoff]])

print("\nEstimated water collected:", round(result[0], 2), "liters")

# graph
plt.scatter(y_test, rf_pred)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Actual vs Predicted")
plt.show()