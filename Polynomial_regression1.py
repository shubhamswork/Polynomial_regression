# Polynomial Regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing datasets
dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,[1]].values
Y=dataset.iloc[:,[2]].values

from sklearn.preprocessing import PolynomialFeatures
regressor2=PolynomialFeatures(degree=4)
X_poly=regressor2.fit_transform(X)

# using linear regression model
from sklearn.linear_model import LinearRegression
regressor1=LinearRegression()
regressor1.fit(X,Y)
Y_pred1=regressor1.predict(6.5)

# Using Polynomial Regression
regressor_poly=LinearRegression()
regressor_poly.fit(X_poly,Y)
Y_pred2=regressor_poly.predict(regressor2.fit_transform(6.5))


plt.scatter(X,Y,color="red")
plt.plot(X,regressor_poly.predict(regressor2.fit_transform(X)),color="blue")