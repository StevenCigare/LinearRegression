import numpy as np
import pandas as pd
from linear_regression import LinearRegression
from sklearn import model_selection
import matplotlib.pyplot as plt

mydata = pd.read_csv("AttendanceMarksSA.csv")

#single
X = mydata[["MSE"]].copy()
y = mydata[["ESE"]].copy()
X["intersect"] = 1
X = X[["intersect", "MSE"]]
# #multiple
# X = mydata[["Attendance", "MSE"]].copy()
# y = mydata[["ESE"]].copy()
# X["intersect"] = 1
# X = X[["intersect", "Attendance", "MSE"]]

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

linear_model = LinearRegression()
coefficients = linear_model.train(x_train, y_train)

predictions = linear_model.predict(x_test)
r_squared = linear_model.r_squared(predictions, y_test)


plt.plot(x_test['MSE'], y_test, 'bo')
plt.plot(x_test['MSE'], predictions)
plt.plot()
plt.show()

print(r_squared)
