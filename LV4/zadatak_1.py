import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import sklearn.linear_model as lm
from sklearn import metrics

data = pd.read_csv('data_C02_emission.csv')
# a)

X = data[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)']]
y = data[['CO2 Emissions (g/km)']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# b)

plt.scatter(X_train[['Engine Size (L)']], y_train, color='blue', s=10)
plt.scatter(X_test[['Engine Size (L)']], y_test, color='red', s=10)
plt.legend(['train', 'test'])
plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 Emissions (g/km)')


# c)

sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

X_train_n = pd.DataFrame(X_train_n, columns=X_train.columns)
X_test_n = pd.DataFrame(X_test_n, columns=X_test.columns)

plt.figure()
plt.hist(X_train['Engine Size (L)'])
plt.title('Engine size histogram (L)')

plt.hist(X_train_n['Engine Size (L)'])
plt.title('Engine size histogram (L)')


# d)

linearModel = lm.LinearRegression()
linearModel.fit(X_train_n , y_train)
print(linearModel.coef_)
print(linearModel.intercept_)


# e)
y_test_p = linearModel.predict(X_test_n)

plt.figure()
plt.scatter(X_test['Fuel Consumption Hwy (L/100km)'], y_test, s=10)
plt.scatter(X_test['Fuel Consumption Hwy (L/100km)'], y_test_p, s=10)
plt.xlabel('CO2 Emissions (g/km)')
plt.ylabel('Fuel Consumption Hwy (L/100km)')
plt.title('Real Data vs Predicted Data')
plt.legend(['real data', 'predicted data'])


# f)
mae = metrics.mean_absolute_error(y_test, y_test_p)
mse = metrics.mean_squared_error(y_test, y_test_p)
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE): ", mae)
print("Mean Squared Error (MSE): ", mse)
print("Root Mean Squared Error (RMSE): ", rmse)


# g)
# half of the train values
print("-------(1/2)-------")
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n[0:int(len(X_train_n)/2)], y_train[0:int(len(y_train)/2)])
print(linearModel.coef_)
print(linearModel.intercept_)
y_test_p = linearModel.predict(X_test_n)

mae = metrics.mean_absolute_error(y_test, y_test_p)
mse = metrics.mean_squared_error(y_test, y_test_p)
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE): ", mae)
print("Mean Squared Error (MSE): ", mse)
print("Root Mean Squared Error (RMSE): ", rmse)

# quarter of the train values
print("-------(1/4)-------")
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n[0:int(len(X_train_n)/4)], y_train[0:int(len(y_train)/4)])
print(linearModel.coef_)
print(linearModel.intercept_)
y_test_p = linearModel.predict(X_test_n)

mae = metrics.mean_absolute_error(y_test, y_test_p)
mse = metrics.mean_squared_error(y_test, y_test_p)
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE): ", mae)
print("Mean Squared Error (MSE): ", mse)
print("Root Mean Squared Error (RMSE): ", rmse)

# 4.5.2

ohe = OneHotEncoder()
data_encoded = data[['Make', 'Model', 'Vehicle Class', 'Engine Size (L)', 'Cylinders', 'Transmission', 'Fuel Type', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)']]
data_encoded['Make'] = ohe.fit_transform(data['Make']).toarray()
data_encoded['Model'] = ohe.fit_transform(data['Model']).toarray()
data_encoded['Vehicle'] = ohe.fit_transform(data['Vehicle']).toarray()
data_encoded['Class'] = ohe.fit_transform(data['Class']).toarray()
data_encoded['Transmission'] = ohe.fit_transform(data['Transmission']).toarray()
data_encoded['Fuel Type'] = ohe.fit_transform(data['Fuel Type']).toarray()

y = data[['CO2 Emissions (g/km)']]

X_train, X_test, y_train, y_test = train_test_split(data_encoded, y, test_size=0.2, random_state=1)


linearModel = lm.LinearRegression()
linearModel.fit(data_encoded, y_train)
print(linearModel.coef_)
print(linearModel.intercept_)
y_test_p = linearModel.predict(X_test_n)


plt.show()