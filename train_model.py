import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

flats = pd.read_csv('flats_moscow.csv')
flats = flats.drop(['Unnamed: 0', 'code'], axis=1)
X = flats.drop('price', axis=1)
y = flats['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
f_reg = LinearRegression()
f_reg.fit(X_train, y_train)
y_pred_train = f_reg.predict(X_train)
y_pred_test = f_reg.predict(X_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f'Train RMSE: {train_rmse}, Test RMSE: {test_rmse}')
joblib.dump(f_reg, 'model/flat_price_model.pkl')