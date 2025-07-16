import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
import os

# Создание папки model, если она не существует
os.makedirs('model', exist_ok=True)

try:
    # Загрузка данных
    flats = pd.read_csv('flats_moscow.csv')
    flats = flats.drop(['Unnamed: 0', 'code'], axis=1, errors='ignore')
except FileNotFoundError:
    print("Error: flats_moscow.csv not found")
    exit(1)
except KeyError as e:
    print(f"Error: Invalid columns in dataset: {e}")
    exit(1)

# Подготовка данных
try:
    X = flats.drop('price', axis=1)
    y = flats['price']
except KeyError as e:
    print(f"Error: Column 'price' not found in dataset: {e}")
    exit(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
f_reg = LinearRegression()
f_reg.fit(X_train, y_train)

# Оценка модели
y_pred_train = f_reg.predict(X_train)
y_pred_test = f_reg.predict(X_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f'Train RMSE: {train_rmse}, Test RMSE: {test_rmse}')

# Сохранение модели
try:
    with open('model/flat_price_model.pkl', 'wb') as f:
        pickle.dump(f_reg, f)
    print("Model saved successfully to model/flat_price_model.pkl")
except Exception as e:
    print(f"Error saving model: {e}")
    exit(1)