import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import joblib

# Загрузка данных
flats = pd.read_csv('flats_moscow.csv')

# Удаление ненужных столбцов
flats = flats.drop(['Unnamed: 0', 'code'], axis=1)

# Проверка данных
print(flats.info())

# Разделение данных на признаки и целевую переменную
X = flats.drop('price', axis=1)
y = flats['price']

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели
f_reg = LinearRegression()
f_reg.fit(X_train, y_train)

# Предсказания на тренировочной и тестовой выборках
y_pred_train = f_reg.predict(X_train)
y_pred_test = f_reg.predict(X_test)

# Оценка модели
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"RMSE на тренировочной выборке: {train_rmse}")
print(f"RMSE на тестовой выборке: {test_rmse}")

# Сохранение модели
joblib.dump(f_reg, 'model/flat_price_model.pkl')
print("Модель сохранена в 'model/flat_price_model.pkl'")