from flask import Flask, render_template, request
import joblib
import numpy as np
import requests
from xml.etree import ElementTree as ET
import os
import pickle

app = Flask(__name__)

# Логирование для отладки
print("Starting Flask application...")

# Загрузка модели
try:
    with open('model/flat_price_model.pkl', 'rb') as f:
        model = joblib.load(f)
    print("Model loaded successfully")
except FileNotFoundError as e:
    print(f"Error: Model file 'model/flat_price_model.pkl' not found: {e}")
    raise
except pickle.UnpicklingError as e:
    print(f"Error: Failed to unpickle model file: {e}")
    raise
except Exception as e:
    print(f"Error loading model: {type(e).__name__}: {e}")
    raise

# Резервный курс доллара к рублю
DEFAULT_USD_TO_RUB = 78.37


def get_usd_to_rub_exchange_rate():
    print("Fetching USD/RUB exchange rate...")
    try:
        response = requests.get('http://www.cbr.ru/scripts/XML_daily.asp', timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for valute in root.findall(".//Valute[@ID='R01235']"):
            value = float(valute.find('Value').text.replace(',', '.'))
            print(f"Exchange rate fetched: {value}")
            return value
        print("USD not found in API response, using default rate")
        return DEFAULT_USD_TO_RUB
    except (requests.RequestException, ET.ParseError, ValueError) as e:
        print(f"Error fetching exchange rate: {e}. Using default rate {DEFAULT_USD_TO_RUB}")
        return DEFAULT_USD_TO_RUB


@app.route('/', methods=['GET', 'POST'])
def index():
    print("Handling request to /")
    prediction = None
    error = None
    form_data = {}

    if request.method == 'POST':
        print("Received POST request")
        try:
            form_data = {
                'totsp': request.form.get('totsp', ''),
                'livesp': request.form.get('livesp', ''),
                'kitsp': request.form.get('kitsp', ''),
                'dist': request.form.get('dist', ''),
                'metrdist': request.form.get('metrdist', ''),
                'walk': request.form.get('walk', '1'),
                'brick': request.form.get('brick', '1'),
                'floor': request.form.get('floor', '1')
            }
            print(f"Form data: {form_data}")

            totsp = float(form_data['totsp'])
            livesp = float(form_data['livesp'])
            kitsp = float(form_data['kitsp'])
            dist = float(form_data['dist'])
            metrdist = float(form_data['metrdist'])
            walk = int(form_data['walk'])
            brick = int(form_data['brick'])
            floor = int(form_data['floor'])

            print("Validating input...")
            if totsp <= 0:
                error = "Общая площадь должна быть положительным числом"
            elif livesp <= 0:
                error = "Жилая площадь должна быть положительным числом"
            elif kitsp <= 0:
                error = "Площадь кухни должна быть положительным числом"
            elif dist <= 0:
                error = "Расстояние до центра должно быть положительным числом"
            elif metrdist < 0:
                error = "Расстояние до метро не может быть отрицательным"
            elif walk not in [0, 1]:
                error = "Некорректное значение для 'Можно дойти до метро пешком'"
            elif brick not in [0, 1]:
                error = "Некорректное значение для 'Кирпичный дом'"
            elif floor not in [0, 1]:
                error = "Некорректное значение для 'Не первый этаж'"
            elif livesp > totsp:
                error = "Жилая площадь не может быть больше общей площади"
            elif kitsp > totsp:
                error = "Площадь кухни не может быть больше общей площади"
            else:
                print("Input validated, making prediction...")
                features = np.array([[totsp, livesp, kitsp, dist, metrdist, walk, brick, floor]])
                prediction_usd = model.predict(features)[0]
                usd_to_rub = get_usd_to_rub_exchange_rate()
                prediction_mln_rub = (prediction_usd * usd_to_rub) / 1000
                prediction = round(prediction_mln_rub, 2)
                print(f"Prediction: {prediction} млн руб")

        except ValueError:
            error = "Пожалуйста, введите корректные числовые значения для всех полей"
            print(f"Validation error: {error}")

    print("Rendering template...")
    return render_template('index.html', prediction=prediction, error=error, form_data=form_data)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"Running Flask on port {port}")
    app.run(host='0.0.0.0', port=port)