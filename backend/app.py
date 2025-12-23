from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend')

@app.route('/')
def serve_frontend():
    return send_from_directory(FRONTEND_DIR, 'index.html')

MODEL_FILE = "car_model.joblib"
CSV_FILE = "cars.csv"

def load_model():
    try:
        return joblib.load(MODEL_FILE)
    except FileNotFoundError:
        return None

@app.route('/api/options', methods=['GET'])
def get_options():
    model_data = load_model()
    if model_data is None:
        return jsonify({'error': 'Model not trained yet. Run train_model.py first'}), 400

    return jsonify({
        'brands': model_data['brands'],
        'models': model_data['models'],
        'fuel_types': model_data['fuel_types'],
        'transmissions': model_data['transmissions']
    })

@app.route('/api/models/<brand>', methods=['GET'])
def get_models_by_brand(brand):
    df = pd.read_csv(CSV_FILE)
    models = df[df['brand'] == brand]['model'].unique().tolist()
    return jsonify({'models': models})

@app.route('/api/predict', methods=['POST'])
def predict():
    model_data = load_model()
    if model_data is None:
        return jsonify({'error': 'Model not trained yet'}), 400

    data = request.json
    brand = data.get('brand')
    car_model = data.get('model')
    year = data.get('year')
    mileage = data.get('mileage')
    fuel_type = data.get('fuel_type')
    transmission = data.get('transmission')

    if not all([brand, car_model, year, mileage, fuel_type, transmission]):
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        brand_encoded = model_data['label_encoders']['brand'].transform([brand])[0]
        model_encoded = model_data['label_encoders']['model'].transform([car_model])[0]
        fuel_encoded = model_data['label_encoders']['fuel_type'].transform([fuel_type])[0]
        trans_encoded = model_data['label_encoders']['transmission'].transform([transmission])[0]
    except ValueError as e:
        return jsonify({'error': f'Unknown value: {str(e)}'}), 400

    features = np.array([[brand_encoded, model_encoded, int(year), int(mileage), fuel_encoded, trans_encoded]])
    prediction = model_data['model'].predict(features)[0]

    return jsonify({
        'predicted_price': round(prediction, 2),
        'input': {
            'brand': brand,
            'model': car_model,
            'year': year,
            'mileage': mileage,
            'fuel_type': fuel_type,
            'transmission': transmission
        }
    })

@app.route('/api/add', methods=['POST'])
def add_car():
    data = request.json
    required = ['brand', 'model', 'year', 'mileage', 'fuel_type', 'transmission', 'price']

    if not all(data.get(field) for field in required):
        return jsonify({'error': 'Missing required fields'}), 400

    df = pd.read_csv(CSV_FILE)
    new_row = pd.DataFrame([{
        'brand': data['brand'],
        'model': data['model'],
        'year': int(data['year']),
        'mileage': int(data['mileage']),
        'fuel_type': data['fuel_type'],
        'transmission': data['transmission'],
        'price': float(data['price'])
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

    return jsonify({'message': 'Car added successfully', 'total_cars': len(df)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
