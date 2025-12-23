# CarValueAI - Car Price Prediction System

An AI-powered web application that predicts used car prices using Machine Learning.

## What It Does

CarValueAI analyzes 4,340+ real car listings to predict the market value of used cars based on:
- Brand & Model
- Year of manufacture
- Mileage (km)
- Fuel type (Petrol, Diesel, CNG, LPG, Electric)
- Transmission (Manual, Automatic)

## Features

- **Instant Price Prediction** - Get car valuations in under 1 second
- **AI-Powered** - Uses Random Forest Regressor algorithm (70% accuracy)
- **Real Data** - Trained on 4,340+ actual car listings
- **29 Brands Supported** - Including Toyota, Honda, BMW, Mercedes, Audi, and more
- **Add New Data** - Contribute to the database to improve predictions

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, Flask |
| ML Model | scikit-learn (Random Forest) |
| Database | CSV |
| Frontend | HTML, CSS, JavaScript |
| HTTP Client | Axios |

## Project Structure

```
231225/
├── backend/
│   ├── app.py              # Flask API server
│   ├── train_model.py      # Model training script
│   ├── cars.csv            # Car database (4,340+ records)
│   ├── car_model.joblib    # Trained ML model
│   └── requirements.txt    # Python dependencies
├── frontend/
│   └── index.html          # Web interface
├── venv/                   # Python virtual environment
└── README.md
```

## Installation & Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd 231225
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Train the model
```bash
cd backend
python train_model.py
```

### 5. Start the server
```bash
python app.py
```

### 6. Open in browser
```
http://127.0.0.1:5000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve frontend |
| GET | `/api/options` | Get available brands, fuel types, transmissions |
| GET | `/api/models/<brand>` | Get models for a specific brand |
| POST | `/api/predict` | Predict car price |
| POST | `/api/add` | Add new car to database |

### Example: Predict Price
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Honda",
    "model": "Honda City",
    "year": 2018,
    "mileage": 50000,
    "fuel_type": "Petrol",
    "transmission": "Manual"
  }'
```

### Example: Add Car
```bash
curl -X POST http://127.0.0.1:5000/api/add \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Toyota",
    "model": "Camry",
    "year": 2020,
    "mileage": 30000,
    "fuel_type": "Petrol",
    "transmission": "Automatic",
    "price": 15000
  }'
```

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest Regressor |
| R2 Score | 0.70 (70% accuracy) |
| Mean Absolute Error | $1,265 |
| Training Data | 4,340 cars |

## Supported Brands

Ambassador, Audi, BMW, Chevrolet, Daewoo, Datsun, Fiat, Force, Ford, Honda, Hyundai, Isuzu, Jaguar, Jeep, Kia, Land, MG, Mahindra, Maruti, Mercedes-Benz, Mitsubishi, Nissan, OpelCorsa, Renault, Skoda, Tata, Toyota, Volkswagen, Volvo

## Data Source

Car price dataset from [YBIFoundation](https://github.com/YBIFoundation/Dataset) - Real used car listings with prices converted to USD.

## License

MIT
