import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

CSV_FILE = "cars.csv"

def train_model():
    df = pd.read_csv(CSV_FILE)

    if len(df) < 10:
        print("Error: Need at least 10 cars in the dataset to train")
        return

    print(f"Training on {len(df)} cars...")

    label_encoders = {}
    categorical_cols = ['brand', 'model', 'fuel_type', 'transmission']

    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df[['brand_encoded', 'model_encoded', 'year', 'mileage', 'fuel_type_encoded', 'transmission_encoded']]
    Y = df['price']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)

    print(f"Model trained successfully!")
    print(f"Mean Absolute Error: {mae:,.0f}")
    print(f"R2 Score: {r2:.2f}")

    joblib.dump({
        'model': model,
        'label_encoders': label_encoders,
        'brands': list(label_encoders['brand'].classes_),
        'models': list(label_encoders['model'].classes_),
        'fuel_types': list(label_encoders['fuel_type'].classes_),
        'transmissions': list(label_encoders['transmission'].classes_)
    }, 'car_model.joblib')

    print("Model saved to car_model.joblib")

if __name__ == "__main__":
    train_model()
