import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime


def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def prepare_data(X, y, timesteps=3):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:(i + timesteps)])
        ys.append(y[i + timesteps])
    return np.array(Xs), np.array(ys)


def predict_duration(model, scaler_X, scaler_y, start_lat, start_lon, end_lat, end_lon, date_str):
    # Convert date format from YYYY-MM-DD to MM-DD
    date = datetime.strptime(date_str, '%Y-%m-%d')
    date_str_mmdd = date.strftime('%m-%d')
    print(f"Converted date: {date_str_mmdd}")

    X_input = np.array([[start_lat, start_lon, end_lat, end_lon]])
    X_scaled = scaler_X.transform(X_input)
    X_reshaped = np.reshape(X_scaled, (1, 1, X_scaled.shape[1]))  # Reshape for LSTM model
    y_pred_scaled = model.predict(X_reshaped)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    return y_pred[0, 0]


# Main code
data_dir = r"C:\Users\Selim\Desktop\ml-project\data\TransactionDatabases_lstm"
print(f"Debug: Checking directory: {data_dir}")
print(f"Debug: Directory exists: {os.path.exists(data_dir)}")
print(f"Debug: Files in directory: {os.listdir(data_dir)}")

# Store trained models and scalers
models = {}
scalers = {}

print("Debug: Starting file processing")
print(f"Debug: JSON files found: {[f for f in os.listdir(data_dir) if f.endswith('.json')]}")

for file in os.listdir(data_dir):
    if file.endswith(".json"):
        file_path = os.path.join(data_dir, file)
        print(f"Debug: Processing file {file}")

        # Read JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Convert JSON to DataFrame
        df = pd.DataFrame(data)
        print(f"Debug: File {file} shape: {df.shape}")
        print(f"Debug: File {file} columns: {df.columns.tolist()}")

        prediction_types = [
            ("supplier", ["Truck_Latitude", "Truck_Longitude", "Supplier_Latitude", "Supplier_Longitude",
                          "Duration_To_Supplier(h)"]),
            ("port",
             ["Supplier_Latitude", "Supplier_Longitude", "Port_Latitude", "Port_Longitude", "Duration_To_Port(h)"]),
            ("customer", ["Tarragona_Latitude", "Tarragona_Longitude", "Customer_Latitude", "Customer_Longitude",
                          "Duration_To_Customer(h)"])
        ]

        for pred_type, columns in prediction_types:
            print(f"Debug: Training model for {pred_type}")
            if all(col in df.columns for col in columns):
                X = df[columns[:-1]]
                y = df[columns[-1]]

                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()

                X_scaled = scaler_X.fit_transform(X)
                y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

                X_reshaped, y_reshaped = prepare_data(X_scaled, y_scaled)

                model = create_lstm_model((3, len(columns) - 1))
                model.fit(X_reshaped, y_reshaped, epochs=1, batch_size=32, verbose=0)

                models[pred_type] = model
                scalers[pred_type] = (scaler_X, scaler_y)
                print(f"Debug: Finished training model for {pred_type}")
            else:
                print(f"Debug: Skipping {pred_type} due to missing columns")

print("Debug: Finished processing all files")
print(f"Debug: models keys: {list(models.keys())}")
print(f"Debug: scalers keys: {list(scalers.keys())}")

# User input and prediction for three types
total_duration = 0
for pred_type in ["supplier", "port", "customer"]:
    if pred_type in models and pred_type in scalers:
        print(f"Predicting duration for {pred_type}")
        start_lat = float(input("Enter start latitude: "))
        start_lon = float(input("Enter start longitude: "))
        end_lat = float(input("Enter end latitude: "))
        end_lon = float(input("Enter end longitude: "))
        date_str = input("Enter date (YYYY-MM-DD): ")

        model = models[pred_type]
        scaler_X, scaler_y = scalers[pred_type]

        predicted_duration = predict_duration(model, scaler_X, scaler_y, start_lat, start_lon, end_lat, end_lon,
                                              date_str)
        total_duration += predicted_duration
        print(f"Predicted duration for {pred_type}: {predicted_duration:.2f} hours")
    else:
        print(f"Model for {pred_type} not found. Skipping...")

print(f"Total Trip Duration (h): {total_duration:.2f}")
