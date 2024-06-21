import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score, mean_absolute_error

# Set the directory path
directory = r"C:\Users\Selim\Desktop\ml-project\data\TransactionDatabases_lstm"

# Define the three prediction scenarios
scenarios = [
    {
        "name": "Supplier",
        "features": ["Truck_Latitude", "Truck_Longitude", "Supplier_Latitude", "Supplier_Longitude",
                     "Speed_To_Supplier(km/h)"],
        "target": "Speed_To_Supplier(km/h)"
    },
    {
        "name": "Port",
        "features": ["Supplier_Latitude", "Supplier_Longitude", "Port_Latitude", "Port_Longitude",
                     "Speed_To_Port(km/h)"],
        "target": "Speed_To_Port(km/h)"
    },
    {
        "name": "Customer",
        "features": ["Tarragona_Latitude", "Tarragona_Longitude", "Customer_Latitude", "Customer_Longitude",
                     "Speed_To_Customer(km/h)"],
        "target": "Speed_To_Customer(km/h)"
    }
]

def create_and_train_model(X, y):
    # Normalize the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape input data for LSTM (samples, time steps, features)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

    # Define the LSTM model
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(1, X_reshaped.shape[2])),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Train the model
    history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test, verbose=0)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate R2 and MAE
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    return model, X_test, y_test, loss, predictions, r2, mae

# Initialize dictionaries to store data for each scenario
scenario_data = {scenario["name"]: {"features": [], "target": []} for scenario in scenarios}

# Iterate through files in the directory
json_files_found = False
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        json_files_found = True
        file_path = os.path.join(directory, filename)
        print(f"Processing file: {filename}")

        # Read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Extract data for each scenario
        for scenario in scenarios:
            if all(col in df.columns for col in scenario["features"]):
                scenario_data[scenario["name"]]["features"].append(df[scenario["features"]].values)
                scenario_data[scenario["name"]]["target"].append(df[scenario["target"]].values)
            else:
                print(f"Warning: Missing columns for {scenario['name']} scenario in {filename}")

if not json_files_found:
    print("No JSON files found in the specified directory.")
    exit()

# Train models and make predictions for each scenario
for scenario in scenarios:
    name = scenario["name"]
    if scenario_data[name]["features"]:
        X = np.concatenate(scenario_data[name]["features"])
        y = np.concatenate(scenario_data[name]["target"])

        print(f"\nScenario: {name}")
        print(f"Total samples: {len(X)}")

        model, X_test, y_test, loss, predictions, r2, mae = create_and_train_model(X, y)

        print(f"Test Loss: {loss}")
        print(f"R-squared (R2): {r2}")
        print(f"Mean Absolute Error (MAE): {mae}")

        # Print some sample predictions
        print("Sample predictions:")
        for i in range(5):
            print(f"Actual: {y_test[i]}, Predicted: {predictions[i][0]}")
    else:
        print(f"\nNo data available for {name} scenario")