import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime
from geopy.distance import geodesic

# Define the path to your JSON files on your PC
path_to_json = r'C:\Users\Selim\Desktop\ml-project\data\TransactionDatabases_lstm'

# Load all JSON files in the directory
data = []
for file_name in os.listdir(path_to_json):
    if file_name.endswith('.json'):
        file_path = os.path.join(path_to_json, file_name)
        try:
            with open(file_path, 'r') as file:
                content = file.read().strip()
                if content:  # Check if the file is not empty
                    data.append(json.loads(content))
                else:
                    print(f"Warning: {file_name} is empty and will be skipped.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {file_name}: {e}")
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")

# Assuming each JSON file contains a list of dictionaries
data = [item for sublist in data for item in sublist]

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Check if DataFrame is empty
if df.empty:
    raise ValueError("The DataFrame is empty. Ensure your JSON files contain data.")

# Print the DataFrame structure and columns
print("DataFrame structure:")
print(df.head())
print("\nDataFrame columns:")
print(df.columns)

# Display column names for debugging
print("Columns in the DataFrame:")
for col in df.columns:
    print(f"'{col}'")

# Function to preprocess data and train the model
def train_model(required_columns, target_column):
    # Check if required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in the DataFrame: {missing_columns}")

    if target_column not in df.columns:
        raise KeyError(f"The target column '{target_column}' is not in the DataFrame. Please check the JSON files for the correct column name.")

    # Select relevant variables
    X = df[required_columns].values
    y = df[target_column].values

    # Reshape X for LSTM [samples, time steps, features]
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit scaler_X based on X_train
    scaler_X.fit(X_train.reshape(-1, X_train.shape[-1]))

    # Transform X_train and X_test with the fitted scaler
    X_train = scaler_X.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train the model with 1 epoch
    model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate the model with MSE
    loss = model.evaluate(X_test, y_test, verbose=1)
    print(f'Model Loss (MSE): {loss}')

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))

    # Calculate MAE
    y_test_reshaped = y_test.reshape(-1, 1)  # Reshape y_test to 2D
    mae = mean_absolute_error(scaler_y.inverse_transform(y_test_reshaped), y_pred)
    print(f'Mean Absolute Error (MAE): {mae}')

    # Calculate R-squared
    r2 = r2_score(scaler_y.inverse_transform(y_test_reshaped), y_pred)
    print(f'R-squared (R2 Score): {r2}')

    return model, scaler_X, scaler_y

# Train the model for 'Distance_To_Supplier(km)' and 'Duration_To_Supplier(h)'
model_supplier, scaler_X_supplier, scaler_y_supplier = train_model(['Distance_To_Supplier(km)'], 'Duration_To_Supplier(h)')

# Train the model for 'Distance_To_Port(km)' and 'Duration_To_Port(h)'
model_port, scaler_X_port, scaler_y_port = train_model(['Distance_To_Port(km)'], 'Duration_To_Port(h)')

# Function to calculate distance based on latitude and longitude
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

# User input for departure and arrival latitudes and longitudes
dep_lat = float(input("Enter the departure latitude: "))
dep_lon = float(input("Enter the departure longitude: "))
arr_lat = float(input("Enter the arrival latitude: "))
arr_lon = float(input("Enter the arrival longitude: "))
month = int(input("Enter the month (1-12): "))
day = int(input("Enter the day (1-31): "))

# Calculate the distance to supplier
distance_to_supplier = calculate_distance(dep_lat, dep_lon, arr_lat, arr_lon)

# Create a new input array based on user input
user_input_date = datetime(2024, month, day)
user_input_features_supplier = np.array([[distance_to_supplier]])

# Transform user_input_features using the fitted scaler_X
user_input_features_scaled_supplier = scaler_X_supplier.transform(user_input_features_supplier)

# Reshape user_input_features_scaled for LSTM input
user_input_features_scaled_supplier = user_input_features_scaled_supplier.reshape((1, 1, 1))

# Predict duration to supplier
predicted_duration_scaled_supplier = model_supplier.predict(user_input_features_scaled_supplier)
predicted_duration_supplier = scaler_y_supplier.inverse_transform(predicted_duration_scaled_supplier.reshape(-1, 1))

# Calculate predicted speed to supplier
predicted_speed_supplier = distance_to_supplier / predicted_duration_supplier[0][0]

# Debugging: Print scaled and unscaled user input features
print("User input features (scaled): ", user_input_features_scaled_supplier)
print("User input features (unscaled): ", user_input_features_supplier)

print(f'\nDistance To Supplier (km): {distance_to_supplier:.2f}')
print(f'Predicted Speed To Supplier (km/h): {predicted_speed_supplier:.2f}')
print(f'Predicted Duration To Supplier (h) for {month}/{day}: {predicted_duration_supplier[0][0]:.2f} hours')

# Fixed port location
port_lat = 38.43
port_lon = 27.14

# Calculate the distance to port
distance_to_port = calculate_distance(port_lat, port_lon, arr_lat, arr_lon)

# Create a new input array based on user input
user_input_features_port = np.array([[distance_to_port]])

# Transform user_input_features using the fitted scaler_X
user_input_features_scaled_port = scaler_X_port.transform(user_input_features_port)

# Reshape user_input_features_scaled for LSTM input
user_input_features_scaled_port = user_input_features_scaled_port.reshape((1, 1, 1))

# Predict duration to port
predicted_duration_scaled_port = model_port.predict(user_input_features_scaled_port)
predicted_duration_port = scaler_y_port.inverse_transform(predicted_duration_scaled_port.reshape(-1, 1))

# Calculate predicted speed to port
predicted_speed_port = distance_to_port / predicted_duration_port[0][0]

print(f'\nDistance To Port (km): {distance_to_port:.2f}')
print(f'Predicted Speed To Port (km/h): {predicted_speed_port:.2f}')
print(f'Predicted Duration To Port (h) for {month}/{day}: {predicted_duration_port[0][0]:.2f} hours')
