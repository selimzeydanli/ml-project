import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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

# Function to calculate distance based on latitude and longitude
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

# Function to preprocess data and train the model
def train_model(required_columns, target_column):
    # Check if required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in the DataFrame: {missing_columns}")

    if target_column not in df.columns:
        raise KeyError(f"The target column '{target_column}' is not in the DataFrame. Please check the JSON files for the correct column name.")

    # Calculate the distance for Truck_Location to Supplier_Location
    df['Distance_To_Supplier(km)'] = df.apply(lambda row: calculate_distance(row['Truck_Latitude'], row['Truck_Longitude'], row['Supplier_Latitude'], row['Supplier_Longitude']), axis=1)

    # Calculate the distance for Truck_Location to Customer_Location
    df['Distance_To_Customer(km)'] = df.apply(lambda row: calculate_distance(row['Truck_Latitude'], row['Truck_Longitude'], row['Customer_Latitude'], row['Customer_Longitude']), axis=1)

    # Calculate the distance for Tarragona to Customer_Location (assuming Tarragona_Latitude and Tarragona_Longitude exist in the DataFrame)
    df['Distance_To_Port(km)'] = df.apply(lambda row: calculate_distance(row['Tarragona_Latitude'], row['Tarragona_Longitude'], row['Customer_Latitude'], row['Customer_Longitude']), axis=1)

    # Select relevant variables
    X = df[['Distance_To_Supplier(km)', 'Distance_To_Customer(km)', 'Distance_To_Port(km)']].values
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

# Assuming customer_lat and customer_lon are defined somewhere in your code
customer_lat = float(input("Enter the customer latitude: "))
customer_lon = float(input("Enter the customer longitude: "))

# Train the model for 'Truck_Latitude', 'Truck_Longitude', 'Supplier_Latitude', 'Supplier_Longitude' and 'Duration_To_Supplier(h)'
model_supplier, scaler_X_supplier, scaler_y_supplier = train_model(['Truck_Latitude', 'Truck_Longitude', 'Supplier_Latitude', 'Supplier_Longitude'], 'Duration_To_Supplier(h)')

# Train the model for 'Truck_Latitude', 'Truck_Longitude', 'Customer_Latitude', 'Customer_Longitude' and 'Duration_To_Customer(h)'
model_customer, scaler_X_customer, scaler_y_customer = train_model(['Truck_Latitude', 'Truck_Longitude', 'Customer_Latitude', 'Customer_Longitude'], 'Duration_To_Customer(h)')

# User input for departure and arrival latitudes and longitudes
dep_lat = float(input("Enter the truck departure latitude: "))
dep_lon = float(input("Enter the truck departure longitude: "))
supplier_lat = float(input("Enter the supplier latitude: "))
supplier_lon = float(input("Enter the supplier longitude: "))

# Calculate the distance to supplier
distance_to_supplier = calculate_distance(dep_lat, dep_lon, supplier_lat, supplier_lon)

# Calculate the distance to customer
distance_to_customer = calculate_distance(dep_lat, dep_lon, customer_lat, customer_lon)

# Assuming Tarragona coordinates are already known or provided
tarragona_lat = 41.1189
tarragona_lon = 1.2445

# Calculate the distance to Tarragona port
distance_to_port = calculate_distance(tarragona_lat, tarragona_lon, customer_lat, customer_lon)

# Create a new input array based on user input for supplier
user_input_features_supplier = np.array([[distance_to_supplier, distance_to_customer, distance_to_port]])

# Transform user_input_features_supplier using the fitted scaler_X_supplier
user_input_features_scaled_supplier = scaler_X_supplier.transform(user_input_features_supplier)

# Reshape user_input_features_scaled_supplier for LSTM input
user_input_features_scaled_supplier = user_input_features_scaled_supplier.reshape((1, 1, user_input_features_supplier.shape[1]))

# Predict duration to supplier
predicted_duration_scaled_supplier = model_supplier.predict(user_input_features_scaled_supplier)
predicted_duration_supplier = scaler_y_supplier.inverse_transform(predicted_duration_scaled_supplier.reshape(-1, 1))

# Calculate predicted speed to supplier
predicted_speed_supplier = distance_to_supplier / predicted_duration_supplier[0][0]

# Calculate the distance to customer (if needed for further calculations)
# distance_to_customer = calculate_distance(dep_lat, dep_lon, customer_lat, customer_lon)

# Create a new input array based on user input for customer (if needed)
# user_input_features_customer = np.array([[distance_to_customer]])

# Transform user_input_features_customer using the fitted scaler_X_customer (if needed)
# user_input_features_scaled_customer = scaler_X_customer.transform(user_input_features_customer)

# Reshape user_input_features_scaled_customer for LSTM input (if needed)
# user_input_features_scaled_customer = user_input_features_scaled_customer.reshape((1, 1, user_input_features_customer.shape[1]))

# Predict duration to customer (if needed)
# predicted_duration_scaled_customer = model_customer.predict(user_input_features_scaled_customer)
# predicted_duration_customer = scaler_y_customer.inverse_transform(predicted_duration_scaled_customer.reshape(-1, 1))

# Calculate predicted speed to customer (if needed)
# predicted_speed_customer = distance_to_customer / predicted_duration_customer[0][0]

# Debugging: Print scaled and unscaled user input features for supplier
print("User input features for Supplier (scaled): ", user_input_features_scaled_supplier)
print("User input features for Supplier (unscaled): ", user_input_features_supplier)

# Assuming Tarragona coordinates are already known or provided
tarragona_lat = 41.1189
tarragona_lon = 1.2445

# Calculate the distance to Tarragona port
distance_to_port = calculate_distance(tarragona_lat, tarragona_lon, customer_lat, customer_lon)

# Create a new input array based on user input for port
user_input_features_port = np.array([[distance_to_port]])

# Transform user_input_features_port using the fitted scaler_X_customer
user_input_features_scaled_port = scaler_X_customer.transform(user_input_features_port)

# Reshape user_input_features_scaled_port for LSTM input
user_input_features_scaled_port = user_input_features_scaled_port.reshape((1, 1, user_input_features_port.shape[1]))

# Predict duration to port
predicted_duration_scaled_port = model_customer.predict(user_input_features_scaled_port)
predicted_duration_port = scaler_y_customer.inverse_transform(predicted_duration_scaled_port.reshape(-1, 1))

# Calculate predicted speed to port
predicted_speed_port = distance_to_port / predicted_duration_port[0][0]

# Debugging: Print scaled and unscaled user input features for port
print("User input features for Port (scaled): ", user_input_features_scaled_port)
print("User input features for Port (unscaled): ", user_input_features_port)

# Print Distance To Port (km), Predicted Speed To Port (km/h), Predicted Duration To Port (h)
print(f'\nDistance To Port (km): {distance_to_port:.2f}')
print(f'Predicted Speed To Port (km/h): {predicted_speed_port:.2f}')
print(f'Predicted Duration To Port (h): {predicted_duration_port[0][0]:.2f} hours')
