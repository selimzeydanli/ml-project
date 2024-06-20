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

# Update the column names based on the actual DataFrame columns
required_columns = ['Distance_To_Supplier(km)', 'Speed_To_Supplier(km/h)']

# Check if required columns are present
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise KeyError(f"Missing columns in the DataFrame: {missing_columns}")

# Correctly identify the target variable column
target_column = 'Duration_To_Supplier(h)'
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

X_train = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.2, verbose=1)

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

# Print a few predictions
print("\nPredictions vs Actuals:")
for i in range(min(5, len(X_test))):
    actual = scaler_y.inverse_transform(y_test_reshaped[i].reshape(1, -1))[0][0]
    predicted = y_pred[i][0]
    print(f'Predicted: {predicted:.2f}, Actual: {actual:.2f}')

# User input for month, day, and distance to supplier
month = int(input("Enter the month (1-12): "))
day = int(input("Enter the day (1-31): "))
distance_to_supplier = float(input("Enter the Distance To Supplier (km): "))

# Create a new input array based on user input
user_input_date = datetime(2024, month, day)
day_of_year = user_input_date.timetuple().tm_yday
user_input_features = np.array([[distance_to_supplier, day_of_year]])

# Scale the user input features
user_input_features_scaled = scaler_X.transform(user_input_features).reshape((1, 1, 2))

# Predict duration to supplier
predicted_duration_scaled = model.predict(user_input_features_scaled)
predicted_duration = scaler_y.inverse_transform(predicted_duration_scaled.reshape(-1, 1))

print(f'Predicted Duration To Supplier (h) for {month}/{day}: {predicted_duration[0][0]:.2f} hours')
