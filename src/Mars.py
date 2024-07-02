import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

print("Current working directory:", os.getcwd())
mars_dir = r"C:\Users\Selim\Desktop\Mars"
file_path = os.path.join(mars_dir, "Numbers_json.json")
print("Attempting to open:", file_path)

try:
    with open(file_path, 'r') as file:
        data = json.load(file)
    print("File successfully opened and loaded.")
except Exception as e:
    print(f"An error occurred while reading the file: {str(e)}")
    exit()

# Convert to DataFrame
df = pd.DataFrame(data)

# Function to check if a string can be converted to a date
def is_date(string):
    try:
        pd.to_datetime(string)
        return True
    except ValueError:
        return False

# Remove rows where 'Date' is not a valid date string
df = df[df['Date'].apply(is_date)]

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

print("\nAfter cleaning:")
print(df.head())
print("\nData Info after cleaning:")
print(df.info())

# Convert datetime to Unix timestamp
X = (df['Date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
y = df['Numbers'].values

print("\nFirst few X values (Unix timestamps):")
print(X.head())

# Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X.values.reshape(-1, 1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Split the data into train, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=42)  # 0.1111 of 90% is 10% of total

# Reshape input for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, 1))
X_val = np.reshape(X_val, (X_val.shape[0], 1, 1))
X_test = np.reshape(X_test, (X_test.shape[0], 1, 1))

# Build the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, 1), return_sequences=True),
    LSTM(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}")

# Generate dates for prediction
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 6, 30)
dates = pd.date_range(start=start_date, end=end_date)

# Prepare input for prediction
X_pred = (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
X_pred_scaled = scaler_X.transform(X_pred.values.reshape(-1, 1))
X_pred_reshaped = np.reshape(X_pred_scaled, (X_pred_scaled.shape[0], 1, 1))

# Make predictions
y_pred_scaled = model.predict(X_pred_reshaped)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Prepare results
results = []
for date, number in zip(dates, y_pred.flatten()):
    results.append({
        "Date": date.strftime("%Y-%m-%d"),
        "Number": float(number)
    })

# Save predictions to JSON file
output_path = os.path.join(mars_dir, "Forecasts.json")
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Forecasts have been saved to {output_path}")