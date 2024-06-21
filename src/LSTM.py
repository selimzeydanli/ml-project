import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

start_time = time.time()

# Function to create and train LSTM model
def create_lstm_model(X_train, y_train, X_test, y_test):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
    return model

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return r2, mae

# Function to make prediction
def make_prediction(model, scaler_X, scaler_y, distance):
    scaled_distance = scaler_X.transform([[distance]])
    scaled_prediction = model.predict(scaled_distance.reshape(1, 1, 1))
    prediction = scaler_y.inverse_transform(scaled_prediction)
    return prediction[0][0]

# Directory containing JSON files
data_dir = r"C:\Users\Selim\Desktop\ml-project\data\TransactionDatabases_lstm"

# Lists to store data
distances_supplier = []
durations_supplier = []
distances_port = []
durations_port = []
distances_customer = []
durations_customer = []

# Iterate through JSON files
for filename in os.listdir(data_dir):
    if filename.endswith('.json'):
        with open(os.path.join(data_dir, filename), 'r') as file:
            data_list = json.load(file)
            for data in data_list:  # Iterate through each dictionary in the list
                distances_supplier.append(data['Distance_To_Supplier(km)'])
                durations_supplier.append(data['Duration_To_Supplier(h)'])
                distances_port.append(data['Distance_To_Port(km)'])
                durations_port.append(data['Duration_To_Port(h)'])
                distances_customer.append(data['Distance_To_Customer(km)'])
                durations_customer.append(data['Duration_To_Customer(h)'])

# Create dataframes
df_supplier = pd.DataFrame({'Distance': distances_supplier, 'Duration': durations_supplier})
df_port = pd.DataFrame({'Distance': distances_port, 'Duration': durations_port})
df_customer = pd.DataFrame({'Distance': distances_customer, 'Duration': durations_customer})

# Function to prepare data, train model, and make predictions
def process_scenario(df, scenario_name):
    X = df['Distance'].values.reshape(-1, 1)
    y = df['Duration'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train).reshape(-1, 1, 1)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test).reshape(-1, 1, 1)
    y_test_scaled = scaler_y.transform(y_test)

    model = create_lstm_model(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
    r2, mae = evaluate_model(model, X_test_scaled, y_test_scaled)

    print(f"\n{scenario_name} Scenario:")
    print(f"R-squared: {r2:.4f}")
    print(f"MAE: {mae:.4f}")

    distance = float(input(f"Enter Distance_To_{scenario_name}(km): "))
    prediction = make_prediction(model, scaler_X, scaler_y, distance)
    print(f"Predicted Duration_To_{scenario_name}(h): {prediction:.2f}")

    return model, scaler_X, scaler_y

# Process each scenario
supplier_model, supplier_scaler_X, supplier_scaler_y = process_scenario(df_supplier, "Supplier")
port_model, port_scaler_X, port_scaler_y = process_scenario(df_port, "Port")
customer_model, customer_scaler_X, customer_scaler_y = process_scenario(df_customer, "Customer")

end_time = time.time()
execution_time = end_time - start_time
print(f"\nExecution time: {execution_time:.2f} seconds")