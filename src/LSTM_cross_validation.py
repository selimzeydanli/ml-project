import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta

start_time = time.time()


# Function to create LSTM model
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


# Function to make prediction
def make_prediction(model, scaler_X, scaler_y, distance):
    scaled_distance = scaler_X.transform([[distance]])
    scaled_prediction = model.predict(scaled_distance.reshape(1, 1, 1))
    prediction = scaler_y.inverse_transform(scaled_prediction)
    return prediction[0][0]


# Function to process scenario with cross-validation
def process_scenario_with_cv(df, scenario_name):
    print(f"\n{scenario_name} Scenario Analysis:")

    X = df['Distance'].values.reshape(-1, 1)
    y = df['Duration'].values.reshape(-1, 1)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    tscv = TimeSeriesSplit(n_splits=5)

    r2_scores = []
    mae_scores = []

    for train_index, test_index in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y_scaled[train_index], y_scaled[test_index]

        X_train = X_train.reshape(-1, 1, 1)
        X_test = X_test.reshape(-1, 1, 1)

        model = create_lstm_model((1, 1))
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

        y_pred_scaled = model.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_true = scaler_y.inverse_transform(y_test)

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        r2_scores.append(r2)
        mae_scores.append(mae)

    print(f"Average R-squared: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})")
    print(f"Average MAE: {np.mean(mae_scores):.4f} (+/- {np.std(mae_scores):.4f})")

    # Train final model on all data
    X_scaled = X_scaled.reshape(-1, 1, 1)
    final_model = create_lstm_model((1, 1))
    final_model.fit(X_scaled, y_scaled, epochs=50, batch_size=32, verbose=0)

    # Make prediction
    print()
    distance = float(input(f"Enter Distance_To_{scenario_name}(km): "))
    print()
    prediction = make_prediction(final_model, scaler_X, scaler_y, distance)
    print(f"Predicted Duration_To_{scenario_name}(h): {prediction:.2f}")
    print(f"Predicted Speed To {scenario_name}(km/h): {distance / prediction:.2f}")

    return final_model, scaler_X, scaler_y, prediction, distance


# Function to predict loading/unloading duration
def predict_duration(min_time, max_time):
    times = np.random.uniform(min_time, max_time, 1000)
    return np.mean(times)


# Directory containing JSON files
data_dir = r"C:\Users\Selim\Desktop\ml-project\data\TransactionDatabases_lstm"

# Load data from JSON files
data = {
    'Supplier': {'Distance': [], 'Duration': []},
    'Port': {'Distance': [], 'Duration': []},
    'Customer': {'Distance': [], 'Duration': []},
    'Loading': [],
    'Unloading': []
}

for filename in os.listdir(data_dir):
    if filename.endswith('.json'):
        with open(os.path.join(data_dir, filename), 'r') as file:
            data_list = json.load(file)
            for item in data_list:
                data['Supplier']['Distance'].append(item['Distance_To_Supplier(km)'])
                data['Supplier']['Duration'].append(item['Duration_To_Supplier(h)'])
                data['Port']['Distance'].append(item['Distance_To_Port(km)'])
                data['Port']['Duration'].append(item['Duration_To_Port(h)'])
                data['Customer']['Distance'].append(item['Distance_To_Customer(km)'])
                data['Customer']['Duration'].append(item['Duration_To_Customer(h)'])
                data['Loading'].append(item['Duration_Loading(h)'])
                data['Unloading'].append(item['Duration_Unloading(h)'])

# Create dataframes
dfs = {scenario: pd.DataFrame(data[scenario]) for scenario in ['Supplier', 'Port', 'Customer']}
loading_df = pd.DataFrame({'Duration': data['Loading']})
unloading_df = pd.DataFrame({'Duration': data['Unloading']})

# Process each scenario
results = {}
for scenario in ['Supplier', 'Port', 'Customer']:
    results[scenario] = process_scenario_with_cv(dfs[scenario], scenario)

# Predict loading and unloading times
loading_duration = predict_duration(4.5, 13)
unloading_duration = predict_duration(3, 5)

# Get user input for truck needed time
while True:
    try:
        truck_needed_str = input("When is the truck needed (DD/MM/YYYY HH:MM:SS): ")
        truck_needed = datetime.strptime(truck_needed_str, "%d/%m/%Y %H:%M:%S")
        break
    except ValueError:
        print("Invalid date format. Please use DD/MM/YYYY HH:MM:SS.")

# Calculate timeline
timeline = {
    'Truck Leave': truck_needed - timedelta(hours=results['Supplier'][3]),
    'Loading Start': truck_needed,
    'Loading Finish': truck_needed + timedelta(hours=loading_duration),
    'Arrival at Port': truck_needed + timedelta(hours=loading_duration + results['Port'][3]),
    'Ferry Take-Off': None,  # Will be calculated
    'Arrival at Tarragona': None,  # Will be calculated
    'Arrival at Customer': None,  # Will be calculated
    'Unloading Finish': None  # Will be calculated
}

# Calculate ferry take-off (next Sunday at 23:30)
next_sunday = timeline['Arrival at Port'] + timedelta(days=(6 - timeline['Arrival at Port'].weekday() + 7) % 7)
timeline['Ferry Take-Off'] = next_sunday.replace(hour=23, minute=30, second=0)

# Complete the timeline
timeline['Arrival at Tarragona'] = timeline['Ferry Take-Off'] + timedelta(hours=72)
timeline['Arrival at Customer'] = timeline['Arrival at Tarragona'] + timedelta(hours=results['Customer'][3])
timeline['Unloading Finish'] = timeline['Arrival at Customer'] + timedelta(hours=unloading_duration)

# Print results
print("\nTimeline:")
for event, time in timeline.items():
    print(f"{event}: {time.strftime('%d/%m/%Y %H:%M:%S') if time else 'N/A'}")

print(f"\nSupplier Distance: {results['Supplier'][4]:.2f} km")
print(f"Port Distance: {results['Port'][4]:.2f} km")
print(f"Customer Distance: {results['Customer'][4]:.2f} km")
print(f"Loading Duration: {loading_duration:.2f} hours")
print(f"Unloading Duration: {unloading_duration:.2f} hours")

# Print execution time and file count
end_time = time.time()
print(f"\nExecution time: {end_time - start_time:.2f} seconds")
print(f"Number of JSON files processed: {sum(1 for filename in os.listdir(data_dir) if filename.endswith('.json'))}")