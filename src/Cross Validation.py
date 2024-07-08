import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta

start_time = time.time()

# Function to create and train LSTM model
def create_lstm_model(X_train, y_train, X_val, y_val):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0, validation_data=(X_val, y_val))
    return model

# Function to make prediction
def make_prediction(model, scaler_X, scaler_y, distance):
    scaled_distance = scaler_X.transform([[distance]])
    scaled_prediction = model.predict(scaled_distance.reshape(1, 1, 1))
    prediction = scaler_y.inverse_transform(scaled_prediction)
    return prediction[0][0]

# Function to check fixed values
def check_fixed_values(df, scenario_name):
    unique_distances = df['Distance'].nunique()
    unique_durations = df['Duration'].nunique()
    print(f"\n{scenario_name} Scenario:")
    print(f"Unique distance values: {unique_distances}")
    print(f"Unique duration values: {unique_durations}")
    return unique_distances == 1

# Function to process fixed scenario
def process_fixed_scenario(df, scenario_name):
    mean_duration = df['Duration'].mean()
    std_duration = df['Duration'].std()

    print(f"\n{scenario_name} Scenario (Fixed Distance):")
    print(f"Standard Deviation of Duration: {std_duration:.2f}")

    distance = df['Distance'].iloc[0]  # Get the fixed distance value
    print(f"\nDistance_To_{scenario_name}(km)     : {distance:.2f}\n")

    speed = distance / mean_duration
    print(f"Predicted Duration To {scenario_name}(h)        : {mean_duration:.2f}\n")
    print(f"Predicted Speed To {scenario_name}(km/h)        : {speed:.2f}\n")
    return float(mean_duration)

# Function to analyze data
def analyze_data(df, scenario_name):
    print(f"\n{scenario_name} Data Analysis:")
    print(df.describe())

    plt.figure(figsize=(12, 4))

    plt.subplot(121)
    sns.histplot(df['Distance'], kde=True)
    plt.title(f'{scenario_name} Distance Distribution')

    plt.subplot(122)
    sns.histplot(df['Duration'], kde=True)
    plt.title(f'{scenario_name} Duration Distribution')

    plt.tight_layout()
    plt.show()

    correlation = df['Distance'].corr(df['Duration'])
    print(f"Correlation between Distance and Duration: {correlation:.4f}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Distance', y='Duration', data=df)
    plt.title(f'{scenario_name} Distance vs Duration')
    plt.show()

# Function to process variable scenario with cross-validation
def process_variable_scenario(df, scenario_name):
    analyze_data(df, scenario_name)

    X = df['Distance'].values.reshape(-1, 1)
    y = df['Duration'].values.reshape(-1, 1)

    # Initialize K-Fold
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Lists to store results
    r2_scores = []
    mae_scores = []

    # Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(k_fold.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train_scaled = scaler_X.fit_transform(X_train).reshape(-1, 1, 1)
        y_train_scaled = scaler_y.fit_transform(y_train)
        X_val_scaled = scaler_X.transform(X_val).reshape(-1, 1, 1)
        y_val_scaled = scaler_y.transform(y_val)

        model = create_lstm_model(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)

        # Predict on validation data
        y_pred_scaled = model.predict(X_val_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        # Calculate metrics
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)

        r2_scores.append(r2)
        mae_scores.append(mae)

        print(f"\n{scenario_name} Scenario - Fold {fold}:")
        print(f"R-squared: {r2:.4f}")
        print(f"MAE: {mae:.4f}")

    # Calculate and print average scores
    print(f"\n{scenario_name} Scenario - Average Scores:")
    print(f"Average R-squared: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})")
    print(f"Average MAE: {np.mean(mae_scores):.4f} (+/- {np.std(mae_scores):.4f})")

    # Train final model on all data for predictions
    scaler_X_final = MinMaxScaler()
    scaler_y_final = MinMaxScaler()
    X_scaled_final = scaler_X_final.fit_transform(X).reshape(-1, 1, 1)
    y_scaled_final = scaler_y_final.fit_transform(y)
    final_model = create_lstm_model(X_scaled_final, y_scaled_final, X_scaled_final, y_scaled_final)

    print()
    distance = float(input(f"Enter Distance_To_{scenario_name}(km): "))
    print()
    prediction = make_prediction(final_model, scaler_X_final, scaler_y_final, distance)
    print()
    print(f"Predicted Duration_To_{scenario_name}(h)            : {prediction:.2f}")
    print()
    speed = distance / prediction
    print(f"Predicted Speed To {scenario_name}(km/h)            : {speed:.2f}")

    return final_model, scaler_X_final, scaler_y_final, float(prediction), distance

# Function to predict Duration_Loading(h) in supplier scenario
def predict_loading_duration():
    loading_times = np.random.uniform(4.5, 13, 1000)  # Simulating loading times between 4.5 and 13 hours
    predicted_loading_time = np.mean(loading_times)
    return predicted_loading_time

# Function to predict Duration_Unloading(h) in customer scenario
def predict_unloading_duration():
    unloading_times = np.random.uniform(3, 5, 1000)  # Simulating unloading times between 3 and 5 hours
    predicted_unloading_time = np.mean(unloading_times)
    return predicted_unloading_time

# Directory containing JSON files
data_dir = r"C:\Users\Selim\Desktop\ml-project\data\TransactionDatabases_lstm"

# Lists to store data
distances_supplier = []
durations_supplier = []
distances_port = []
durations_port = []
distances_customer = []
durations_customer = []
loading_times_customers = []
unloading_times_customers = []

# Iterate through JSON files
for filename in os.listdir(data_dir):
    if filename.endswith('.json'):
        with open(os.path.join(data_dir, filename), 'r') as file:
            data_list = json.load(file)
            for data in data_list:
                distances_supplier.append(data['Distance_To_Supplier(km)'])
                durations_supplier.append(data['Duration_To_Supplier(h)'])
                distances_port.append(data['Distance_To_Port(km)'])
                durations_port.append(data['Duration_To_Port(h)'])
                distances_customer.append(data['Distance_To_Customer(km)'])
                durations_customer.append(data['Duration_To_Customer(h)'])
                loading_times_customers.append(data['Duration_Loading(h)'])
                unloading_times_customers.append(data['Duration_Unloading(h)'])

# Create dataframes
df_supplier = pd.DataFrame({'Distance': distances_supplier, 'Duration': durations_supplier})
df_port = pd.DataFrame({'Distance': distances_port, 'Duration': durations_port})
df_customer = pd.DataFrame({'Distance': distances_customer, 'Duration': durations_customer})
df_loading_times = pd.DataFrame({'loading_times_customers': loading_times_customers})
mean_loading_duration = df_loading_times['loading_times_customers'].mean()
std_loading_duration = df_loading_times['loading_times_customers'].std()
predicted_loading_time = mean_loading_duration + std_loading_duration
df_unloading_times = pd.DataFrame({'unloading_times_customers': unloading_times_customers})
mean_unloading_duration = df_unloading_times['unloading_times_customers'].mean()
std_unloading_duration = df_unloading_times['unloading_times_customers'].std()
predicted_unloading_time = mean_unloading_duration + std_unloading_duration

supplier_duration = None
port_duration = None
loading_duration = None
customer_duration = None
unloading_duration = None

# Process each scenario
for scenario_name, df in [("Supplier", df_supplier), ("Port", df_port), ("Customer", df_customer)]:
    is_fixed = check_fixed_values(df, scenario_name)
    if is_fixed:
        if scenario_name == "Supplier":
            supplier_duration = process_fixed_scenario(df, scenario_name)
            loading_duration = predict_loading_duration()
        elif scenario_name == "Port":
            port_duration = process_fixed_scenario(df, scenario_name)
        elif scenario_name == "Customer":
            customer_duration = process_fixed_scenario(df, scenario_name)
            unloading_duration = predict_unloading_duration()
    else:
        if scenario_name == "Supplier":
            _, _, _, supplier_duration, just_supplier_distance = process_variable_scenario(df, scenario_name)
            loading_duration = predict_loading_duration()
        elif scenario_name == "Port":
            _, _, _, port_duration, just_port_distance = process_variable_scenario(df, scenario_name)
        elif scenario_name == "Customer":
            _, _, _, customer_duration, just_customer_distance = process_variable_scenario(df, scenario_name)
            unloading_duration = predict_unloading_duration()

while True:
    try:
        truck_needed_str = input("When is the truck needed (DD/MM/YYYY HH:MM:SS): ")
        truck_needed = datetime.strptime(truck_needed_str, "%d/%m/%Y %H:%M:%S")
        break
    except ValueError:
        print("Invalid date format. Please use DD/MM/YYYY HH:MM:SS.")

# Calculate when the truck has to leave
if supplier_duration is not None and port_duration is not None and customer_duration is not None:
    truck_leave = truck_needed - timedelta(hours=supplier_duration)
    take_off_time = truck_needed + timedelta(hours=predicted_loading_time)
    arrival_at_port = take_off_time + timedelta(hours=port_duration)
    def next_sunday(date):
        days_until_sunday = (6 - date.weekday() + 7) % 7
        next_sunday_date = date + timedelta(days=days_until_sunday)
        return next_sunday_date

    def set_time(date):
        return date.replace(hour=23, minute=30, second=0)

    port_arrival = datetime.now()

    next_sunday_date = next_sunday(arrival_at_port)
    ferry_take_off = set_time(next_sunday_date)

    Arrival_At_Tarragona = ferry_take_off + timedelta(hours=72)
    Arrival_At_Customer = Arrival_At_Tarragona + timedelta(hours=customer_duration)
    Unloading_Finishes = Arrival_At_Customer + timedelta(hours=predicted_unloading_time)

    print()
    print(f"Within a Radius of                          : {just_supplier_distance}")
    print()
    print(f"For a Response Time of (h)                  : {supplier_duration:.2f}")
    print()
    print(f"To start loading                            : {truck_needed.strftime('%d/%m/%Y %H:%M:%S')}")
    print()
    print(f"Truck to start for customer                 : {truck_leave.strftime('%d/%m/%Y %H:%M:%S')}")
    print()
    print(f"Predicted Duration of Loading & Waiting(h)  : {predicted_loading_time:.2f}")
    print()
    print(f"Loading finish / truck take-off             : {take_off_time.strftime('%d/%m/%Y %H:%M:%S')}")
    print()
    print(f"Port Distance (km)                          : {just_port_distance}")
    print()
    print(f"Predicted Duration To Port (h)              : {port_duration:.2f}")
    print()
    print(f"Arrival Alsancak Port                       : {arrival_at_port.strftime('%d/%m/%Y %H:%M:%S')}")
    print()
    print(f"Ferry-Take-Off                              : {ferry_take_off.strftime('%d/%m/%Y %H:%M:%S')}")
    print()
    print(f"Arrival Tarragona-Port                      : {Arrival_At_Tarragona.strftime('%d/%m/%Y %H:%M:%S')}")
    print()
    print(f"Predicted Time To Customer (h)              : {customer_duration:.2f}")
    print()
    print(f"Arrival Customer                            : {Arrival_At_Customer.strftime('%d/%m/%Y %H:%M:%S')}")
    print()
    print(f"Predicted Duration of Unloading &Waiting (h): {predicted_unloading_time:.2f}")
    print()
    print(f"Unloading Finish / Truck Free               : {Unloading_Finishes.strftime('%d/%m/%Y %H:%M:%S')}")
else:
    print("Error: Some durations were not calculated.")

# Add these lines at the end of the script
end_time = time.time()
execution_time = end_time - start_time
print(f"\nExecution time: {execution_time:.2f} seconds")

# New function to count and print the number of JSON files
def print_json_file_count(directory):
    json_count = sum(1 for filename in os.listdir(directory) if filename.endswith('.json'))
    print(f"Number of JSON files processed: {json_count}")

# Call the new function
print_json_file_count(data_dir)