import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta

start_time = time.time()


# Function to create and train LSTM model
def create_lstm_model(X_train, y_train, X_test, y_test):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0, validation_split=0.2)
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
    print(f"Mean Duration: {mean_duration:.2f}")
    print(f"Standard Deviation of Duration: {std_duration:.2f}")

    distance = df['Distance'].iloc[0]  # Get the fixed distance value
    print(f"Fixed Distance_To_{scenario_name}(km): {distance:.2f}")
    print(f"Predicted Duration_To_{scenario_name}(h): {mean_duration:.2f} ± {std_duration:.2f}")

    speed = distance / mean_duration
    print(f"Average Speed To {scenario_name}(km/h): {speed:.2f}")

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


# Function to process variable scenario
def process_variable_scenario(df, scenario_name):
    analyze_data(df, scenario_name)

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

    # Predict on test data
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n{scenario_name} Scenario:")
    print(f"R-squared: {r2:.4f}")
    print(f"MAE: {mae:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Duration')
    plt.ylabel('Predicted Duration')
    plt.title(f'{scenario_name} Actual vs Predicted Duration')
    plt.show()

    distance = float(input(f"Enter Distance_To_{scenario_name}(km): "))
    prediction = make_prediction(model, scaler_X, scaler_y, distance)
    print(f"Predicted Duration_To_{scenario_name}(h): {prediction:.2f}")

    speed = distance / prediction
    print(f"Average Speed To {scenario_name}(km/h): {speed:.2f}")

    return model, scaler_X, scaler_y, float(prediction)


# New function to predict Duration_Loading(h) in supplier scenario
def predict_loading_duration():
    loading_times = np.random.uniform(4.5, 13, 1000)  # Simulating loading times between 4.5 and 13 hours
    predicted_loading_time = np.mean(loading_times)
    print(f"\nPredicted Duration_Loading(h) at Supplier: {predicted_loading_time:.2f}")
    return predicted_loading_time


# New function to predict Duration_Unloading(h) in customer scenario
def predict_unloading_duration():
    unloading_times = np.random.uniform(3, 5, 1000)  # Simulating unloading times between 3 and 5 hours
    predicted_unloading_time = np.mean(unloading_times)
    print(f"\nPredicted Duration_Unloading(h) at Customer: {predicted_unloading_time:.2f}")
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

# Create dataframes
df_supplier = pd.DataFrame({'Distance': distances_supplier, 'Duration': durations_supplier})
df_port = pd.DataFrame({'Distance': distances_port, 'Duration': durations_port})
df_customer = pd.DataFrame({'Distance': distances_customer, 'Duration': durations_customer})

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
            _, _, _, supplier_duration = process_variable_scenario(df, scenario_name)
            loading_duration = predict_loading_duration()
        elif scenario_name == "Port":
            _, _, _, port_duration = process_variable_scenario(df, scenario_name)
        elif scenario_name == "Customer":
            _, _, _, customer_duration = process_variable_scenario(df, scenario_name)
            unloading_duration = predict_unloading_duration()

# Get user input for truck needed time
truck_needed_str = input("When is the truck needed (DD/MM/YYYY HH:MM:SS): ")
truck_needed = datetime.strptime(truck_needed_str, "%d/%m/%Y %H:%M:%S")

# Calculate when the truck has to leave
if supplier_duration is not None and port_duration is not None and customer_duration is not None:
    truck_leave = truck_needed - timedelta(hours=supplier_duration)
    print(f"In order to start loading at : {truck_needed.strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Truck needs to leave previous location : {truck_leave.strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Predicted Duration to Supplier (h) : {supplier_duration:.2f}")
    print(f"Predicted Duration of Loading (h) : {loading_duration:.2f}")

    take_off_time = truck_needed + timedelta(hours=loading_duration)
    print(f"Loading finishes and truck takes off : {take_off_time.strftime('%d/%m/%Y %H:%M:%S')}")

    print(f"Predicted Duration To Port (h): {port_duration:.2f}")
    arrival_at_port = take_off_time + timedelta(hours=port_duration)
    print(f"Arrival at the Port : {arrival_at_port.strftime('%d/%m/%Y %H:%M:%S')}")

    # Adding Ferry-Take-Off Date-Time
    ferry_delay = 2  # 2 hours delay for ferry take-off after truck arrival
    ferry_take_off = arrival_at_port + timedelta(hours=ferry_delay)
    print(f"Ferry-Take-Off Date-Time : {ferry_take_off.strftime('%d/%m/%Y %H:%M:%S')}")

    # Adding Arrival-At-Tarragona-Port
    Arrival_AtTarragona = ferry_take_off + timedelta(hours=72)
    print(f"Arrival-At-Tarragona-Port : {Arrival_AtTarragona.strftime('%d/%m/%Y %H:%M:%S')}")

    # Adding Arrival_At_Customer
    Arrival_At_Customer = Arrival_AtTarragona + timedelta(hours=customer_duration)
    print(f"Arrival_At_Customer : {Arrival_At_Customer.strftime('%d/%m/%Y %H:%M:%S')}")

    # Adding Unloading_Finishes
    Unloading_Finishes = Arrival_At_Customer + timedelta(hours=unloading_duration)
    print(f"Unloading Finishes at : {Unloading_Finishes.strftime('%d/%m/%Y %H:%M:%S')}")
else:
    print("Error: Some durations were not calculated.")

end_time = time.time()
execution_time = end_time - start_time
print(f"\nExecution time: {execution_time:.2f} seconds")