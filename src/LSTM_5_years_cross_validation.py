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

from src.LSTM_5_years import predict_loading_duration, df_supplier, df_port, df_customer, predict_unloading_duration

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
    print(f"Standard Deviation of Duration: {std_duration:.2f}")

    distance = df['Distance'].iloc[0]  # Get the fixed distance value
    print()
    print(f"Distance_To_{scenario_name}(km)     : {distance:.2f}")
    print()

    speed = distance / mean_duration
    print(f"Predicted Duration To {scenario_name}(h)        : {mean_duration:.2f}")
    print()
    print(f"Predicted Speed To {scenario_name}(km/h)        : {speed:.2f}")
    print()
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

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)

    # Perform k-fold cross-validation for performance evaluation
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    r2_scores = []
    mae_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train_scaled), 1):
        X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
        y_train_fold, y_val_fold = y_train_scaled[train_index], y_train_scaled[val_index]

        model = create_lstm_model(X_train_fold.reshape(-1, 1, 1), y_train_fold, X_val_fold.reshape(-1, 1, 1), y_val_fold)

        y_pred_fold = model.predict(X_val_fold.reshape(-1, 1, 1))
        r2 = r2_score(y_val_fold, y_pred_fold)
        mae = mean_absolute_error(y_val_fold, y_pred_fold)

        r2_scores.append(r2)
        mae_scores.append(mae)

        print(f"Fold {fold}: R-squared: {r2:.4f}, MAE: {mae:.4f}")

    print(f"\n{scenario_name} Cross-Validation Results:")
    print(f"Average R-squared: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})")
    print(f"Average MAE: {np.mean(mae_scores):.4f} (+/- {np.std(mae_scores):.4f})")

    # Train final model on all training data
    final_model = create_lstm_model(X_train_scaled.reshape(-1, 1, 1), y_train_scaled, X_test_scaled.reshape(-1, 1, 1), y_test_scaled)

    # Evaluate on test set
    y_pred_test = final_model.predict(X_test_scaled.reshape(-1, 1, 1))
    r2_test = r2_score(y_test_scaled, y_pred_test)
    mae_test = mean_absolute_error(y_test_scaled, y_pred_test)

    print(f"\nTest Set Results:")
    print(f"R-squared: {r2_test:.4f}")
    print(f"MAE: {mae_test:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, scaler_y.inverse_transform(y_pred_test), alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Duration')
    plt.ylabel('Predicted Duration')
    plt.title(f'{scenario_name} Actual vs Predicted Duration')
    plt.show()

    print()
    distance = float(input(f"Enter Distance_To_{scenario_name}(km): "))
    print()
    prediction = make_prediction(final_model, scaler_X, scaler_y, distance)
    print()
    print(f"Predicted Duration_To_{scenario_name}(h)            : {prediction:.2f}")
    print()
    speed = distance / prediction
    print(f"Predicted Speed To {scenario_name}(km/h)            : {speed:.2f}")

    return final_model, scaler_X, scaler_y, float(prediction), distance

# ... (rest of the code remains the same)

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

# ... (rest of the code remains the same)