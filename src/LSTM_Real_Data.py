import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


class TransportDurationPredictor:
    def __init__(self, json_path):
        """
        Initialize the predictor with data from JSON file.

        :param json_path: Path to the JSON file containing transport data.
        """
        try:
            with open(json_path, 'r') as file:
                self.data = pd.DataFrame(json.load(file))
        except FileNotFoundError:
            print(f"Error: File not found at {json_path}")
            raise
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in file {json_path}")
            raise

        self.data['Departure_Time'] = pd.to_datetime(self.data['Departure_Time'])
        numeric_columns = [
            'Duration_Loading(h)', 'Duration_To_Port(h)',
            'Supplier_Latitude', 'Supplier_Longitude',
            'Port_Latitude', 'Port_Longitude'
        ]
        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        self.loading_scaler = MinMaxScaler()
        self.port_duration_scaler = MinMaxScaler()

    def predict_loading_duration(self, sup_id):
        """
        Predict the next loading duration for the given supplier ID.
        Use global fallback if no specific data is available.

        :param sup_id: Supplier ID.
        :return: Predicted loading duration.
        """
        durations = self.data[self.data['Sup_ID'] == sup_id]['Duration_Loading(h)'].dropna()

        if len(durations) == 0:
            print(f"No data available for Sup_ID: {sup_id}.")
            return None
        elif len(durations) == 1:
            print(f"Only one entry available for Sup_ID: {sup_id}. Using the existing data: {durations.iloc[0]:.2f} hours")
            return durations.iloc[0]
        elif len(durations) < 3:
            fallback_prediction = durations.mean()
            print(f"Insufficient data for Sup_ID: {sup_id}. Fallback prediction (mean): {fallback_prediction:.2f} hours")
            return fallback_prediction
        else:
            sequences = [durations.values[-3:].reshape(1, 3, 1)]
            scaled_sequences = self.loading_scaler.fit_transform(np.array(durations).reshape(-1, 1))
            model_input = np.array(scaled_sequences[-3:]).reshape(1, 3, 1)

            model = self.create_loading_model((3, 1))
            prediction = model.predict(model_input)
            prediction = self.loading_scaler.inverse_transform(prediction)
            return prediction[0, 0]

    def predict_port_duration(self, departure_time, sup_lat, sup_lon, port_lat, port_lon):
        """
        Predict the port duration for the given coordinates and departure time.
        Use global fallback if no specific data is available.

        :param departure_time: Departure time.
        :param sup_lat: Supplier latitude.
        :param sup_lon: Supplier longitude.
        :param port_lat: Port latitude.
        :param port_lon: Port longitude.
        :return: Predicted port duration.
        """
        mask = (
            (self.data['Supplier_Latitude'] == sup_lat) &
            (self.data['Supplier_Longitude'] == sup_lon) &
            (self.data['Port_Latitude'] == port_lat) &
            (self.data['Port_Longitude'] == port_lon)
        )
        durations = self.data[mask]['Duration_To_Port(h)'].dropna()

        if len(durations) == 0:
            print(f"No data available for the given coordinates.")
            return None
        elif len(durations) == 1:
            print(f"Only one entry available for the given coordinates. Using the existing data: {durations.iloc[0]:.2f} hours")
            return durations.iloc[0]
        elif len(durations) < 3:
            fallback_prediction = durations.mean()
            print(f"Insufficient data for the given coordinates. Fallback prediction (mean): {fallback_prediction:.2f} hours")
            return fallback_prediction
        else:
            sequences = [durations.values[-3:].reshape(1, 3, 1)]
            scaled_sequences = self.port_duration_scaler.fit_transform(np.array(durations).reshape(-1, 1))
            model_input = np.array(scaled_sequences[-3:]).reshape(1, 3, 1)

            model = self.create_port_duration_model((3, 1))
            prediction = model.predict(model_input)
            prediction = self.port_duration_scaler.inverse_transform(prediction)
            return prediction[0, 0]

    def create_loading_model(self, input_shape):
        """
        Create and return a placeholder LSTM model for loading duration prediction.

        :param input_shape: Shape of input sequences.
        :return: Compiled Keras model.
        """
        model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def create_port_duration_model(self, input_shape):
        """
        Create and return a placeholder LSTM model for port duration prediction.

        :param input_shape: Shape of input sequences.
        :return: Compiled Keras model.
        """
        model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model


# Main execution
if __name__ == "__main__":
    predictor = TransportDurationPredictor(r'C:\Users\Selim\Desktop\Deneme.json')

    while True:
        try:
            sup_id = input("Enter Supplier ID (Sup_ID): ")
            sup_id = int(sup_id)
            break
        except ValueError:
            print("Invalid Supplier ID. Please enter a numeric value.")

    while True:
        try:
            departure_time = input("Enter Departure Time (yyyy-mm-dd): ")
            departure_time = pd.to_datetime(departure_time)
            break
        except ValueError:
            print("Invalid Departure Time format. Please use yyyy-mm-dd.")

    while True:
        try:
            sup_lat = float(input("Enter Supplier Latitude: "))
            sup_lon = float(input("Enter Supplier Longitude: "))
            port_lat = float(input("Enter Port Latitude: "))
            port_lon = float(input("Enter Port Longitude: "))
            break
        except ValueError:
            print("Invalid coordinate value. Please enter numeric values.")

    print("\nPredicting Loading Duration...")
    loading_prediction = predictor.predict_loading_duration(sup_id)
    if loading_prediction is not None:
        print(f"Predicted Loading Duration: {loading_prediction:.2f} hours")

    print("\nPredicting Port Duration...")
    port_prediction = predictor.predict_port_duration(departure_time, sup_lat, sup_lon, port_lat, port_lon)
    if port_prediction is not None:
        print(f"Predicted Port Duration: {port_prediction:.2f} hours")
