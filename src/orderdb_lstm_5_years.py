# -- coding: utf-8 --
"""
Created on Thu Apr 18 12:38:45 2024

@author: Selim
"""

import json
import os
import random
from pathlib import Path
import pandas as pd
from datetime import date, timedelta

from utils import get_data_dir, get_suppliers_lstm_file, empty_folder

# Rastgelelik tohumunu ayarlayın
random.seed(42)

# Read the suppliers file into a pandas dataframe
suppliers_df = pd.read_json(get_suppliers_lstm_file(), orient='records')

# Define the path to the order databases directory
order_dbs_dir = os.path.join(get_data_dir(), 'OrderDatabases_lstm_5_years')

empty_folder(order_dbs_dir)

# Define the start date
start_date = date(2019, 1, 1)

# Define the end date
end_date = date(2023, 12, 31)

# Define a random_supplier variable
suppliers = suppliers_df.to_dict(orient='records')


def run():
    last_order_id = 1
    # Loop through each day between the start and end dates
    for day in pd.date_range(start_date, end_date):
        # remove time from day
        ready_day = day.strftime("%Y-%m-%d")

        # Collect orders to a new array. each element in the array should represent a single order
        orders = []

        # Günlük sipariş sayısı kadar tedarikçi seçip karıştırın
        daily_suppliers = random.sample(suppliers, k=1000)
        random.shuffle(daily_suppliers)

        for supplier in daily_suppliers:
            order_day = (day - timedelta(days=random.randint(7, 10))).strftime("%Y-%m-%d")

            # Check if "ReadyHour" key exists in the supplier data
            if 'ReadyHour' in supplier:
                ready_hour = supplier['ReadyHour']  # Use existing value
            else:
                ready_hour = f"{random.randint(6, 22):02d}:00:00"  # Generate random hour

            # Merge "ReadyDate" and "ReadyHour" into a single key
            ready_date_and_time = f"{ready_day} {ready_hour}"

            # Assign a random Trailer_Type
            trailer_type = random.choice(['box', 'hanger'])

            order = {'OrderID': last_order_id,
                     'OrderDate': order_day,
                     'Ready_Date_and_Time': ready_date_and_time,
                     'SupID': supplier['SupID'],
                     'Arr. Lat': round(supplier['Arr. Lat'], 2),
                     'Arr. Lon': round(supplier['Arr. Lon'], 2),
                     'Trailer_Type': trailer_type}  # Add Trailer_Type key
            orders.append(order)
            last_order_id += 1

        # Sort orders based on Ready_Date_and_Time
        orders.sort(key=lambda x: x['Ready_Date_and_Time'])

        # create a dataframe from the orders array
        order_df = pd.DataFrame(orders)

        order_df.to_json(os.path.join(order_dbs_dir, f'OrderDatabase_lstm_5_years-{ready_day}.json'), orient='records')


run()  # Run the function to generate orders














