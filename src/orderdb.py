import json
import os
import random
from pathlib import Path

import pandas as pd
from datetime import date, timedelta, datetime

from src.utils import get_data_dir, get_suppliers_file, empty_folder

# Read the suppliers file into a pandas dataframe
suppliers_df = pd.read_json(get_suppliers_file(), orient='records')

# Define the path to the order databases directory
order_dbs_dir = os.path.join(get_data_dir(), 'OrderDatabases')

empty_folder(order_dbs_dir)

# Define the start date
start_date = date(2023, 11, 24)

# Define the end date
end_date = start_date + timedelta(days=30)

# Define a random_supplier variable
suppliers = suppliers_df.sample(30).to_dict(orient='records')


def generate_possible_hours():
    possible_hours = [datetime.strptime(f"{h:02}:00:00", "%H:%M:%S") for h in range(6, 23)]
    random.shuffle(possible_hours)
    return possible_hours


# Generate all possible hours within the range (06:00:00 to 22:00:00) initially
possible_hours = generate_possible_hours()


def run():
    global possible_hours  # Declare possible_hours as global

    last_order_id = 1
    # Loop through each day between the start and end dates
    for day in pd.date_range(start_date, end_date):
        # remove time from day
        ready_day = day.strftime("%Y-%m-%d")

        # Collect orders to a new array. each element in the array should represent a single order
        orders = []
        while len(orders) < 20:  # Ensure at least 20 orders are generated per day
            random_supplier = random.choice(suppliers)
            order_day = (day - timedelta(days=random.randint(7, 10))).strftime("%Y-%m-%d")
            if not possible_hours:
                possible_hours = generate_possible_hours()  # Regenerate if the list is empty
            ready_hour = possible_hours.pop()  # Get the next available ready_hour from the shuffled list
            orders.append({'OrderID': last_order_id,
                           'OrderDate': order_day,
                           'ReadyDate': ready_day,
                           'ReadyHour': ready_hour.strftime("%H:%M:%S"),
                           'SupID': random_supplier['SupID'],
                           'Arr. Lat': random_supplier['Arr. Lat'],
                           'Arr. Lon': random_supplier['Arr. Lon']})
            last_order_id += 1

        # create a dataframe from the orders array
        order_df = pd.DataFrame(orders)

        # Save the dataframe to a JSON file with only specific columns
        file_path = os.path.join(order_dbs_dir, f'OrderDatabase-{ready_day}.json')
        order_df[['OrderID', 'OrderDate', 'ReadyDate', 'ReadyHour', 'SupID', 'Arr. Lat', 'Arr. Lon']].to_json(file_path,
                                                                                                              orient='records')
        print("File saved to:", file_path)


run()
