import json
import os
import random
from pathlib import Path

import pandas as pd
from datetime import date, timedelta

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


def run():
    # Loop through each day between the start and end dates
    for day in pd.date_range(start_date, end_date):
        # remove time from day
        day = day.strftime("%Y-%m-%d")

        # Collect orders to a new array. each element in the array should represent a single order
        orders = []
        for i in range(1, 31):
            random_supplier = random.choice(suppliers)
            print(random.choice(suppliers))
            orders.append({'OrderID': i,
                           'OrderDate': day,
                           'SupID': random_supplier['SupID'],
                           'Arr. Lat': random_supplier['Arr. Lat'],
                           'Arr. Lon': random_supplier['Arr. Lon']})

        # create a dataframe from the orders array
        order_df = pd.DataFrame(orders)

        order_df.to_json(os.path.join(order_dbs_dir, f'OrderDatabase-{day}.json'), orient='records')