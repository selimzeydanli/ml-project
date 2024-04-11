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
    last_order_id = 1
    # Loop through each day between the start and end dates
    for day in pd.date_range(start_date, end_date):
        # remove time from day
        ready_day = day.strftime("%Y-%m-%d")

        # Collect orders to a new array. each element in the array should represent a single order
        orders = []
        for i in range(random.randint(10, 20)):
            random_supplier = random.choice(suppliers)
            order_day = (day - timedelta(days=random.randint(7, 10))).strftime("%Y-%m-%d")
            print(random.choice(suppliers))
            orders.append({'OrderID': last_order_id,
                           'OrderDate': order_day,
                           'ReadyDate': ready_day,
                           'SupID': random_supplier['SupID'],
                           'Arr. Lat': random_supplier['Arr. Lat'],
                           'Arr. Lon': random_supplier['Arr. Lon']})
            last_order_id += 1

        # create a dataframe from the orders array
        order_df = pd.DataFrame(orders)

        order_df.to_json(os.path.join(order_dbs_dir, f'OrderDatabase-{ready_day}.json'), orient='records')

run() # TODO : DELETE