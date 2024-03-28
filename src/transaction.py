# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:42:50 2024

@author: Selim
"""
import os.path

from src.utils import get_assignment_dbs_dir, read_json_to_dataframe, get_truck_dbs_dir, get_order_dbs_dir, \
    get_transaction_dbs_dir, empty_folder

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 1 13:06:54 2023

@author: Selim
"""

import pandas as pd
import shutil
from datetime import datetime, timedelta

empty_folder(get_transaction_dbs_dir())

# Create a function to generate Transactions for a given day
def generate_transactions(day):
    # Calculate the date for the current day
    current_date = datetime(2023, 11, 24) + timedelta(days=day)
    current_date_str = current_date.strftime("%Y-%m-%d")

    # Read data from files
    try:
        assignments = read_json_to_dataframe(os.path.join(get_assignment_dbs_dir(), f'Assignments-{current_date_str}.json'))
        truck_db = read_json_to_dataframe(os.path.join(get_truck_dbs_dir(), f'TruckDatabase-{current_date_str}.json'))
        order_db = read_json_to_dataframe(os.path.join(get_order_dbs_dir(), f'OrderDatabase-{current_date_str}.json'))
    except ValueError as e:
        print(f"Error reading JSON file: {e}")
        raise

    try:
        indexes = ['Dep. Lat', 'Dep. Lon', 'Arr. Lat', 'Arr. Lon', 'SupplierID', 'TruckID', 'TrailerID', 'TrailerType',
                   'OrderID', 'OrderDate']

        merged_data = pd.DataFrame(index=indexes).from_records(assignments)
        merged_data['TrailerType'] = truck_db['TrailerType']
        merged_data['OrderID'] = order_db['OrderID']
        merged_data['OrderDate'] = order_db['OrderDate']
    except KeyError as e:
        print(f"Error merging dataframes: {e}")
        raise

    # Print the filled-in TransactionDatabase
    print(merged_data)

    # Save the filled-in TransactionDatabase to a new file
    merged_data.to_json(os.path.join(get_transaction_dbs_dir(), f'TransactionDatabase-{current_date_str}.json'), orient='records')

    # Additional message
    print(
        f'\nFilled-in TransactionDatabase for {current_date.strftime("%Y-%m-%d")} saved to: {get_transaction_dbs_dir()}')


def run():
    # Create Transactions for 30 consecutive days
    for day in range(30):
        generate_transactions(day)
