import datetime
import json
import os
import random
import shutil
from pathlib import Path

import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).parent.parent

def get_data_dir() -> str:
    return os.path.join(get_project_root(), 'data')

def get_input_dir() -> str:
    return os.path.join(get_project_root(), 'input')

def get_suppliers_file() -> str:
    return os.path.join(get_input_dir(),'suppliers.json')

def get_order_dbs_dir() -> str:
    return os.path.join(get_data_dir(), 'OrderDatabases')

def get_truck_dbs_dir() -> str:
    return os.path.join(get_data_dir(), 'TruckDatabases')

def get_assignment_dbs_dir() -> str:
    return os.path.join(get_data_dir(), 'AssignmentDatabases')

def get_transaction_dbs_dir() -> str:
    return os.path.join(get_data_dir(), 'TransactionDatabases')

def get_plotting_dir() -> str:
    return os.path.join(get_data_dir(), 'Plotting')

def get_truck_plotting_dir() -> str:
    return os.path.join(get_data_dir(), 'TruckEntirePlotting')

# define a function to read json into dataframe
def read_json_to_dataframe(filename: str) -> pd.DataFrame:
    with open(filename, 'r') as file:
        data = json.load(file)

    return pd.DataFrame(data)

def empty_folder(dbs_dir):
    print("Deleting old data")
    # if folder is empty return
    if os.listdir(dbs_dir) == []:
        return
    # delete all files in folde
    for file in os.listdir(dbs_dir):
        # skip .gitkeep file
        if file == '.gitkeep':
            continue
        current_file = os.path.join(dbs_dir, file)
        try:
            if os.path.exists(current_file):
                if os.path.isdir(current_file):
                    shutil.rmtree(current_file)
                os.remove(current_file)
        except Exception as e:
            print('Error during delete, continue')

def clean_generated_data():
    print("Deleting old data")
    for file in os.listdir(get_data_dir()):
        # skip .gitkeep file
        if file == '.gitkeep':
            continue
        empty_folder(os.path.join(get_data_dir(), file))

def get_random_time(day):
    random_time = datetime.time(hour=random.randint(0, 23), minute=random.randint(0, 59), second=random.randint(0, 59))
    return datetime.datetime.combine(day, random_time).strftime("%Y-%m-%d %H:%M:%S")

# clean_generated_data()