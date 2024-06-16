import os
import json
from datetime import datetime, timedelta

from utils import get_truck_dbs_dir

truck_dbs_dir = get_truck_dbs_dir()


def update_date_for_specific_items(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    for item in data:
        if 'Available_Date_and_Time' in item and item['Available_Date_and_Time'] != '00:00:00':
            existing_date_time = item['Available_Date_and_Time']
            date_time_parts = existing_date_time.split()

            # Check if the value has both date and time parts
            if len(date_time_parts) == 2:
                existing_date_str, existing_time_str = date_time_parts

                existing_date = datetime.strptime(existing_date_str, '%Y-%m-%d')
                new_date = existing_date + timedelta(days=1)
                new_date_str = new_date.strftime('%Y-%m-%d')

                item['Available_Date_and_Time'] = f"{new_date_str} {existing_time_str}"

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def update_files():
    for file_name in os.listdir(truck_dbs_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(truck_dbs_dir, file_name)
            update_date_for_specific_items(file_path)
            print(f"Updated file: {file_name}")


update_files()
