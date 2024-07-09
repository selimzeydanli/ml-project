import random

import pandas as pd
from datetime import date, timedelta
import os

import pending

from src.utils import get_truck_dbs_lstm_5_years_dir, empty_folder

empty_folder(get_truck_dbs_lstm_5_years_dir())

# Define the start date
start_date = date(2019, 1, 1)

# Define the end date
end_date = date(2023, 12, 31)


def run():
    last_id=0
    # Loop through each day between the start and end dates
    for day in pd.date_range(start_date, end_date):
        # remove time from day
        day = day.strftime("%Y-%m-%d")

        # Collect trucks to a new array. each element in the array should represent a single truck
        trucks = []
        trucks_count = random.randint(1000, 1100)
        pending_trucks_count = random.randint(200, 300)
        # if day is start_date, generate trucks for pending trucks
        if day == start_date.strftime("%Y-%m-%d") or day == (start_date + timedelta(days=1)).strftime("%Y-%m-%d"):
            last_id = generate_trucks(last_id, trucks, trucks_count, day, True)
        else:
            # generate trucks for the rest of the days
            last_id = generate_trucks(last_id, trucks, trucks_count, day, False)
            last_id = generate_trucks(last_id, trucks, pending_trucks_count, day, True)

        # create a dataframe from the trucks array
        truck_df = pd.DataFrame(trucks)

        truck_df.to_json(os.path.join(get_truck_dbs_lstm_5_years_dir(), f'TruckDatabase_lstm_5_years_{day}.json'), orient='records')


def generate_trucks(last_id, trucks, trucks_count, available_day, pending):
    coordinates = []
    for i in range(trucks_count):
        location = get_random_coordinates(pending)
        if (pending == False) :
            while (coordinates.__contains__(location)):
                location = get_random_location(False)
        available_hour = f"{random.randint(6, 22):02d}:00:00"  # Generate random hour
        available_date_and_time = f"{available_day} {available_hour}"
        trucks.append({'TruckID': last_id,
                       'TrailerID': last_id,
                       'TrailerType': ['box', 'hanger'][i % 2],
                       'Dep. Lat': round(location['latitude'],2),
                       'Dep. Lon': round(location['longitude'],2),
                       'Available_Date_and_Time': available_date_and_time})
        coordinates.append(location)
        last_id += 1
    return last_id


def get_random_location(pending=False):
    if pending:
        return random.uniform(38.30, 38.35), random.uniform(26.30, 26.35)
    return random.uniform(36, 41), random.uniform(26, 49)


def get_random_coordinates(pending=False):
    latitude, longitude = get_random_location(pending)
    return {'latitude': latitude, 'longitude': longitude}


run() # TODO : Delete
