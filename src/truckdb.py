import random

import pandas as pd
from datetime import date, timedelta
import os

from src.utils import get_truck_dbs_dir, empty_folder

empty_folder(get_truck_dbs_dir())

# Define the start date
start_date = date(2023, 11, 24)

# Define the end date
end_date = start_date + timedelta(days=30)


def run():
    last_id=0
    # Loop through each day between the start and end dates
    for day in pd.date_range(start_date, end_date):
        # remove time from day
        day = day.strftime("%Y-%m-%d")

        # Collect trucks to a new array. each element in the array should represent a single truck
        trucks = []
        trucks_count = random.randint(60, 80)
        pending_trucks_count = random.randint(10, 20)
        # if day is start_date, generate trucks for pending trucks
        if day == start_date.strftime("%Y-%m-%d") or day == (start_date + timedelta(days=1)).strftime("%Y-%m-%d"):
            last_id = generate_trucks(last_id, trucks, trucks_count, day, True)
        else:
            # generate trucks for the rest of the days
            last_id = generate_trucks(last_id, trucks, trucks_count, day, False)
            last_id = generate_trucks(last_id, trucks, pending_trucks_count, day, True)

        # create a dataframe from the trucks array
        truck_df = pd.DataFrame(trucks)

        truck_df.to_json(os.path.join(get_truck_dbs_dir(), f'TruckDatabase-{day}.json'), orient='records')


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
                       'Dep. Lat': location['latitude'],
                       'Dep. Lon': location['longitude'],
                       'Available_Date_and_Time': available_date_and_time})
        coordinates.append(location)
        last_id += 1
    return last_id


def get_random_location(pending=False):
    if pending:
        return random.uniform(32.49, 32.50), random.uniform(39.45, 39.50)
    return random.uniform(37.2, 39.5), random.uniform(26, 45)


def get_random_coordinates(pending=False):
    latitude, longitude = get_random_location(pending)
    return {'latitude': latitude, 'longitude': longitude}


run() # TODO : Delete
