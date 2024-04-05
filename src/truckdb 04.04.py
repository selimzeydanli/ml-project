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
    # Loop through each day between the start and end dates
    for day in pd.date_range(start_date, end_date):
        # remove time from day
        day = day.strftime("%Y-%m-%d")

        # Collect trucks to a new array. each element in the array should represent a single truck
        trucks = []
        trucks_count = random.randint(40, 60)

        for i in range(trucks_count):
            location = get_random_coordinates()
            trucks.append({'TruckID': i,
                           'TrailerID': i,
                           'TrailerType': ['box', 'hanger'][i % 2],
                           'Dep. Lat': location['latitude'],
                           'Dep. Lon': location['longitude']})

        # create a dataframe from the trucks array
        truck_df = pd.DataFrame(trucks)

        truck_df.to_json(os.path.join(get_truck_dbs_dir(), f'TruckDatabase-{day}.json'), orient='records')


def get_random_location():
    pending = random.randint(0, 1)
    if pending:
        return random.uniform(32.49, 32.50), random.uniform(39.45, 39.50)
    return random.uniform(36, 42), random.uniform(26, 45)


def get_random_coordinates():
    latitude, longitude = get_random_location()
    return {'latitude': latitude, 'longitude': longitude}


run()
