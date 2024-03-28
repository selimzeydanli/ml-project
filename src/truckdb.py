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
        for i in range(1, 31):
            trucks.append({'TruckID': i,
                           'TrailerID': i,
                           'TrailerType': ['box', 'hanger'][i % 2],
                           'Dep. Lat': random.uniform(32.49, 32.50),
                           'Dep. Lon': random.uniform(39.45, 39.50)})

        # create a dataframe from the trucks array
        truck_df = pd.DataFrame(trucks)

        truck_df.to_json(os.path.join(get_truck_dbs_dir(), f'TruckDatabase-{day}.json'), orient='records')
