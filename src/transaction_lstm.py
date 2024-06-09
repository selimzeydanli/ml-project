import os
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta
import pandas as pd
import random

import cartopy.crs as ccrs

from src.utils import get_order_dbs_lstm_dir, read_json_to_dataframe, get_truck_dbs_lstm_dir, empty_folder, \
    get_transaction_dbs_lstm_dir


empty_folder(get_transaction_dbs_lstm_dir())


def haversine(lat1: float, lon1: float, lat2: float, lon2: float):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 6371 * c  # Radius of the Earth in kilometers
    return distance


def find_closest(origin_lat: float, origin_lon: float, points: dict):
    # dict: {id:(lat,long)}
    min_distance = float('inf')
    closest_point_id = None

    for point_id, point_coords in points.items():
        distance = haversine(origin_lat, origin_lon, point_coords[0], point_coords[1])
        if distance < min_distance:
            min_distance = distance
            closest_point_id = point_id

    return closest_point_id, min_distance


def get_day_name(date):
    """
    Get the day name from a datetime object.

    Args:
    - date: A datetime object representing the date.

    Returns:
    - A string representing the day name (e.g., 'Monday', 'Tuesday', etc.).
    """
    return date.strftime('%A')


# Function to calculate the next Sunday from a given date
#def next_sunday(date):
    #days_until_sunday = (6 - date.weekday()) % 7
    #return date + timedelta(days=days_until_sunday)

from datetime import datetime, timedelta

# Function to find the next Sunday from a given date
def next_sunday(date):
    days_until_sunday = (6 - date.weekday() + 7) % 7  # Calculate days until next Sunday
    next_sunday_date = date + timedelta(days=days_until_sunday)  # Add days to get to next Sunday
    return next_sunday_date

# Function to set the time to 23:30
def set_time(date):
    return date.replace(hour=23, minute=30)

# Assuming port_arrival is your initial date/time
port_arrival = datetime.now()  # Example date/time, replace it with your actual value

# Calculate the next Sunday
next_sunday_date = next_sunday(port_arrival)

# Set the time to 23:30
ferry_date_time = set_time(next_sunday_date)

def subtract_hours_from_datetime(input_datetime:str, hours_to_subtract:float):
    # Convert input_datetime to datetime object if it's not already
    if not isinstance(input_datetime, datetime):
        input_datetime = datetime.strptime(input_datetime, '%Y-%m-%d %H:%M:%S')

    # Calculate timedelta for the given hours
    subtract_timedelta = timedelta(hours=hours_to_subtract)

    # Subtract the timedelta from input_datetime
    result_datetime = input_datetime - subtract_timedelta

    return result_datetime


import cartopy.crs as ccrs


orderDate = datetime(2023, 1, 3)

endLoopDate = datetime(2023, 12, 31)

job_id = 1

while orderDate <= endLoopDate:
    orderDateStr = orderDate.strftime("%Y-%m-%d")
    truckDateFirst = orderDate - timedelta(days=1)
    truckDateFirstStr = truckDateFirst.strftime("%Y-%m-%d")
    truckDateSecond = orderDate - timedelta(days=2)
    truckDateSecondStr = truckDateSecond.strftime("%Y-%m-%d")
    print(orderDateStr, truckDateFirstStr, truckDateSecondStr)

    order_df = read_json_to_dataframe(os.path.join(get_order_dbs_lstm_dir(), f'OrderDatabase_lstm-{orderDateStr}.json'))
    truck_1_df = read_json_to_dataframe(os.path.join(get_truck_dbs_lstm_dir(), f'TruckDatabase_lstm-{truckDateFirstStr}.json'))
    truck_2_df = read_json_to_dataframe(os.path.join(get_truck_dbs_lstm_dir(), f'TruckDatabase_lstm-{truckDateSecondStr}.json'))
    truck_df = pd.concat([truck_1_df, truck_2_df]).reset_index(drop=True)


    job_entries = []

    for order in order_df.iterrows():
        order_id = order[1]["OrderID"]
        sup_id = order[1]["SupID"]
        order_type = order[1]["Trailer_Type"]

        origin_lat = order[1]["Arr. Lat"]
        origin_long = order[1]["Arr. Lon"]

        ready_datetime = order[1]["Ready_Date_and_Time"]
        indexes = truck_df["TrailerType"] == "box" #order_type
        truck_locations = truck_df[indexes].iloc[:, [3, 4]].to_dict(orient="index")
        truck_locations = {k: (v["Dep. Lat"], v["Dep. Lon"]) for k, v in truck_locations.items()}

        closest_truck_id, distance = find_closest(origin_lat, origin_long, truck_locations)

        trip_duration = distance / round(random.randint(10, 80), 2)

        job_starttime = (str(subtract_hours_from_datetime(ready_datetime, trip_duration)))

        job_entry = [job_id, order_id, sup_id, closest_truck_id, distance, job_starttime, trip_duration]
        job_entries.append(job_entry)

        job_id += 1
        truck_df = truck_df[truck_df["TruckID"] != closest_truck_id]

    #checkout_df = pd.DataFrame(job_entries, columns=["JobID", "OrderID", "SupID", "TruckID", "Distance", "JobDatetime",
                                                     #"JobDuration(h)"])

    job_entries = []
    truck_df_copy = truck_df.copy()
    port_lat = 38.42
    port_long = 27.14
    for order in order_df.iterrows():
        order_id = order[1]["OrderID"]
        sup_id = order[1]["SupID"]
        order_type = order[1]["Trailer_Type"]

        origin_lat = order[1]["Arr. Lat"]
        origin_long = order[1]["Arr. Lon"]

        ready_datetime_str = order[1]["Ready_Date_and_Time"]

        truck_locations = truck_df_copy[truck_df_copy["TrailerType"] == order_type].iloc[:, [3, 4]].to_dict(
            orient="index")
        truck_locations = {k: (v["Dep. Lat"], v["Dep. Lon"]) for k, v in truck_locations.items()}

        triptimes = {k: haversine(origin_lat, origin_long, v[0], v[1]) / round(random.randint(10, 80), 2) for k, v in truck_locations.items()}
        triptimes = {k: v for k, v in sorted(triptimes.items(), key=lambda item: item[1])}

        job_entry = None
        for k, v in triptimes.items():
            truck_id = truck_df_copy.loc[k, "TruckID"]
            truck_type = truck_df_copy.loc[k, "TrailerType"]
            available_time_str = truck_df_copy.loc[k, "Available_Date_and_Time"]
            available_time = datetime.strptime(available_time_str, '%Y-%m-%d %H:%M:%S')
            tripstart_time = subtract_hours_from_datetime(ready_datetime_str, v)
            ready_datetime = datetime.strptime(ready_datetime_str, '%Y-%m-%d %H:%M:%S')
            random_speed = round(random.randint(10, 80), 2)

            if tripstart_time >= available_time:
                dist_to_port = haversine(port_lat, port_long, origin_lat, origin_long)
                duration_to_port = dist_to_port / random_speed
                port_arrival = ready_datetime + timedelta(hours=6 + duration_to_port)

                day_name = get_day_name(port_arrival)
                ferry_date_time = next_sunday(port_arrival)
                arrival_tarragona = ferry_date_time + timedelta(hours=72)
                arrival_customer = arrival_tarragona + timedelta(hours=6)
                unloading_complete_time = arrival_customer + timedelta(hours=6)
                status = "Free"
                job_entry = [job_id, order_id, sup_id, order_type, truck_type, truck_id, truck_locations[k],
                             (origin_lat, origin_long),
                             round (v * random_speed,2), round(random_speed,2), tripstart_time.strftime("%Y-%m-%d %H:%M:%S"), round (v,2), orderDateStr, ready_datetime_str, (ready_datetime + timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                             port_arrival.strftime("%Y-%m-%d %H:%M:%S"), round(dist_to_port,2), round(duration_to_port,2), random_speed, day_name, ferry_date_time.strftime("%Y-%m-%d %H:%M:%S"), arrival_tarragona.strftime("%Y-%m-%d %H:%M:%S"), arrival_customer.strftime("%Y-%m-%d %H:%M:%S"),
                             unloading_complete_time.strftime("%Y-%m-%d %H:%M:%S"), status]

                job_entries.append(job_entry)

                job_id += 1
                truck_df_copy = truck_df_copy[truck_df_copy["TruckID"] != truck_id]
                break

        if not job_entry:
            job_entry = [job_id, order_id, sup_id, order_type, "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN",
                         "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN"]
            job_entries.append(job_entry)
            job_id += 1

    checkout_df = pd.DataFrame(job_entries,
                               columns=["JobID", "OrderID", "SupID", "RequestedTrailerType", "ProvidedTrailer Type", "TruckID", "TruckLocation",
                                        "SupplierLocation",
                                        "DistanceToSupplier", "SpeedToSupplier(km/h)", "JobDatetime", "DurationToSupplier(h)", "OrderDate", "ReadyDatetime", "TakeoffDatetime",
                                        "PortArrivalDatetime", "DistanceToPort(km)", "DurationToPort(h)", "Speed(km/h)", "DayName", "FerryDateTime", "ArrivalTarragona",
                                        "ArrivalCustomer", "UnloadingCompleteTime", "Status"])

    checkout_df.to_json(os.path.join(get_transaction_dbs_lstm_dir(), f'TransactionDatabase_lstm-{orderDateStr}.json'), orient='records')
    orderDate = orderDate + timedelta(days=1)

    print ()
    print ()

    pd.set_option('display.max_columns', None)

    try:
        # Attempt to get the terminal width
        terminal_width = os.get_terminal_size().columns
    except OSError:
        # If OSError occurs (not supported on Windows), set a default width
        terminal_width = 160  # Set a default width suitable for your screen

    # Set the display width
    pd.set_option('display.width', terminal_width)

