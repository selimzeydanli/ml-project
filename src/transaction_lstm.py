import os
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta
import pandas as pd
import random

import cartopy.crs as ccrs

from src.utils import (
    get_order_dbs_lstm_dir,
    read_json_to_dataframe,
    get_truck_dbs_lstm_dir,
    empty_folder,
    get_transaction_dbs_lstm_dir,
)


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
    min_distance = float("inf")
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
    return date.strftime("%A")


# Function to calculate the next Sunday from a given date
# def next_sunday(date):
# days_until_sunday = (6 - date.weekday()) % 7
# return date + timedelta(days=days_until_sunday)

from datetime import datetime, timedelta


# Function to find the next Sunday from a given date
def next_sunday(date):
    days_until_sunday = (6 - date.weekday() + 7) % 7  # Calculate days until next Sunday
    next_sunday_date = date + timedelta(
        days=days_until_sunday
    )  # Add days to get to next Sunday
    return next_sunday_date


# Function to set the time to 23:30
def set_time(date):
    return date.replace(hour=23, minute=30, second=0)


# Assuming port_arrival is your initial date/time
port_arrival = datetime.now()  # Example date/time, replace it with your actual value

# Calculate the next Sunday
next_sunday_date = next_sunday(port_arrival)

# Set the time to 23:30
ferry_date_time = set_time(next_sunday_date)


def subtract_hours_from_datetime(input_datetime: str, hours_to_subtract: float):
    # Convert input_datetime to datetime object if it's not already
    if not isinstance(input_datetime, datetime):
        input_datetime = datetime.strptime(input_datetime, "%Y-%m-%d %H:%M:%S")

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

    order_df = read_json_to_dataframe(
        os.path.join(
            get_order_dbs_lstm_dir(), f"OrderDatabase_lstm-{orderDateStr}.json"
        )
    )
    truck_1_df = read_json_to_dataframe(
        os.path.join(
            get_truck_dbs_lstm_dir(), f"TruckDatabase_lstm-{truckDateFirstStr}.json"
        )
    )
    truck_2_df = read_json_to_dataframe(
        os.path.join(
            get_truck_dbs_lstm_dir(), f"TruckDatabase_lstm-{truckDateSecondStr}.json"
        )
    )
    truck_df = pd.concat([truck_1_df, truck_2_df]).reset_index(drop=True)


    # I am searching for the closest truck but each truck has a random speed which simulates different road conditions
    job_entries = []
    port_lat = 38.42
    port_long = 27.14
    for index, order in order_df.iterrows():
        order_id = order["OrderID"]
        sup_id = order["SupID"]
        order_type = order["Trailer_Type"]

        supplier_latitude = order["Arr. Lat"]
        supplier_longitude = order["Arr. Lon"]
        ready_datetime_str = order["Ready_Date_and_Time"]

        indexes = truck_df["TrailerType"] == order_type
        truck_locations = truck_df[indexes].iloc[:, [3, 4]].to_dict(orient="index")
        truck_locations = {
            k: {
                "Departure_Latitude": v["Dep. Lat"],
                "Departure_Longitude": v["Dep. Lon"],
                "Distance_to_supplier": haversine(supplier_latitude, supplier_longitude, v["Dep. Lat"], v["Dep. Lon"]),
                "Random_Speed (km/h)": round(random.randint(10, 80), 2)
                } for k, v in truck_locations.items()
        }

        triptimes = {
            k: {
                "Distance_to_supplier": truck_information["Distance_to_supplier"],
                "Duration_to_supplier (h)": truck_information["Distance_to_supplier"] / truck_information["Random_Speed (km/h)"],
                "Random_Speed (km/h)": truck_information["Random_Speed (km/h)"]
            }
            for k, truck_information in truck_locations.items()
        }


        triptimes = {
            k: v for k, v in sorted(triptimes.items(), key=lambda item: item[1]["Duration_to_supplier (h)"])
        }

        job_entry = {}
        for k, truck_trip_information in triptimes.items():
            truck_id = truck_df.loc[k, "TruckID"]
            truck_type = truck_df.loc[k, "TrailerType"]
            available_time_str = truck_df.loc[k, "Available_Date_and_Time"]
            available_time = datetime.strptime(available_time_str, "%Y-%m-%d %H:%M:%S")
            tripstart_time = subtract_hours_from_datetime(ready_datetime_str, truck_trip_information["Duration_to_supplier (h)"])
            ready_datetime = datetime.strptime(ready_datetime_str, "%Y-%m-%d %H:%M:%S")
            random_speed = round(random.randint(10, 80), 2)

            if tripstart_time >= available_time:
                dist_to_port = haversine(port_lat, port_long, supplier_latitude, supplier_longitude)
                duration_to_port = dist_to_port / random_speed
                port_arrival = ready_datetime + timedelta(hours=6 + duration_to_port)

                day_name = get_day_name(port_arrival)
                ferry_date_time = next_sunday(port_arrival)
                ferry_date_time = set_time(ferry_date_time)
                arrival_tarragona = ferry_date_time + timedelta(hours=72)
                time_to_customer = 300 / round(random.randint(10, 80), 2)
                arrival_customer = arrival_tarragona + timedelta(hours=time_to_customer)
                unloading_complete_time = arrival_customer + timedelta(hours=6)
                status = "Free"

                job_entry = {
                    "JobID": job_id,
                    "OrderID": order_id,
                    "SupID": sup_id,
                    "Trailer Type": truck_type,
                    "TruckID": truck_id,
                    "TruckLocation": (truck_locations[k]["Departure_Latitude"],truck_locations[k]["Departure_Longitude"]),
                    "SupplierLocation": (supplier_latitude, supplier_longitude),
                    "DistanceToSupplier": round(truck_trip_information["Distance_to_supplier"], 2),
                    "SpeedToSupplier(km/h)": truck_trip_information["Random_Speed (km/h)"],
                    "JobDatetime": tripstart_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "DurationToSupplier(h)": round(truck_trip_information["Duration_to_supplier (h)"], 2),
                    "OrderDate": orderDateStr,
                    "ReadyDatetime": ready_datetime_str,
                    "TakeoffDatetime": (ready_datetime + timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                    "PortArrivalDatetime": port_arrival.strftime("%Y-%m-%d %H:%M:%S"),
                    "DistanceToPort(km)": round(dist_to_port, 2),
                    "DurationToPort(h)": round(duration_to_port, 2),
                    "Speed(km/h)": random_speed,
                    "DayName": day_name,
                    "FerryDateTime": ferry_date_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "ArrivalTarragona": arrival_tarragona.strftime("%Y-%m-%d %H:%M:%S"),
                    "ArrivalCustomer": arrival_customer.strftime("%Y-%m-%d %H:%M:%S"),
                    "UnloadingCompleteTime": unloading_complete_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Status": status,
                }

                job_entries.append(job_entry)
                job_id += 1
                truck_df = truck_df.drop(k)
                break

        if not job_entry:
            job_entry = {
                "JobID": job_id,
                "OrderID": order_id,
                "SupID": sup_id,
                "Trailer Type": "NaN",
                "TruckID": "NaN",
                "TruckLocation": "NaN",
                "SupplierLocation": "NaN",
                "DistanceToSupplier": "NaN",
                "SpeedToSupplier(km/h)": "NaN",
                "JobDatetime": "NaN",
                "DurationToSupplier(h)": "NaN",
                "OrderDate": "NaN",
                "ReadyDatetime": "NaN",
                "TakeoffDatetime": "NaN",
                "PortArrivalDatetime": "NaN",
                "DistanceToPort(km)": "NaN",
                "DurationToPort(h)": "NaN",
                "Speed(km/h)": "NaN",
                "DayName": "NaN",
                "FerryDateTime": "NaN",
                "ArrivalTarragona": "NaN",
                "ArrivalCustomer": "NaN",
                "UnloadingCompleteTime": "NaN",
                "Status": "NaN",
            }
            job_entries.append(job_entry)
            job_id += 1
    checkout_df = pd.DataFrame(job_entries)

    checkout_df.to_json(
        os.path.join(
            get_transaction_dbs_lstm_dir(),
            f"TransactionDatabase_lstm-{orderDateStr}.json",
        ),
        orient="records",
    )
    orderDate = orderDate + timedelta(days=1)

    pd.set_option("display.max_columns", None)

    try:
        # Attempt to get the terminal width
        terminal_width = os.get_terminal_size().columns
    except OSError:
        # If OSError occurs (not supported on Windows), set a default width
        terminal_width = 160  # Set a default width suitable for your screen

    # Set the display width
    pd.set_option("display.width", terminal_width)
