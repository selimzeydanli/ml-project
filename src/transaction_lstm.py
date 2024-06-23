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
        lat_Tar, lon_Tar = 41.11, 1.25
        lat_Cus, lon_Cus = 42.25, 1.25
        random_hours_loading = random.uniform(4.5, 13)
        random_hours_unloading = random.uniform(3, 5)
        job_entry = {}
        for k, truck_trip_information in triptimes.items():
            truck_id = truck_df.loc[k, "TruckID"]
            truck_type = truck_df.loc[k, "TrailerType"]
            available_time_str = truck_df.loc[k, "Available_Date_and_Time"]
            available_time = datetime.strptime(available_time_str, "%Y-%m-%d %H:%M:%S")
            tripstart_time = subtract_hours_from_datetime(ready_datetime_str, truck_trip_information["Duration_to_supplier (h)"])
            ready_datetime = datetime.strptime(ready_datetime_str, "%Y-%m-%d %H:%M:%S")
            take_off_date_time = (ready_datetime + timedelta(hours=random_hours_loading))

            random_speed = round(random.randint(10, 80), 2)

            if tripstart_time >= available_time:
                dist_to_port = haversine(port_lat, port_long, supplier_latitude, supplier_longitude)
                duration_to_port = dist_to_port / random_speed
                port_arrival = ready_datetime + timedelta(hours=6 + duration_to_port)

                day_name = get_day_name(port_arrival)
                ferry_date_time = next_sunday(port_arrival)
                ferry_date_time = set_time(ferry_date_time)
                arrival_tarragona = ferry_date_time + timedelta(hours=72)
                speed_to_customer = round(random.randint(10, 80), 2)
                distance_to_customer = round(haversine(lat_Tar, lon_Tar, lat_Cus, lon_Cus),2)
                time_to_customer = round(distance_to_customer / speed_to_customer,2)
                arrival_customer = arrival_tarragona + timedelta(hours=time_to_customer)
                unloading_complete_time = arrival_customer + timedelta(hours=random_hours_unloading)
                status = "Free"

                job_entry = {
                    "Job_ID": job_id,
                    "Order_ID": order_id,
                    "Order_Date": orderDateStr,
                    "Ready_Date_Time": ready_datetime_str,
                    "Job_Date_time": tripstart_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Sup_ID": sup_id,
                    "Trailer_Type": truck_type,
                    "Truck_ID": truck_id,
                    "Truck_Latitude": (truck_locations[k]["Departure_Latitude"]),
                    "Truck_Longitude": (truck_locations[k]["Departure_Longitude"]),
                    "Supplier_Latitude": (supplier_latitude),
                    "Supplier_Longitude": (supplier_longitude),
                    "Distance_To_Supplier(km)": round(truck_trip_information["Distance_to_supplier"], 2),
                    "Speed_To_Supplier(km/h)": truck_trip_information["Random_Speed (km/h)"],
                    "Duration_To_Supplier(h)": round(truck_trip_information["Duration_to_supplier (h)"], 2),
                    "Take_off_Date_Time": take_off_date_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Duration_Loading(h)": round((take_off_date_time - ready_datetime).total_seconds()/3600,2),
                    "Port_Latitude": "38.42",
                    "Port_Longitude": "27.14",
                    "Port_Arrival_Date_Time": port_arrival.strftime("%Y-%m-%d %H:%M:%S"),
                    "Distance_To_Port(km)": round(dist_to_port, 2),
                    "Speed_To_Port(km/h)": random_speed,
                    "Duration_To_Port(h)": round(duration_to_port, 2),
                    "Day_Name": day_name,
                    "Ferry_Date_Time": ferry_date_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Arrival_At_Tarragona": arrival_tarragona.strftime("%Y-%m-%d %H:%M:%S"),
                    "Tarragona_Latitude": lat_Tar,
                    "Tarragona_Longitude": lon_Tar,
                    "Customer_Latitude": lat_Cus,
                    "Customer_Longitude": lon_Cus,
                    "Distance_To_Customer(km)": distance_to_customer,
                    "Speed_To_Customer(km/h)": speed_to_customer,
                    "Duration_To_Customer(h)": time_to_customer,
                    "Arrival_At_Customer": arrival_customer.strftime("%Y-%m-%d %H:%M:%S"),
                    "Unloading_Complete_Date_Time": unloading_complete_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Duration_Unloading(h)": round((unloading_complete_time - arrival_customer).total_seconds() / 3600, 2),
                    "Status": status,
                }

                job_entries.append(job_entry)
                job_id += 1
                truck_df = truck_df.drop(k)
                break

        if not job_entry:
            job_entry = {
                "Job_ID": job_id,
                "Order_ID": order_id,
                "Order_Date": "NaN",
                "Ready_Date_Time": "NaN",
                "Job_Date_Time": "NaN",
                "Sup_ID": sup_id,
                "Trailer_Type": "NaN",
                "Truck_ID": "NaN",
                "Truck_Location": "NaN",
                "Supplier_Location": "NaN",
                "Distance_To_Supplier(km)": "NaN",
                "Speed_To_Supplier(km/h)": "NaN",
                "Duration_To_Supplier(h)": "NaN",
                "Take_off_Date_Time": "NaN",
                "Duration_Loading(h)": "NaN",
                "Port_Latitude": "NaN",
                "Port_Longitude": "NaN",
                "Port_Arrival_Date_Time": "NaN",
                "Distance_To_Port(km)": "NaN",
                "Duration_To_Port(h)": "NaN",
                "Speed(km/h)": "NaN",
                "Day_Name": "NaN",
                "Ferry_Date_Time": "NaN",
                "Arrival_At_Tarragona": "NaN",
                "Tarragona_Latitude": "NaN",
                "Tarragona_Longitude": "NaN",
                "Customer_Latitude": "NaN",
                "Customer_Longitude": "NaN",
                "Distance_To_Customer(km)": "NaN",
                "Speed_To_Customer(km/h)": "NaN",
                "Duration_To_Customer(h)": "NaN",
                "Arrival_At_Customer": "NaN",
                "Unloading_Complete_Date_Time": "NaN",
                "Duration_Unloading(h)": "NaN",
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
