import os
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from src.utils import get_order_dbs_dir, read_json_to_dataframe, get_truck_dbs_dir, get_plotting_dir, empty_folder

empty_folder(get_plotting_dir())

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
def next_sunday(date):
    days_until_sunday = (6 - date.weekday()) % 7
    return date + timedelta(days=days_until_sunday)

def subtract_hours_from_datetime(input_datetime:str, hours_to_subtract:float):
    # Convert input_datetime to datetime object if it's not already
    if not isinstance(input_datetime, datetime):
        input_datetime = datetime.strptime(input_datetime, '%Y-%m-%d %H:%M:%S')

    # Calculate timedelta for the given hours
    subtract_timedelta = timedelta(hours=hours_to_subtract)

    # Subtract the timedelta from input_datetime
    result_datetime = input_datetime - subtract_timedelta

    return result_datetime


import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def plot_coordinates(origins, destinations, filename):
    # Create a plot with a geo-projection
    fig = plt.figure(figsize=(10, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()

    # Set the extent to cover Turkey (approximately)
    ax.set_extent([25, 45, 36, 42], crs=ccrs.PlateCarree())  # Longitude from 25 to 45, Latitude from 35 to 43

    # Check that origins and destinations are the same length
    if len(origins) != len(destinations):
        raise ValueError("Origins and destinations lists must be of the same length.")

    # Handles for legend entries
    handles = []

    # Loop through the origins and destinations
    for (lat1, lon1), (lat2, lon2) in zip(destinations, origins):
        # Plot the line connecting each point
        line = ax.plot([lon1, lon2], [lat1, lat2], color="black", linewidth=1, transform=ccrs.Geodetic(),
                       label="Connection Line", linestyle="--")[0]

        # Plot the origin point in blue
        origin_point = \
        ax.plot(lon1, lat1, marker='o', color='blue', markersize=7, transform=ccrs.Geodetic(), label="Origin",
                alpha=0.5)[0]

        # Plot the destination point in red
        destination_point = \
        ax.plot(lon2, lat2, marker='o', color='red', markersize=7, transform=ccrs.Geodetic(), label="Destination",
                alpha=0.5)[0]

    # Add gridlines and labels
    ax.gridlines(draw_labels=True)

    # Add legend if not already added
    if handles == []:
        handles.extend([origin_point, destination_point])
        labels = ["Destination", "Origin"]
        ax.legend(handles, labels)

    # Display the plot
    # plt.show()
    plt.savefig(filename + '.jpg', format='jpg')


orderDate = datetime(2023, 11, 26)

endLoopDate = datetime(2023, 12, 25)

while orderDate <= endLoopDate:
    orderDateStr = orderDate.strftime("%Y-%m-%d")
    truckDateFirst = orderDate - timedelta(days=1)
    truckDateFirstStr = truckDateFirst.strftime("%Y-%m-%d")
    truckDateSecond = orderDate - timedelta(days=2)

    truckDateSecondStr = truckDateSecond.strftime("%Y-%m-%d")
    print(orderDateStr, truckDateFirstStr, truckDateSecondStr)

    order_df = read_json_to_dataframe(os.path.join(get_order_dbs_dir(), f'OrderDatabase-{orderDateStr}.json'))
    truck_1_df = read_json_to_dataframe(os.path.join(get_truck_dbs_dir(), f'TruckDatabase-{truckDateFirstStr}.json'))
    truck_2_df = read_json_to_dataframe(os.path.join(get_truck_dbs_dir(), f'TruckDatabase-{truckDateSecondStr}.json'))
    truck_df = pd.concat([truck_1_df, truck_2_df]).reset_index(drop=True)

    job_id = 1
    job_entries = []

    for order in order_df.iterrows():
        order_id = order[1]["OrderID"]
        sup_id = order[1]["SupID"]
        order_type = order[1]["Trailer_Type"]

        origin_lat = order[1]["Arr. Lat"]
        origin_long = order[1]["Arr. Lon"]

        ready_datetime = order[1]["Ready_Date_and_Time"]

        truck_locations = truck_df[truck_df["TrailerType"] == "box"].iloc[:, [3, 4]].to_dict(orient="index")
        truck_locations = {k: (v["Dep. Lat"], v["Dep. Lon"]) for k, v in truck_locations.items()}

        closest_truck_id, distance = find_closest(origin_lat, origin_long, truck_locations)

        trip_duration = distance / 50

        job_starttime = str(subtract_hours_from_datetime(ready_datetime, trip_duration))

        job_entry = [job_id, order_id, sup_id, closest_truck_id, distance, job_starttime, trip_duration]
        job_entries.append(job_entry)

        job_id += 1
        truck_df = truck_df[truck_df["TruckID"] != closest_truck_id]

    checkout_df = pd.DataFrame(job_entries, columns=["JobID", "OrderID", "SupID", "TruckID", "Distance", "JobDatetime",
                                                     "JobDuration"])

    job_id = 1
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

        triptimes = {k: haversine(origin_lat, origin_long, v[0], v[1]) / 50 for k, v in truck_locations.items()}
        triptimes = {k: v for k, v in sorted(triptimes.items(), key=lambda item: item[1])}

        job_entry = None
        for k, v in triptimes.items():
            truck_id = truck_df_copy.loc[k, "TruckID"]
            available_time_str = truck_df_copy.loc[k, "Available_Date_and_Time"]
            available_time = datetime.strptime(available_time_str, '%Y-%m-%d %H:%M:%S')
            tripstart_time = subtract_hours_from_datetime(ready_datetime_str, v)
            ready_datetime = datetime.strptime(ready_datetime_str, '%Y-%m-%d %H:%M:%S')

            if tripstart_time >= available_time:
                dist_to_port = haversine(port_lat, port_long, origin_lat, origin_long)
                duration_to_port = dist_to_port / 50
                port_arrival = ready_datetime + timedelta(hours=6 + duration_to_port)
                day_name = get_day_name(port_arrival)
                ferry_date_time = next_sunday(port_arrival)
                arrival_tarragona = ferry_date_time + timedelta(hours=72)
                arrival_inditex = arrival_tarragona + timedelta(hours=6)
                unloading_complete_time = arrival_inditex + timedelta(hours=6)
                status = "Free"
                job_entry = [job_id, order_id, sup_id, order_type, truck_id, truck_locations[k],
                             (origin_lat, origin_long),
                             v * 50, tripstart_time, v, ready_datetime, ready_datetime + timedelta(hours=6),
                             port_arrival, day_name, ferry_date_time, arrival_tarragona, arrival_inditex,
                             unloading_complete_time, status]

                job_entries.append(job_entry)

                job_id += 1
                truck_df_copy = truck_df_copy[truck_df_copy["TruckID"] != truck_id]
                break

        if not job_entry:
            job_entry = [job_id, order_id, sup_id, order_type, "Nan", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN",
                         "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN"]
            job_entries.append(job_entry)
            job_id += 1

    checkout_df = pd.DataFrame(job_entries,
                               columns=["JobID", "OrderID", "SupID", "TrailerType", "TruckID", "TruckLocation",
                                        "CustomerLocation",
                                        "Distance", "JobDatetime", "JobDuration", "ReadyDatetime", "TakeoffDatetime",
                                        "PortArrivalDatetime", "DayName", "FerryDateTime", "ArrivalTarragona",
                                        "ArrivalInditex", "UnloadingCompleteTime", "Status"])

    checkout_df.to_json(os.path.join(get_plotting_dir(), f'Plotting-{orderDateStr}.json'), orient='records', lines=True)
    plot_coordinates(checkout_df["TruckLocation"], checkout_df["CustomerLocation"], os.path.join(get_plotting_dir(), f'Plotting-{orderDateStr}'))
    orderDate = orderDate + timedelta(days=1)
