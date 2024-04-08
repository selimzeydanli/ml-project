# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:41:45 2024

@author: Selim
"""

import os
import shutil
import json
import datetime
import nbformat
import time
import pandas as pd
from geopy.distance import geodesic
from ortools.linear_solver import pywraplp

from src.utils import read_json_to_dataframe, get_assignment_dbs_dir, empty_folder, get_truck_dbs_dir, \
    get_order_dbs_dir, get_random_time

empty_folder(get_assignment_dbs_dir())

# Initialize solver for MILP
solver = pywraplp.Solver.CreateSolver('SCIP')

if not solver:
    raise Exception("Solver not found!")


# Function to calculate distance between two coordinates
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).kilometers


# Function to solve the MILP problem
def solve_milp(distance_matrix, list_data):
    num_vehicles = len(distance_matrix[0])
    num_suppliers = len(distance_matrix[0])

    # Create variables for the assignment problem
    x = {}
    for i in range(num_vehicles):
        for j in range(num_suppliers):
            x[i, j] = solver.IntVar(0, 1, f'x[{i},{j}]')

    # Define constraints
    for i in range(num_vehicles):
        solver.Add(solver.Sum(x[i, j] for j in range(num_suppliers)) == 1)

    for j in range(num_suppliers):
        solver.Add(solver.Sum(x[i, j] for i in range(num_vehicles)) == 1)

    # Define the objective function
    solver.Minimize(
        solver.Sum(distance_matrix[i][j] * x[i, j] for i in range(num_vehicles) for j in range(num_suppliers)))

    # Solve the MILP
    solver.Solve()

    # Create the 'Assignments' DataFrame
    assignments = []
    for i in range(num_vehicles):
        for j in range(num_suppliers):
            if x[i, j].solution_value():
                assignments.append({
                    'Dep. Lat': list_data.iloc[i]['Dep. Lat'],
                    'Dep. Lon': list_data.iloc[i]['Dep. Lon'],
                    'Arr. Lat': list_data.iloc[j]['Arr. Lat'],
                    'Arr. Lon': list_data.iloc[j]['Arr. Lon'],
                    'SupID': order_data.iloc[j]['SupID'],
                    'TruckID': truck_data.iloc[i]['TruckID'],
                    'TrailerID': truck_data.iloc[i]['TrailerID'],
                    'AssignTime': get_random_time(datetime.datetime.strptime(list_data.iloc[i]['ReadyDate'], "%Y-%m-%d"))
                })

    return pd.DataFrame(assignments)


def run():
    global truck_data, order_data
    # Inside the loop for each day
    for day in range(30):
        # Calculate the date for the current day
        current_date = datetime.datetime(2023, 11, 24) + datetime.timedelta(days=day)
        current_date_str = current_date.strftime("%Y-%m-%d")

        # Load TruckDatabase and OrderDatabase for the current day
        TruckDatabasejsonfilename = os.path.join(get_truck_dbs_dir(), f'TruckDatabase-{current_date_str}.json')
        truck_data = read_json_to_dataframe(TruckDatabasejsonfilename)
        OrderDatabasejsonfilename = os.path.join(get_order_dbs_dir(), f'OrderDatabase-{current_date_str}.json')
        order_data = read_json_to_dataframe(OrderDatabasejsonfilename)

        list_data = pd.merge(truck_data[['Dep. Lat', 'Dep. Lon']], order_data[['Arr. Lat', 'Arr. Lon', 'ReadyDate']], how='cross')

        print("list data is printing")
        print(list_data)
        print("printed")
        # Calculate distances between vehicles and suppliers
        distance_matrix = [[calculate_distance((list_data.iloc[i]['Dep. Lat'], list_data.iloc[i]['Dep. Lon']),
                                               (list_data.iloc[j]['Arr. Lat'], list_data.iloc[j]['Arr. Lon']))
                            for j in range(len(order_data))] for i in range(len(truck_data))]
        print("distances are printing")
        #print(len(distance_matrix.columns))
        print(len(distance_matrix[0]))
        print(distance_matrix)
        print("distance printed")
        # Solve MILP problem and get Assignments dataframe
        assignments_df = solve_milp(distance_matrix, list_data)

        # Print Assignments dataframe without index numbers
        print(f"Assignments DataFrame for {current_date_str}:")
        print(assignments_df)

        # Save Assignments dataframe as JSON file
        assignments_df.to_json(os.path.join(get_assignment_dbs_dir(), f'Assignments-{current_date_str}.json'),
                               orient='records')
        print(
            f'Assignments for {current_date_str} saved to: {get_assignment_dbs_dir()}Assignments - {current_date_str}.json')


run()  # TODO : Delete
