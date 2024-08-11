import os
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta
import pandas as pd
import random
import numpy as np
from ortools.linear_solver import pywraplp

from src.transaction_lstm import haversine
from src.utils import read_json_to_dataframe, get_order_dbs_lstm_dir, get_truck_dbs_lstm_dir


# ... [Keep the existing import statements and utility functions] ...

# truck assignmnet with MILP
def solve_assignment(order_df, truck_df):
    solver = pywraplp.Solver.CreateSolver('SCIP')

    num_orders = len(order_df)
    num_trucks = len(truck_df)

    # Create binary variables for assignment
    x = {}
    for i in range(num_orders):
        for j in range(num_trucks):
            x[i, j] = solver.IntVar(0, 1, f'x[{i},{j}]')

    # Constraint: Each order is assigned to at most one truck
    for i in range(num_orders):
        solver.Add(solver.Sum([x[i, j] for j in range(num_trucks)]) <= 1)

    # Constraint: Each truck is assigned to at most one order
    for j in range(num_trucks):
        solver.Add(solver.Sum([x[i, j] for i in range(num_orders)]) <= 1)

    # Objective: Minimize total distance
    objective = solver.Objective()
    for i in range(num_orders):
        for j in range(num_trucks):
            distance = haversine(
                order_df.iloc[i]['Arr. Lat'], order_df.iloc[i]['Arr. Lon'],
                truck_df.iloc[j]['Dep. Lat'], truck_df.iloc[j]['Dep. Lon']
            )
            objective.SetCoefficient(x[i, j], distance)

    objective.SetMinimization()

    # Solve
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        assignments = []
        for i in range(num_orders):
            for j in range(num_trucks):
                if x[i, j].solution_value() > 0.5:
                    assignments.append((i, j))
        return assignments
    else:
        return None


# ... [Keep other utility functions] ...

orderDate = datetime(2023, 1, 3)
endLoopDate = datetime(2023, 12, 31)
job_id = 1

while orderDate <= endLoopDate:
    orderDateStr = orderDate.strftime("%Y-%m-%d")
    truckDateFirst = orderDate - timedelta(days=1)
    truckDateFirstStr = truckDateFirst.strftime("%Y-%m-%d")
    truckDateSecond = orderDate - timedelta(days=2)
    truckDateSecondStr = truckDateSecond.strftime("%Y-%m-%d")


    order_df = read_json_to_dataframe(os.path.join(get_order_dbs_lstm_dir(), f"OrderDatabase_lstm-{orderDateStr}.json"))
    truck_1_df = read_json_to_dataframe(
        os.path.join(get_truck_dbs_lstm_dir(), f"TruckDatabase_lstm-{truckDateFirstStr}.json"))
    truck_2_df = read_json_to_dataframe(
        os.path.join(get_truck_dbs_lstm_dir(), f"TruckDatabase_lstm-{truckDateSecondStr}.json"))
    truck_df = pd.concat([truck_1_df, truck_2_df]).reset_index(drop=True)

    # Solve the assignment problem
    assignments = solve_assignment(order_df, truck_df)

    job_entries = []
    if assignments:
        for order_idx, truck_idx in assignments:
            order = order_df.iloc[order_idx]
            truck = truck_df.iloc[truck_idx]

            # ... [Calculate all the necessary fields as in the original code] ...

            job_entry = {
                "Job_ID": job_id,
                "Order_ID": order['OrderID'],
                "Order_Date": orderDateStr,
                # ... [Fill in all the fields as in the original code] ...
            }

            job_entries.append(job_entry)
            job_id += 1

    # Handle unassigned orders
    assigned_orders = set([a[0] for a in assignments]) if assignments else set()
    for idx, order in order_df.iterrows():
        if idx not in assigned_orders:
            job_entry = {
                "Job_ID": job_id,
                "Order_ID": order['OrderID'],
                "Order_Date": orderDateStr,
                # ... [Fill with 'NaN' values as in the original code for unassigned orders] ...
            }
            job_entries.append(job_entry)
            job_id += 1

    checkout_df = pd.DataFrame(job_entries)

    # ... [Save the results and continue with the next date] ...

    orderDate = orderDate + timedelta(days=1)

# ... [Keep the display settings at the end] ...