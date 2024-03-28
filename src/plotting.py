# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:43:14 2024

@author: Selim
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import nbformat
import os
import shutil
from nbformat.v4 import new_notebook, new_code_cell
from datetime import datetime, timedelta

from src.utils import get_plotting_dir, empty_folder, get_assignment_dbs_dir, read_json_to_dataframe, \
    get_transaction_dbs_dir

empty_folder(get_plotting_dir())

# Create a function to generate and save the plot for a given day
def generate_and_save_plot(day):
    # Calculate the date for the current day
    current_date = datetime(2023, 11, 24) + timedelta(days=day)
    current_date_str = current_date.strftime("%Y-%m-%d")

    # Read data from Assignments.json
    try:
        assignments = read_json_to_dataframe(os.path.join(get_assignment_dbs_dir(), f'Assignments-{current_date_str}.json'))
    except ValueError as e:
        print(f"Error reading JSON file: {e}")
        raise

    # Define file path for TransactionDatabase.json
    transaction_path = os.path.join(get_transaction_dbs_dir(), f'TransactionDatabase-{current_date.strftime("%Y-%m-%d")}.json')

    # Read data from TransactionDatabase.json
    try:
        transactions = read_json_to_dataframe(transaction_path)
    except ValueError as e:
        print(f"Error reading JSON file: {e}")
        raise

    # print transactions dataframe
    print(f"Transactions DataFrame for {current_date.strftime('%Y-%m-%d')}:")
    print(transactions)

    # Extract coordinates for plotting
    dep_latitudes = assignments['Dep. Lat']
    dep_longitudes = assignments['Dep. Lon']
    arr_latitudes = assignments['Arr. Lat']
    arr_longitudes = assignments['Arr. Lon']

    # Plot the diagram with lines
    plt.figure(figsize=(10, 8))

    # Plot departure locations (vehicles)
    plt.scatter(dep_latitudes, dep_longitudes, color='blue', label='Departure (Vehicle)')

    # Plot arrival locations (suppliers)
    plt.scatter(arr_latitudes, arr_longitudes, color='red', label='Arrival (Supplier)')

    # Draw lines between each vehicle and supplier
    for i in range(len(dep_latitudes)):
        plt.plot([dep_latitudes[i], arr_latitudes[i]], [dep_longitudes[i], arr_longitudes[i]], color='black', linestyle='-', linewidth=0.5)

    plt.title('Vehicle Assignments with Lines')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.legend()
    plt.grid(True)

    # Create notebook structure
    nb = new_notebook()

    # Add code cells
    nb.cells.append(new_code_cell("%matplotlib inline"))
    nb.cells.append(new_code_cell(f"plt.scatter({dep_latitudes}, {dep_longitudes}, color='blue', label='Departure (Vehicle)')"))
    nb.cells.append(new_code_cell(f"plt.scatter({arr_latitudes}, {arr_longitudes}, color='red', label='Arrival (Supplier)')"))
    nb.cells.append(new_code_cell("for i in range(len(dep_latitudes)): plt.plot([dep_latitudes[i], arr_latitudes[i]], [dep_longitudes[i], arr_longitudes[i]], color='black', linestyle='-', linewidth=0.5)"))
    nb.cells.append(new_code_cell("plt.title('Vehicle Assignments with Lines')"))
    nb.cells.append(new_code_cell("plt.xlabel('Latitude')"))
    nb.cells.append(new_code_cell("plt.ylabel('Longitude')"))
    nb.cells.append(new_code_cell("plt.legend()"))
    nb.cells.append(new_code_cell("plt.grid(True)"))
    nb.cells.append(new_code_cell("plt.show()"))

    # Save the notebook to a file
    plot_ipynb_path = os.path.join(get_plotting_dir(), f'Plotting-{current_date.strftime("%Y-%m-%d")}.ipynb')
    with open(plot_ipynb_path, 'w') as f:
        nbformat.write(nb, f)

    print(f'Plot for {current_date.strftime("%Y-%m-%d")} saved to: {plot_ipynb_path}')

    # Save the plot as a PNG file
    plot_png_path = os.path.join(get_plotting_dir(), f'Plotting-{current_date.strftime("%Y-%m-%d")}.png')
    plt.savefig(plot_png_path)

    print(f'Plot for {current_date.strftime("%Y-%m-%d")} saved as: {plot_png_path}')

    # Close the plot to free up memory
    plt.close()


def run():
    # Create plots for 30 consecutive days
    for day in range(30):
        generate_and_save_plot(day)
