import pandas as pd
import json
import os

def excel_to_json(excel_file_path, json_file_name):
    """Converts an Excel file to a JSON file, handling timestamps and formatting specified fields.

    Args:
        excel_file_path (str): Path to the Excel file.
        json_file_name (str): Path to the output JSON file.
    """

    # Read the Excel file into a Pandas DataFrame
    df = pd.read_excel(excel_file_path)

    # Handle timestamps if present
    for col in df.columns:
        if pd.api.types.is_datetime64_dtype(df[col]):
            df[col] = df[col].dt.strftime('%Y-%m-%dT%H:%M:%S')  # Format to ISO 8601

    # Format specific columns to 2 decimal places
    df['Supplier_Latitude'] = df['Supplier_Latitude'].round(2)
    df['Supplier_Longitude'] = df['Supplier_Longitude'].round(2)
    df['Port_Latitude'] = df['Port_Latitude'].round(2)
    df['Port_Longitude'] = df['Port_Longitude'].round(2)
    df['Duration_Loading(h)'] = df['Duration_Loading(h)'].round(2)
    df['Duration_To_Port(h)'] = df['Duration_To_Port(h)'].round(2)

    # Convert the DataFrame to a list of dictionaries
    data = df.to_dict(orient='records')

    # Print the JSON data to the console for inspection
    print(json.dumps(data, indent=4))

    # Write the JSON data to a file
    with open(json_file_name, 'w') as f:
        json.dump(data, f, indent=4)

# Specify the file paths
excel_file_path = "C:\\Users\\Selim\\Desktop\\Prepared Data.xlsx"
json_file_name = "C:\\Users\\Selim\\Desktop\\Deneme.json"

# Convert the Excel file to JSON
excel_to_json(excel_file_path, json_file_name)