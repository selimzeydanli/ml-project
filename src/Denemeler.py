import json
import requests
import os
import time


def get_driving_distance(supplier_lat, supplier_lon, port_lat, port_lon):
    """Fetch the driving distance between supplier and port using the TomTom API."""
    api_key = '4iQviYFRJ2pHD7GgSvzLsaFUvyejWWav'  # Replace with your actual API key
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{supplier_lat},{supplier_lon}:{port_lat},{port_lon}/json"
    params = {
        'key': api_key,
        'routeType': 'fastest',
        'travelMode': 'car'
    }

    try:
        response = requests.get(url, params=params, timeout=10)  # 10 seconds timeout
        response.raise_for_status()  # Raise an error for bad responses
        distance_data = response.json()
        distance = distance_data['routes'][0]['summary']['lengthInMeters']  # Get distance in meters
        return distance
    except requests.exceptions.RequestException as e:
        print(f"Error fetching driving distance from TomTom: {e}")
        return None


def process_records(data):
    """Process the records and calculate distances."""
    distances = []
    for idx, record in enumerate(data):
        supplier_lat = record['Supplier_Latitude']
        supplier_lon = record['Supplier_Longitude']
        port_lat = record['Port_Latitude']
        port_lon = record['Port_Longitude']

        distance = get_driving_distance(supplier_lat, supplier_lon, port_lat, port_lon)
        if distance is not None:
            distances.append(distance)
            record['distance'] = distance  # Add distance to the record

        # Save periodically after every 100 calculations to avoid data loss
        if (idx + 1) % 100 == 0:
            save_to_file(data, 'C:\\Users\\Selim\\Desktop\\Updated.json')  # Save progress

    # Final save after processing all records
    save_to_file(data, 'C:\\Users\\Selim\\Desktop\\Updated.json')

    return distances


def save_to_file(data, file_path):
    """Save the updated data to a JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Data saved to {file_path}")
    except IOError as e:
        print(f"Error saving data to file: {e}")


def main():
    """Main function to load data, process it, and save results."""
    # Load your data here
    file_path = r'C:\Users\Selim\Desktop\Deneme.json'  # Absolute path to the JSON file
    try:
        with open(file_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    except json.JSONDecodeError:
        print("Error decoding JSON. Please check the file format.")
        return

    # Process records
    distances = process_records(data)

    print("Distance calculations complete.")
    print(f"Total distances calculated: {len(distances)}")


if __name__ == "__main__":
    main()
