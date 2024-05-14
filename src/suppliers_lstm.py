import random
import json


def generate_unique_values(start, end, count):
    """Generate a list of unique random values within the specified range."""
    values = set()
    while len(values) < count:
        values.add(round(random.uniform(start, end), 2))
    return list(values)


def generate_suppliers_data(num_items):
    """Generate a dictionary with 1200 items and 3 keys: SupID, Arr. Lat, and Arr. Lon."""
    suppliers_data = {}
    sup_ids = generate_unique_values(127654, 458534, num_items)
    arr_lats = generate_unique_values(37.2, 39.5, num_items)
    arr_lons = generate_unique_values(27.1, 44.5, num_items)

    for i in range(num_items):
        suppliers_data[i] = {
            'SupID': sup_ids[i],
            'Arr. Lat': arr_lats[i],
            'Arr. Lon': arr_lons[i]
        }

    return suppliers_data


# Generate the dictionary with 1200 items
suppliers_dict = generate_suppliers_data(1200)

# Save the dictionary as a JSON file
file_path = r'C:\Users\Selim\Desktop\ml-project\input\suppliers_lstm.json'
with open(file_path, 'w') as file:
    json.dump(suppliers_dict, file)

print("JSON file saved successfully at:", file_path)
