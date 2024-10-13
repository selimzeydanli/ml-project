import json
import requests


def load_coordinates(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['spot1'], data['spot2']


def get_distance(api_key, start_lat, start_lon, end_lat, end_lon):
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{start_lat},{start_lon}:{end_lat},{end_lon}/json"
    params = {
        'key': api_key,
        'routeType': 'fastest'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data['routes'][0]['summary']['lengthInMeters']
    else:
        return None


def main():
    # Replace 'coordinates.json' with your JSON file path
    #spot1, spot2 = load_coordinates('coordinates.json')

    # Replace 'YOUR_API_KEY' with your actual TomTom API key
    api_key = '4iQviYFRJ2pHD7GgSvzLsaFUvyejWWav'

    #distance = get_distance(api_key, spot1['lat'], spot1['lon'], spot2['lat'], spot2['lon'])
    #d28.9783589!2d41.0082376!1
    distance = get_distance(api_key, 41.0082376, 28.9783589, 38.439983, 27.14908)

    if distance:
        print(f"The real road distance between the two spots is {distance} meters.")
    else:
        print("Failed to retrieve the distance. Please check your API key and coordinates.")


if __name__ == "__main__":
    main()