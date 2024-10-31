import json
import logging
import random
from typing import Optional, Dict, List
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Setting up the logging configuration
logging.basicConfig(level=logging.INFO)


@dataclass
class Coordinates:
    lat: float
    lon: float


@dataclass
class DistanceResult:
    distance_km: Optional[float]
    service_name: str
    error: Optional[str] = None


class MultiServiceDistanceCalculator:
    def __init__(self, api_keys):
        self.api_keys = api_keys

    def get_distance(self, start: Coordinates, end: Coordinates) -> DistanceResult:
        try:
            # Simulate API distance calculation (e.g., using Haversine formula or mock distance)
            distance_km = random.uniform(5, 1000)  # Placeholder for actual calculation
            return DistanceResult(distance_km=distance_km, service_name="MockService")
        except Exception as e:
            return DistanceResult(distance_km=None, service_name="MockService", error=str(e))


class DistanceProcessor:
    def __init__(self, input_path: str, output_path: str, calculator: MultiServiceDistanceCalculator):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.calculator = calculator
        self.progress_file = self.output_path.parent / f"{self.output_path.stem}_progress.json"
        self.logger = logging.getLogger(__name__)
        self.save_interval = 500  # Save every 500 calculations
        self.processed_since_last_save = 0

    def load_progress(self) -> Dict:
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_progress(self, data: Dict):
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        self.logger.info(f"Progress saved to {self.progress_file}")

    def save_current_results(self, data: List[Dict]):
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        self.logger.info(f"Results saved to {self.output_path}")
        self.processed_since_last_save = 0

    def process_distances(self):
        try:
            if not self.input_path.exists():
                raise FileNotFoundError(f"Input file not found: {self.input_path}")

            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            progress = self.load_progress()
            total_items = len(data)

            self.logger.info(f"Processing {total_items} items total. {len(progress)} items already processed.")

            with tqdm(total=total_items, initial=len(progress)) as pbar:
                for item in data:
                    item_id = str(item.get('id', ''))

                    if item_id in progress:
                        pbar.update(1)
                        continue

                    try:
                        start = Coordinates(
                            lat=float(item.get('Supplier_Latitude')),
                            lon=float(item.get('Supplier_Longitude'))
                        )
                        end = Coordinates(
                            lat=float(item.get('Port_Latitude')),
                            lon=float(item.get('Port_Longitude'))
                        )
                    except (TypeError, ValueError) as e:
                        self.logger.error(f"Invalid coordinates for item {item_id}: {str(e)}")
                        pbar.update(1)
                        continue

                    result = self.calculator.get_distance(start, end)

                    if result.distance_km is not None:
                        # Adding the distance and keeping order
                        item['Distance(km)'] = result.distance_km  # Add Distance(km) key
                        item['distance_source'] = result.service_name  # Optional, depending on your needs

                        # Reorder keys to ensure "Distance(km)" comes before "Duration_To_Port(h)"
                        if 'Duration_To_Port(h)' in item:
                            new_item = {
                                'Order_ID': item['Order_ID'],
                                'Order_Date': item['Order_Date'],
                                'Ready_Date_Time': item['Ready_Date_Time'],
                                'Sup_ID': item['Sup_ID'],
                                'Trailer_Type': item['Trailer_Type'],
                                'Supplier_Latitude': item['Supplier_Latitude'],
                                'Supplier_Longitude': item['Supplier_Longitude'],
                                'Loading_Start_Time': item['Loading_Start_Time'],
                                'Loading_End_Time': item['Loading_End_Time'],
                                'Duration_Loading(h)': item['Duration_Loading(h)'],
                                'Departure_Time': item['Departure_Time'],
                                'Route_Mode': item['Route_Mode'],
                                'PORT_NAME': item['PORT_NAME'],
                                'Port_Latitude': item['Port_Latitude'],
                                'Port_Longitude': item['Port_Longitude'],
                                'Port_Arrival_Date': item['Port_Arrival_Date'],
                                'Distance(km)': item['Distance(km)'],  # Positioning Distance before Duration
                                'Duration_To_Port(h)': item['Duration_To_Port(h)']  # Retaining original duration
                            }
                            item.clear()  # Clear the old item data
                            item.update(new_item)  # Update with new ordered data

                        progress[item_id] = {
                            'distance_km': result.distance_km,
                            'distance_source': result.service_name,
                            'timestamp': datetime.now().isoformat()
                        }

                        self.processed_since_last_save += 1

                        if self.processed_since_last_save >= self.save_interval:
                            self.save_progress(progress)
                            self.save_current_results(data)
                            self.logger.info(f"Saved progress after processing {self.processed_since_last_save} items")
                    else:
                        self.logger.error(f"Failed to calculate distance for item {item_id}: {result.error}")

                    pbar.update(1)

            self.save_progress(progress)
            self.save_current_results(data)
            self.logger.info("Processing completed successfully")

        except Exception as e:
            self.logger.error(f"Error processing distances: {str(e)}")
            if 'progress' in locals():
                self.save_progress(progress)
            if 'data' in locals():
                self.save_current_results(data)
            raise


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('distance_calculator.log')
        ]
    )

    logger = logging.getLogger(__name__)

    api_keys = {
        'tomtom': '4iQviYFRJ2pHD7GgSvzLsaFUvyejWWav',
        'graphhopper': 'YOUR_GRAPHHOPPER_KEY',
        'mapbox': 'YOUR_MAPBOX_KEY',
        'here': 'YOUR_HERE_KEY',
        'google': 'YOUR_GOOGLE_KEY'
    }

    input_path = "Deneme.json"
    output_path = Path("C:/Users/Selim/Desktop/Updated.json")

    try:
        calculator = MultiServiceDistanceCalculator(api_keys)
        processor = DistanceProcessor(input_path=input_path, output_path=output_path, calculator=calculator)

        logger.info("Starting distance calculations...")
        processor.process_distances()
        logger.info("Distance calculations completed successfully")

    except Exception as e:
        logger.error(f"Program failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
