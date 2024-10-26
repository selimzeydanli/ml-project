import json
import logging
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
from datetime import datetime

# Set up logging configuration
logging.basicConfig(level=logging.INFO)


class Coordinates:
    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon


class DistanceResult:
    def __init__(self, distance_km: Optional[float], service_name: str, error: Optional[str] = None):
        self.distance_km = distance_km
        self.service_name = service_name
        self.error = error


class DistanceCalculator:
    def get_distance(self, start: Coordinates, end: Coordinates) -> DistanceResult:
        try:
            # Placeholder for actual API calculation
            distance_km = 10.0  # This should be replaced with actual distance calculation
            return DistanceResult(distance_km=distance_km, service_name="MockService")
        except Exception as e:
            return DistanceResult(distance_km=None, service_name="MockService", error=str(e))


class DistanceProcessor:
    def __init__(self, input_path: str, output_path: str, calculator: DistanceCalculator):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.calculator = calculator
        self.progress_file = self.output_path.parent / f"{self.output_path.stem}_progress.json"
        self.logger = logging.getLogger(__name__)

    def load_progress(self) -> Dict:
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_progress(self, data: Dict):
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def process_distances(self):
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            progress = self.load_progress()
            total_items = len(data)

            with tqdm(total=total_items, initial=len(progress)) as pbar:
                for item in data:
                    item_id = str(item.get('id', ''))

                    # Skip if this item is already processed
                    if item_id in progress:
                        continue

                    # Check for valid coordinates
                    try:
                        start = Coordinates(
                            lat=float(item.get('Supplier_Latitude')),
                            lon=float(item.get('Supplier_Longitude'))
                        )
                        end = Coordinates(
                            lat=float(item.get('Port_Latitude')),
                            lon=float(item.get('Port_Longitude'))
                        )
                    except (TypeError, ValueError):
                        self.logger.error(f"Invalid coordinates for item {item_id}, skipping.")
                        pbar.update(1)
                        continue

                    result = self.calculator.get_distance(start, end)

                    if result.distance_km is not None:
                        item['distance_km'] = result.distance_km
                        item['distance_source'] = result.service_name
                        progress[item_id] = {
                            'distance_km': result.distance_km,
                            'distance_source': result.service_name,
                            'timestamp': datetime.now().isoformat()
                        }
                        self.save_progress(progress)
                    else:
                        self.logger.error(f"Failed to calculate distance for item {item_id}: {result.error}")

                    pbar.update(1)

            # Save final results
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Results saved to {self.output_path}")

        except Exception as e:
            self.logger.error(f"Error processing distances: {str(e)}")
            raise


def main():
    # Define paths
    input_path = "Deneme.json"
    # Explicit path for output
    output_path = Path("C:/Users/Selim/Desktop/Updated.json")

    calculator = DistanceCalculator()
    processor = DistanceProcessor(input_path=input_path, output_path=output_path, calculator=calculator)

    logging.info("Starting distance calculations...")
    processor.process_distances()
    logging.info("Distance calculations completed.")


if __name__ == "__main__":
    main()
