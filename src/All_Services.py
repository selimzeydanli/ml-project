import json
import logging
import random
from typing import Optional, Dict, List, Any, TypedDict
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import sys
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import json
import requests
import os
import time
from typing import List, Dict, Optional, Union
from datetime import datetime
import logging
from pathlib import Path


# Type definitions for better type safety
class ProgressEntry(TypedDict):
    distance_km: float
    distance_source: str
    timestamp: str


class ServiceStatus(Enum):
    ACTIVE = "active"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"


@dataclass
class Coordinates:
    lat: float
    lon: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional['Coordinates']:
        try:
            return cls(
                lat=float(data.get('Latitude', 0)),
                lon=float(data.get('Longitude', 0))
            )
        except (TypeError, ValueError):
            return None

    def is_valid(self) -> bool:
        return -90 <= self.lat <= 90 and -180 <= self.lon <= 180


@dataclass
class DistanceResult:
    distance_km: Optional[float]
    service_name: str
    error: Optional[str] = None
    retry_after: Optional[int] = None


class DistanceService:
    def __init__(self, name: str, api_key: str):
        self.name = name
        self.api_key = api_key
        self.status = ServiceStatus.ACTIVE
        self.retry_after = 0
        self.error_count = 0
        self.MAX_ERRORS = 3
        self.BACKOFF_TIME = 60  # seconds

    def is_available(self) -> bool:
        if self.status == ServiceStatus.FAILED:
            return False
        if self.status == ServiceStatus.RATE_LIMITED:
            return time.time() >= self.retry_after
        return True

    def mark_error(self, retry_after: Optional[int] = None):
        self.error_count += 1
        if retry_after:
            self.status = ServiceStatus.RATE_LIMITED
            self.retry_after = time.time() + retry_after
        elif self.error_count >= self.MAX_ERRORS:
            self.status = ServiceStatus.FAILED
        else:
            self.retry_after = time.time() + self.BACKOFF_TIME

    def reset_errors(self):
        self.error_count = 0
        self.status = ServiceStatus.ACTIVE


class MultiServiceDistanceCalculator:
    def __init__(self, api_keys: Dict[str, str]):
        # Create an OrderedDict with the desired service order
        ordered_services = OrderedDict()

        # Define the preferred order of services
        preferred_order = ['tomtom', 'here', 'graphhopper', 'mapbox', 'google']

        # First, add services in the preferred order if they exist in api_keys
        for service_name in preferred_order:
            if service_name in api_keys:
                ordered_services[service_name] = api_keys[service_name]

        # Then add any remaining services that weren't in the preferred order
        for service_name, api_key in api_keys.items():
            if service_name not in ordered_services:
                ordered_services[service_name] = api_key

        self.services = {
            name: DistanceService(name, key)
            for name, key in ordered_services.items()
        }
        self.logger = logging.getLogger(__name__)
        self.service_names = list(self.services.keys())
        self.current_service_index = 0

    def get_next_available_service(self) -> Optional[str]:
        start_index = self.current_service_index
        while True:
            service_name = self.service_names[self.current_service_index]
            service = self.services[service_name]

            # Move to next service for next time
            self.current_service_index = (self.current_service_index + 1) % len(self.service_names)

            if service.is_available():
                return service_name

            # If we've checked all services and come back to where we started
            if self.current_service_index == start_index:
                return None

    def get_distance(self, start: Coordinates, end: Coordinates) -> DistanceResult:
        if not (start.is_valid() and end.is_valid()):
            return DistanceResult(None, "validation", "Invalid coordinates")

        service_name = self.get_next_available_service()
        if not service_name:
            return DistanceResult(None, "system", "No available services")

        service = self.services[service_name]
        self.base_url = "https://api.tomtom.com/routing/1/calculateRoute"
        url = f"{self.base_url}/{start.lat},{start.lon}:{end.lat},{end.lon}/json"
        params = {
            'key': service.api_key,
            'routeType': 'fastest',
            'travelMode': 'truck',  # Changed to truck for more accurate commercial routing
            'avoid': 'unpavedRoads',
            'vehicleMaxSpeed': 90  # Typical truck speed limit in km/h
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            distance_data = response.json()
            distance_km = distance_data['routes'][0]['summary']['lengthInMeters']/1000.0
            return DistanceResult(distance_km=distance_km, service_name=service_name)

        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {str(e)}")
            return DistanceResult(None, service_name, str(e))
        except (KeyError, IndexError) as e:
            logging.error(f"Error parsing API response: {str(e)}")
            return DistanceResult(None, service_name, str(e))


class DistanceProcessor:
    def __init__(self, input_path: str, output_path: str, calculator: MultiServiceDistanceCalculator):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.calculator = calculator
        self.progress_file = self.output_path.parent / f"{self.output_path.stem}_progress.json"
        self.logger = logging.getLogger(__name__)
        self.save_interval = 10
        self._processed_count = 0
        self.required_fields = {
            'Order_ID', 'Order_Date', 'Ready_Date_Time', 'Sup_ID', 'Trailer_Type',
            'Supplier_Latitude', 'Supplier_Longitude', 'Loading_Start_Time', 'Loading_End_Time',
            'Duration_Loading(h)', 'Departure_Time', 'Route_Mode', 'PORT_NAME',
            'Port_Latitude', 'Port_Longitude', 'Port_Arrival_Date', 'Duration_To_Port(h)'
        }

    def validate_input_data(self, data: List[Dict[str, Any]]) -> bool:
        if not isinstance(data, list):
            self.logger.error("Input data must be a list of dictionaries")
            return False

        for item in data:
            missing_fields = self.required_fields - set(item.keys())
            if missing_fields:
                self.logger.error(f"Missing required fields: {missing_fields}")
                return False

        return True

    def load_progress(self) -> Dict[str, ProgressEntry]:
        return {}
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.logger.warning("Progress file corrupted, starting fresh")
                return {}
        return {}

    def save_progress(self, data: Dict[str, ProgressEntry]):
        temp_file = self.progress_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.progress_file)
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def save_current_results(self, data: List[Dict[str, Any]]):
        temp_file = self.output_path.with_suffix('.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.output_path)
            self._processed_count = 0
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def process_item(self, item: Dict[str, Any], progress: Dict[str, ProgressEntry]) -> bool:
        item_id = str(item.get('Order_ID', ''))
        if 'timestamp' in item:
            self.logger.warning("Skipping Item")
            return True
        if not item_id:
            self.logger.error("Item missing Order_ID")
            return False

        if item_id in progress:
            return True

        try:
            start = Coordinates(
                lat=float(item['Supplier_Latitude']),
                lon=float(item['Supplier_Longitude'])
            )
            end = Coordinates(
                lat=float(item['Port_Latitude']),
                lon=float(item['Port_Longitude'])
            )
        except (TypeError, ValueError) as e:
            self.logger.error(f"Invalid coordinates for item {item_id}: {e}")
            return False

        result = self.calculator.get_distance(start, end)

        if result.distance_km is not None:
            item['Distance(km)'] = result.distance_km
            item['timestamp'] = datetime.now().isoformat()
            item['distance_source'] = result.service_name

            progress[item_id] = {
                'distance_km': result.distance_km,
                'distance_source': result.service_name,
                'timestamp': datetime.now().isoformat()
            }

            return True
        else:
            self.logger.error(f"Failed to calculate distance for item {item_id}: {result.error}")
            return False

    def process_distances(self):
        try:
            if not self.input_path.exists():
                raise FileNotFoundError(f"Input file not found: {self.input_path}")

            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not self.validate_input_data(data):
                raise ValueError("Invalid input data format")


            progress = self.load_progress()
            total_items = len(data)
            processed_count = len(progress)

            self.logger.info(f"Processing {total_items} items. {processed_count} already processed.")

            with tqdm(total=total_items, initial=processed_count) as pbar:
                for item in data:
                    if self.process_item(item, progress):
                        self._processed_count += 1

                        if self._processed_count >= self.save_interval:
                            self.save_progress(progress)
                            self.save_current_results(data)

                    pbar.update(1)

            self.save_progress(progress)
            self.save_current_results(data)
            self.logger.info("Processing completed successfully")

        except Exception as e:
            self.logger.error(f"Error processing distances: {e}")
            if 'progress' in locals():
                self.save_progress(progress)
            if 'data' in locals():
                self.save_current_results(data)
            raise


def setup_logging(log_file: str = 'distance_calculator.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    api_keys = {
        'tomtom': '4iQviYFRJ2pHD7GgSvzLsaFUvyejWWav',
        #'here': '1vlQX4aF5gJIFiRNBpD4USymG9rU_FWIc6Vdc14amWM',
        #'graphhopper': 'YOUR_GRAPHHOPPER_KEY',
        #'mapbox': 'YOUR_MAPBOX_KEY',
        #'google': 'YOUR_GOOGLE_KEY'
    }

    input_path = Path("C:/Users/Selim/Desktop/Deneme.json")
    output_path = Path("C:/Users/Selim/Desktop/Deneme.json")

    try:
        calculator = MultiServiceDistanceCalculator(api_keys)
        processor = DistanceProcessor(input_path=input_path, output_path=output_path, calculator=calculator)

        logger.info("Starting distance calculations...")
        processor.process_distances()
        logger.info("Distance calculations completed successfully")

    except Exception as e:
        logger.error(f"Program failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()