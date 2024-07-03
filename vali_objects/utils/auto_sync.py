import gzip
import io
import json
import traceback
import zipfile

import requests

from time_util.time_util import TimeUtil
from vali_objects.utils.validator_sync_base import ValidatorSyncBase
import bittensor as bt



class PositionSyncer(ValidatorSyncBase):
    def __init__(self, shutdown_dict=None, signal_sync_lock=None, signal_sync_condition=None, n_orders_being_processed=None):
        super().__init__(shutdown_dict, signal_sync_lock, signal_sync_condition, n_orders_being_processed)

    def read_validator_checkpoint_from_gcloud_zip(url):
        # URL of the zip file
        url = "https://storage.googleapis.com/validator_checkpoint/validator_checkpoint.json.gz"
        try:
            # Send HTTP GET request to the URL
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses

            # Read the content of the gz file from the response
            with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz_file:
                # Decode the gzip content to a string
                json_bytes = gz_file.read()
                json_str = json_bytes.decode('utf-8')

                # Load JSON data from the string
                json_data = json.loads(json_str)
                return json_data

        except requests.HTTPError as e:
            bt.logging.error(f"HTTP Error: {e}")
        except zipfile.BadZipFile:
            bt.logging.error("The downloaded file is not a zip file or it is corrupted.")
        except json.JSONDecodeError:
            bt.logging.error("Error decoding JSON from the file.")
        except Exception as e:
            bt.logging.error(f"An unexpected error occurred: {e}")
        return None

    def sync_positions_with_cooldown(self, auto_sync_enabled:bool):
        # Check if the time is right to sync signals
        if not auto_sync_enabled:
            return
        now_ms = TimeUtil.now_in_millis()
        # Already performed a sync recently
        if now_ms - self.last_signal_sync_time_ms < 1000 * 60 * 30:
            return

        # Check if we are between 6:09 AM and 6:19 AM UTC
        datetime_now = TimeUtil.generate_start_timestamp(0)  # UTC
        if not (datetime_now.hour == 6 and (8 < datetime_now.minute < 20)):
            return

        with self.signal_sync_lock:
            while self.n_orders_being_processed[0] > 0:
                self.signal_sync_condition.wait()
            # Ready to perform in-flight refueling
            try:
                candidate_data = self.read_validator_checkpoint_from_gcloud_zip()
                if not candidate_data:
                    bt.logging.error("Unable to read validator checkpoint file. Sync canceled")
                else:
                    self.sync_positions(False, candidate_data=candidate_data)
            except Exception as e:
                bt.logging.error(f"Error syncing positions: {e}")
                bt.logging.error(traceback.format_exc())

        self.last_signal_sync_time_ms = TimeUtil.now_in_millis()


if __name__ == "__main__":
    bt.logging.enable_default()
    position_syncer = PositionSyncer()
    candidate_data = position_syncer.read_validator_checkpoint_from_gcloud_zip()
    position_syncer.sync_positions(False, candidate_data=candidate_data)
