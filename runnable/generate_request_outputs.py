import time
import traceback

from runnable.generate_request_core import generate_request_core
from runnable.generate_request_minerstatistics import generate_request_minerstatistics

from vali_objects.exceptions.corrupt_data_exception import ValiBkpCorruptDataException
from vali_objects.utils.logger_utils import LoggerUtils
from time_util.time_util import TimeUtil
from json import JSONDecodeError

if __name__ == "__main__":
    logger = LoggerUtils.init_logger("generate_request_outputs")
    last_write_time = time.time()
    n_updates = 0
    try:
        while True:
            current_time = time.time()
            write_output = False
            # Check if it's time to write the legacy output
            if current_time - last_write_time >= 15:
                write_output = True

            current_time_ms = TimeUtil.now_in_millis()
            if write_output:
                # Generate the request outputs
                generate_request_core(time_now=current_time_ms)
                generate_request_minerstatistics(time_now=current_time_ms)

            if write_output:
                last_validator_checkpoint_time = current_time

            # Log completion duration
            if write_output:
                n_updates += 1
                if n_updates % 10 == 0:
                    logger.info("Completed writing outputs in " + str(time.time() - current_time) + " seconds. n_updates: " + str(n_updates))
    except (JSONDecodeError, ValiBkpCorruptDataException) as e:
        logger.error("error occurred trying to decode position json. Probably being written to simultaneously.")
        logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        # traceback
        logger.error(traceback.format_exc())
        time.sleep(10)
    # Sleep for a short time to prevent tight looping, adjust as necessary
    time.sleep(1)
