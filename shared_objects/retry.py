import threading
import bittensor as bt
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
from functools import wraps

def retry(tries=5, delay=5, backoff=1):
    """
    Retry decorator with exponential backoff, works for all exceptions.

    Parameters:
    - tries: number of times to try (not retry) before giving up.
    - delay: initial delay between retries in seconds.
    - backoff: backoff multiplier e.g. value of 2 will double the delay each retry.

    Usage:
    @retry(tries=5, delay=5, backoff=2)
    def my_func():
        pass
    """
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    bt.logging.error(f"Error: {str(e)}, Retrying in {mdelay} seconds...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)  # Last attempt
        return f_retry
    return deco_retry


class TimeoutException(Exception):
    pass


def timeout(timeout_duration):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout_duration)

            if thread.is_alive():
                raise TimeoutException(f"Function call timed out after {timeout_duration} seconds.")

            if exception[0]:
                raise exception[0]

            return result[0]

        return wrapper

    return decorator


def retry_with_timeout(retries=4, initial_delay=5, backoff_factor=2, timeout_duration=30):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(1, retries + 1):
                try:
                    result = timeout(timeout_duration)(func)(*args, **kwargs)
                    return result
                except TimeoutException:
                    print(f"Attempt {attempt}: perf ledger refresh_price timed out after {timeout_duration} seconds.")
                except Exception as e:
                    print(f"Attempt {attempt}: perf ledger refresh_price failed with error: {e}. Retrying in {delay} seconds.")
                time.sleep(delay)
                delay *= backoff_factor
            raise Exception(f"Function failed after {retries} retries.")

        return wrapper

    return decorator


def periodic_heartbeat(interval=5, message="Heartbeat..."):
    def decorator(func):
        def wrapped(*args, **kwargs):
            def heartbeat():
                while not stop_event.is_set():
                    print(message)
                    time.sleep(interval)

            stop_event = threading.Event()
            heartbeat_thread = threading.Thread(target=heartbeat)
            heartbeat_thread.start()

            try:
                return func(*args, **kwargs)
            finally:
                stop_event.set()
                heartbeat_thread.join()

        return wrapped
    return decorator