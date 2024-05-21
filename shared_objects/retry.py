import threading
import time
from functools import wraps
import bittensor as bt


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