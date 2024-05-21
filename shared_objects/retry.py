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


@retry(tries=5, delay=5, backoff=2)
def retry_with_timeout(func, timeout, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout)
            return result
        except TimeoutError:
            bt.logging.error(f"retry_with_timeout: {func.__name__} execution exceeded {timeout} seconds.")
            future.cancel()
            raise TimeoutError(f"retry_with_timeout: {func.__name__} execution exceeded the timeout limit.")
        except Exception as e:
            bt.logging.error(f"retry_with_timeout: {func.__name__} Unexpected exception {type(e).__name__} occurred: {e}")
            future.cancel()
            raise e  # Re-raise the exception to handle it in the retry logic.



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