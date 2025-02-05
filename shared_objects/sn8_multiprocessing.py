import multiprocessing
from copy import deepcopy
from multiprocessing import Manager
from types import MappingProxyType

from time_util.time_util import TimeUtil, timeme


def get_ipc_metagraph(manager: Manager):
    metagraph = manager.Namespace()
    metagraph.neurons = manager.list()
    metagraph.hotkeys = manager.list()
    metagraph.uids = manager.list()
    return metagraph

def managerize_objects(cls, manager, obj_dict) -> None:
    """
    Converts objects into manager-compatible shared objects and
    sets them as attributes of the validator object.

    Args:
        manager: The multiprocessing.Manager() instance.
        obj_dict: A dictionary of objects to managerize {name: object}.
    """

    def simple_managerize(obj):
        # Handle the special case for the 'metagraph' object
        if name == "metagraph":
            temp = manager.Namespace()
            temp.neurons = manager.list()
            temp.hotkeys = manager.list()
            temp.uids = manager.list()
            return temp

        # Managerize dictionaries
        elif isinstance(obj, dict):
            managed_dict = manager.dict()
            return managed_dict

        # Managerize lists
        elif isinstance(obj, list):
            managed_list = manager.list()
            return managed_list
        else:
            raise ValueError(f"Unsupported object type: {type(obj)}")

    # Managerize each object, with special handling for 'metagraph'
    for name, obj in obj_dict.items():
        setattr(cls, name, simple_managerize(obj))


class CachedIPCPerfLedgerBundles:
    """A wrapper around a managerized dict that automatically caches updates per process."""

    def __init__(self, ipc_manager, class_def):
        self.shared_dict = ipc_manager.dict()  # Managerized dict
        self._process_cache = {}  # Per-process cache
        self._last_cache_update = 0  # Last refresh timestamp (per process)
        self.class_def = class_def
        self.last_update_ms = multiprocessing.Value("d", 0)  # Shared value to track latest timestamp

    def get_read_only_dict(self):
        self._refresh_cache()
        return MappingProxyType(self._process_cache)

    def get_deepcopied_dict(self):
        self._refresh_cache()
        return deepcopy(self._process_cache)

    @timeme
    def _get_last_update_ms(self):
        """Determines if the local cache is stale by comparing timestamps."""
        return self.last_update_ms.value  # Efficient shared access


        if not self.shared_dict:  # Edge case: empty managerized dict
            return 0

        latest_manager_update = max(
            (
                perf_ledger.last_update_ms if perf_ledger else 0
                for sub_dict in self.shared_dict.values()  # Navigate first-level keys
                for perf_ledger in sub_dict.values()  # Navigate second-level PerfLedger objects
                if isinstance(perf_ledger, self.class_def)  # Ensure it's the right type
            ),
            default=0  # Handle case where all values are missing
        )

        return latest_manager_update

    def _refresh_cache(self):
        """Pulls the latest data from the manager if an update is detected."""
        last_update_ms = self._get_last_update_ms()
        if last_update_ms != self._last_cache_update:
            current_last_update_formatted = TimeUtil.millis_to_formatted_date_str(self._last_cache_update)
            last_update_ms_formatted = TimeUtil.millis_to_formatted_date_str(last_update_ms)
            print(f"[{multiprocessing.current_process().name}] Refreshing cache. {current_last_update_formatted} -> {last_update_ms_formatted}")
            self._process_cache = dict(self.shared_dict)  # Fetch entire dict
            self._last_cache_update = last_update_ms

    def __delitem__(self, key):
        """Delete item from both cache and shared dictionary."""
        del self.shared_dict[key]  # Ensure update is in shared memory
        del self._process_cache[key]

    def __getitem__(self, key):
        """Auto-refresh before accessing an item."""
        self._refresh_cache()
        return self._process_cache.get(key, None)  # Avoid KeyError

    def __setitem__(self, key, value):
        """Set item in both cache and shared dictionary."""
        self.shared_dict[key] = value  # Ensure update is in shared memory
        self._process_cache[key] = value
        self.last_update_ms.value = max(self.last_update_ms.value,
                                        value['portfolio'].last_update_ms)  # Keep the latest timestamp

    def keys(self):
        """Return dictionary keys, refreshing first if needed."""
        self._refresh_cache()
        return self._process_cache.keys()

    def values(self):
        """Return dictionary values, refreshing first if needed."""
        self._refresh_cache()
        return self._process_cache.values()

    def items(self):
        """Return dictionary items, refreshing first if needed."""
        self._refresh_cache()
        return self._process_cache.items()

    def get(self, key, default=None):
        """Get item with optional default, refreshing first if needed."""
        self._refresh_cache()
        return self._process_cache.get(key, default)

    def __contains__(self, key):
        """Check if key exists in the dictionary."""
        self._refresh_cache()
        return key in self._process_cache

    def __repr__(self):
        """Representation of the dictionary (shows the cached version)."""
        self._refresh_cache()
        return repr(self._process_cache)