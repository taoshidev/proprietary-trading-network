import multiprocessing
import os
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


class CachedIPCDict:
    """A wrapper around a managerized dict that automatically caches updates per process."""

    def __init__(self, ipc_manager, class_def):
        self.shared_dict = ipc_manager.dict()  # Managerized dict
        self._process_cache = {}  # Per-process cache
        self.class_def = class_def
        self._hk_to_last_cache_update = {}
        self.ipc_hk_to_last_update_ms = ipc_manager.dict()  # Last refresh timestamp (per hotkey)

    def get_read_only_dict(self):
        self._refresh_cache_all_keys()
        return MappingProxyType(self._process_cache)

    def get_deepcopied_dict(self):
        self._refresh_cache_all_keys()
        return deepcopy(self._process_cache)

    def _get_ipc_last_update_ms(self, hotkey):
        """Determines if the local cache is stale by comparing timestamps."""
        return self.ipc_hk_to_last_update_ms.get(hotkey, 0)  # Efficient shared access

    def _get_last_cache_update_ms(self, hotkey):
        return self._hk_to_last_cache_update.get(hotkey, 0)

    def get_ipc_keys(self):
        return self.shared_dict.keys()

    @timeme
    def _refresh_cache_all_keys(self):
        """Pulls the latest data from the manager if an update is detected."""
        for hotkey, cache_last_update_ms in list(self.ipc_hk_to_last_update_ms.items()):
            self._refresh_cache_one_key(hotkey, cache_last_update_ms)

    def _refresh_cache_one_key(self, hotkey, cache_last_update_ms=None):
        """Pulls the latest data from the manager if an update is detected."""
        ipc_last_update_ms = self._get_ipc_last_update_ms(hotkey)
        if cache_last_update_ms is None:
            cache_last_update_ms = self._get_last_cache_update_ms(hotkey)
        if not ipc_last_update_ms:  # deleted or never existed
            if hotkey in self._process_cache:
                del self._process_cache[hotkey]
            if hotkey in self._hk_to_last_cache_update:
                del self._hk_to_last_cache_update[hotkey]
            return
        if cache_last_update_ms != ipc_last_update_ms:
            current_last_update_formatted = TimeUtil.millis_to_formatted_date_str(cache_last_update_ms)
            last_update_ms_formatted = TimeUtil.millis_to_formatted_date_str(ipc_last_update_ms)
            process = multiprocessing.current_process()
            print(f"{self.class_def} [{process.name}] Running in PID: {os.getpid()} (Parent: {os.getppid()}) Refreshing cache. {current_last_update_formatted} -> {last_update_ms_formatted}")
            new_value = self.shared_dict.get(hotkey)
            if new_value:
                self._process_cache[hotkey] = deepcopy(new_value)
                self._hk_to_last_cache_update[hotkey] = ipc_last_update_ms
            else:
                if hotkey in self._process_cache:
                    del self._process_cache[hotkey]
                if hotkey in self._hk_to_last_cache_update:
                    del self._hk_to_last_cache_update[hotkey]

    def __delitem__(self, key):
        """Delete item from both cache and shared dictionary."""
        if key in self.shared_dict:
            del self.shared_dict[key]  # Ensure update is in shared memory
            del self._hk_to_last_cache_update[key]
        del self._process_cache[key]
        del self.ipc_hk_to_last_update_ms[key]

    def __getitem__(self, key):
        """Auto-refresh before accessing an item."""
        self._refresh_cache_one_key(key)
        return self._process_cache.get(key, None)  # Avoid KeyError

    def __setitem__(self, key, value):
        """Set item in both cache and shared dictionary."""
        self.shared_dict[key] = value  # Ensure update is in shared memory
        self._process_cache[key] = value
        now_ms = TimeUtil.now_in_millis()
        self.ipc_hk_to_last_update_ms[key] = now_ms
        self._hk_to_last_cache_update[key] = now_ms

    def keys(self):
        self._refresh_cache_all_keys()
        """Return dictionary keys"""
        return self._process_cache.keys()

    def values(self):
        self._refresh_cache_all_keys()
        """Return dictionary values."""
        return self._process_cache.values()

    def items(self):
        """Return dictionary items, refreshing first if needed."""
        self._refresh_cache_all_keys()
        return self._process_cache.items()

    def get(self, key, default=None):
        """Get item with optional default, refreshing first if needed."""
        self._refresh_cache_one_key(key)
        return self._process_cache.get(key, default)

    def __contains__(self, key):
        """Check if key exists in the dictionary."""
        self._refresh_cache_one_key(key)
        return key in self._process_cache

    def __repr__(self):
        """Representation of the dictionary (shows the cached version)."""
        return repr(self._process_cache)