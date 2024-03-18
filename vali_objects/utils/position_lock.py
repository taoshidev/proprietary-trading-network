from threading import Lock
import bittensor as bt

class PositionLocks:
    """
    Updating positions in the validator is vulnerable to race conditions on a per-miner and per-trade-pair basis. This
    class aims to solve that problem by locking the positions for a given miner and trade pair.
    """
    def __init__(self):
        self.locks = {}
        self.global_lock = Lock()

    def get_lock(self, miner_hotkey, trade_pair):
        #bt.logging.info(f"Getting lock for miner_hotkey [{miner_hotkey}] and trade_pair [{trade_pair}].")
        lock_key = (miner_hotkey, trade_pair)
        with self.global_lock:  # Ensure thread-safe access to the locks dictionary
            if lock_key not in self.locks:
                self.locks[lock_key] = Lock()
        return self.locks[lock_key]

    def cleanup_locks(self, active_miner_hotkeys):
        with self.global_lock:  # Ensure thread-safe modification of the locks dictionary
            keys_to_delete = [key for key in self.locks.keys() if key[0] not in active_miner_hotkeys]
            for key in keys_to_delete:
                del self.locks[key]

