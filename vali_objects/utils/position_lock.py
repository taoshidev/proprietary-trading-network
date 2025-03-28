from multiprocessing import Lock as MPLock
from threading import Lock
class PositionLocks:
    """
    Updating positions in the validator is vulnerable to race conditions on a per-miner and per-trade-pair basis. This
    class aims to solve that problem by locking the positions for a given miner and trade pair.
    """
    def get_new_lock(self):
        return Lock() if self.is_backtesting else MPLock()

    def __init__(self, hotkey_to_positions=None, is_backtesting=False):
        self.locks = {}
        self.is_backtesting = is_backtesting
        if hotkey_to_positions:
            for hotkey, positions in hotkey_to_positions.items():
                for p in positions:
                    key = (hotkey, p.trade_pair.trade_pair_id)
                    if key not in self.locks:
                        self.locks[key] = self.get_new_lock()
        #self.global_lock = Lock()

    def get_lock(self, miner_hotkey:str, trade_pair_id:str):
        #bt.logging.info(f"Getting lock for miner_hotkey [{miner_hotkey}] and trade_pair [{trade_pair}].")
        lock_key = (miner_hotkey, trade_pair_id)
        if lock_key not in self.locks:
            self.locks[lock_key] = self.get_new_lock()
        return self.locks[lock_key]

    #def cleanup_locks(self, active_miner_hotkeys):
    #    with self.global_lock:  # Ensure thread-safe modification of the locks dictionary
    #        keys_to_delete = [key for key in self.locks.keys() if key[0] not in active_miner_hotkeys]
    #        for key in keys_to_delete:
    #            del self.locks[key]

