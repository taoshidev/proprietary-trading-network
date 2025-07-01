from collections import deque
import threading

class UUIDTracker:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.uuids = deque()
        self.uuid_set = set()
        self.lock = threading.Lock()

    def add_initial_uuids(self, hk_to_positions):
        try:
            n_orders_added = 0
            for hk, positions in hk_to_positions.items():
                for p in positions:
                    n_orders_added += len(p.orders)
                    for o in p.orders:
                        self.add(o.order_uuid)
            print(f"Loaded {n_orders_added} order uuids into uuid_tracker")
        except Exception as e:
            print(f"Error adding initial UUIDs: {e}. Continuing with empty UUID tracker.")

    def add(self, uuid):
        with self.lock:  # Ensure exclusive access within this block
            if uuid in self.uuid_set:
                return  # Avoid adding duplicates
            if len(self.uuids) >= self.capacity:
                old_uuid = self.uuids.popleft()  # Remove the oldest UUID
                self.uuid_set.remove(old_uuid)
            self.uuids.append(uuid)
            self.uuid_set.add(uuid)

    def remove(self, uuid):
        with self.lock:  # Ensure exclusive access within this block
            if uuid in self.uuid_set:
                self.uuids = deque(u for u in self.uuids if u != uuid)
                self.uuid_set.remove(uuid)

    def exists(self, uuid):
        with self.lock:  # Ensure exclusive access within this block
            return uuid in self.uuid_set