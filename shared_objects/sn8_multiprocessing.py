from multiprocessing import Manager

def get_ipc_metagraph(manager: Manager):
    metagraph = manager.Namespace()
    metagraph.neurons = manager.list()
    metagraph.hotkeys = manager.list()
    metagraph.uids = manager.list()
    metagraph.block_at_registration = manager.list()
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