import os
from enum import Enum
from multiprocessing import Manager, Pool


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


class ParallelizationMode(Enum):
    SERIAL = 0
    PYSPARK = 1
    MULTIPROCESSING = 2

def get_multiprocessing_pool(parallel_mode: ParallelizationMode, num_processes: int = 0):
    print(f"parallel_mode: {parallel_mode} ({type(parallel_mode)})")
    pool = None
    if parallel_mode == ParallelizationMode.MULTIPROCESSING:
        print("Creating multiprocessing pool...")
        pool = Pool(num_processes) if num_processes else Pool()
        print(f"Pool created: {pool}")
    else:
        print("Not using multiprocessing mode.")
    return pool
def get_spark_session(parallel_mode: ParallelizationMode):
    if parallel_mode == ParallelizationMode.PYSPARK:
        # Check if running in Databricks
        is_databricks = 'DATABRICKS_RUNTIME_VERSION' in os.environ
        # Initialize Spark
        if is_databricks:
            # In Databricks, 'spark' is already available in the global namespace
            print("Running in Databricks environment, using existing spark session")
            should_close = False

        else:
            # Create a new Spark session if not in Databricks
            from pyspark.sql import SparkSession

            print("getOrCreate Spark session")
            spark = SparkSession.builder \
                .appName("PerfLedgerManager") \
                .config("spark.executor.memory", "4g") \
                .config("spark.driver.memory", "6g") \
                .config("spark.executor.cores", "4") \
                .config("spark.driver.maxResultSize", "2g") \
                .getOrCreate()
            should_close = True

    else:
        spark = None
        should_close = False

    return spark, should_close