# developer: jbonilla
# Copyright © 2024 Taoshi Inc

"""
Custom multiprocessing Manager for PTN shared objects.

This manager registers PTN-specific classes so they can be shared across
processes with automatic IPC marshaling.
"""

from typing import TYPE_CHECKING
from multiprocessing.managers import BaseManager

if TYPE_CHECKING:
    # Type hints for PyCharm/mypy - these methods are created at runtime by register()
    from shared_objects.metagraph_updater import MetagraphUpdater


class PTNManager(BaseManager):
    """
    Custom Manager for PTN that handles registration of shared objects.

    Usage:
        manager = PTNManager()
        manager.start()

        # Create objects in manager process - returns proxies
        metagraph_updater = manager.MetagraphUpdater(config, metagraph, ...)

        # Pass proxy to subprocess - method calls automatically execute in manager
        subprocess_obj = SubtensorWeightSetter(metagraph_updater=metagraph_updater)
    """

    if TYPE_CHECKING:
        # Type stubs for IDE support - actual methods created by register() at runtime
        def MetagraphUpdater(self, config, metagraph, hotkey: str, is_miner: bool,
                            position_inspector=None, position_manager=None,
                            shutdown_dict=None, slack_notifier=None,
                            weight_request_queue=None) -> 'MetagraphUpdater':
            """
            Create a MetagraphUpdater instance in the manager process.
            Returns a proxy for IPC-safe method calls.
            """
            ...


def register_ptn_classes():
    """
    Register PTN classes with the manager.

    This must be called before creating manager instances.
    Import here to avoid circular dependencies.
    """
    from shared_objects.metagraph_updater import MetagraphUpdater

    # Register MetagraphUpdater so it can be shared across processes
    PTNManager.register('MetagraphUpdater', MetagraphUpdater)


# Register classes when module is imported
register_ptn_classes()
