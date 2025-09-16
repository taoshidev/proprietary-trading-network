import bittensor as bt
from typing import Dict, Optional

from time_util.time_util import TimeUtil
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.vali_config import TradePairCategory, ValiConfig
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils

ASSET_CLASS_SELECTION_TIME_MS = 1758092400000

class AssetSelectionManager:
    """
    Manages asset class selection for miners. Each miner can select an asset class (forex, crypto, etc.) 
    only once. Once selected, the miner cannot trade any trade pair from a different asset class.
    Asset selections are persisted to disk and loaded on startup.
    """

    def __init__(self, ipc_manager=None, running_unit_tests=False):
        """
        Initialize the AssetSelectionManager.
        
        Args:
            running_unit_tests: Whether the manager is being used in unit tests
        """
        self.running_unit_tests = running_unit_tests
        if ipc_manager:
            self.asset_selections = ipc_manager.dict()
        else:
            self.asset_selections: Dict[str, TradePairCategory] = {}  # miner_hotkey -> TradePairCategory

        self.ASSET_SELECTIONS_FILE = ValiBkpUtils.get_asset_selections_file_location(running_unit_tests=running_unit_tests)
        self._load_asset_selections_from_disk()

    def _load_asset_selections_from_disk(self) -> None:
        """Load asset selections from disk into memory using ValiUtils pattern."""
        try:
            disk_data = ValiUtils.get_vali_json_file_dict(self.ASSET_SELECTIONS_FILE)
            parsed_selections = self._parse_asset_selections_dict(disk_data)
            self.asset_selections.clear()
            self.asset_selections.update(parsed_selections)
            bt.logging.info(f"Loaded {len(self.asset_selections)} asset selections from disk")
        except Exception as e:
            bt.logging.error(f"Error loading asset selections from disk: {e}")

    def _save_asset_selections_to_disk(self) -> None:
        """Save asset selections from memory to disk using ValiBkpUtils pattern."""
        try:
            selections_data = self._to_dict()
            ValiBkpUtils.write_file(self.ASSET_SELECTIONS_FILE, selections_data)
            bt.logging.debug(f"Saved {len(self.asset_selections)} asset selections to disk")
        except Exception as e:
            bt.logging.error(f"Error saving asset selections to disk: {e}")

    def _to_dict(self) -> Dict:
        """Convert in-memory asset selections to disk format."""
        return {
            hotkey: asset_class.value
            for hotkey, asset_class in self.asset_selections.items()
        }

    @staticmethod
    def _parse_asset_selections_dict(json_dict: Dict) -> Dict[str, TradePairCategory]:
        """Parse disk format back to in-memory format."""
        parsed_selections = {}
        
        for hotkey, asset_class_str in json_dict.items():
            try:
                if asset_class_str:
                    # Convert string back to TradePairCategory enum
                    asset_class = TradePairCategory(asset_class_str)
                    parsed_selections[hotkey] = asset_class
            except ValueError as e:
                bt.logging.warning(f"Invalid asset selection for miner {hotkey}: {e}")
                continue
                
        return parsed_selections

    def is_valid_asset_class(self, asset_class: str) -> bool:
        """
        Validate if the provided asset class is valid.
        
        Args:
            asset_class: The asset class string to validate
            
        Returns:
            True if valid, False otherwise
        """
        valid_asset_classes = [category.value for category in TradePairCategory]
        return asset_class.lower() in [cls.lower() for cls in valid_asset_classes]

    def validate_order_asset_class(self, miner_hotkey: str, trade_pair_category: TradePairCategory, timestamp_ms: int=None) -> bool:
        """
        Check if a miner is allowed to trade a specific asset class.

        Args:
            miner_hotkey: The miner's hotkey
            trade_pair_category: The trade pair category to check

        Returns:
            True if the miner can trade this asset class, False otherwise
        """
        if timestamp_ms is None:
            timestamp_ms = TimeUtil.now_in_millis()
        if timestamp_ms < ASSET_CLASS_SELECTION_TIME_MS:
            return True

        selected_asset_class = self.asset_selections.get(miner_hotkey, None)
        if selected_asset_class is None:
            return False

        # Check if the selected asset class matches the trade pair category
        return selected_asset_class == trade_pair_category

    def process_asset_selection_request(self, asset_selection: str, miner: str) -> Dict[str, str]:
        """
        Process an asset selection request from a miner.
        
        Args:
            asset_selection: The asset class the miner wants to select
            miner: The miner's hotkey
            
        Returns:
            Dict containing success status and message
        """
        try:
            # Validate asset class
            if not self.is_valid_asset_class(asset_selection):
                valid_classes = [category.value for category in TradePairCategory]
                return {
                    'successfully_processed': False,
                    'error_message': f'Invalid asset class. Valid options are: {", ".join(valid_classes)}'
                }

            # Check if miner has already selected an asset class
            if miner in self.asset_selections:
                current_selection = self.asset_selections.get(miner)
                return {
                    'successfully_processed': False,
                    'error_message': f'Asset class already selected: {current_selection.value}. Cannot change selection.'
                }

            # Convert string to TradePairCategory and set the asset selection
            asset_class = TradePairCategory(asset_selection.lower())
            self.asset_selections[miner] = asset_class
            self._save_asset_selections_to_disk()
            
            bt.logging.info(f"Miner {miner} selected asset class: {asset_selection}")
            
            return {
                'successfully_processed': True,
                'success_message': f'Miner {miner} successfully selected asset class: {asset_selection}'
            }
            
        except Exception as e:
            bt.logging.error(f"Error processing asset selection request for miner {miner}: {e}")
            return {
                'successfully_processed': False,
                'error_message': 'Internal server error processing asset selection request'
            }

    def clear_all_selections(self) -> None:
        """
        Clear all asset selections
        """
        self.asset_selections.clear()
        self._save_asset_selections_to_disk()
        bt.logging.warning("Cleared all asset selections")
