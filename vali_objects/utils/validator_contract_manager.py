import threading
from datetime import timezone, datetime, timedelta
import bittensor as bt
from bittensor_wallet import Wallet
from collateral_sdk import CollateralManager, Network
from typing import Dict, Any, Optional, List
import traceback
import asyncio
import json
from time_util.time_util import TimeUtil
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
import template.protocol

class CollateralRecord:
    def __init__(self, account_size, update_time_ms):
        self.account_size = account_size
        self.update_time_ms = update_time_ms
        self.valid_date_timestamp = CollateralRecord.valid_from_ms(update_time_ms)

    @staticmethod
    def valid_from_ms(update_time_ms) -> int:
        """Returns timestamp of start of next day (00:00:00 UTC) when this record is valid"""
        dt = datetime.fromtimestamp(update_time_ms / 1000, tz=timezone.utc)
        start_of_day = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        # Record is valid from the start of the next day
        start_of_next_day = start_of_day + timedelta(days=1)
        return int(start_of_next_day.timestamp() * 1000)

    @property
    def valid_date_str(self) -> str:
        """Returns YYYY-MM-DD format for easy reading"""
        return TimeUtil.millis_to_short_date_str(self.valid_date_timestamp)

    def __repr__(self):
        """String representation"""
        return str(vars(self))


class ValidatorContractManager:
    """
    Manages collateral contract interactions for validators.
    Handles deposit processing, withdrawal validation, and EVM contract operations.
    This class acts as the validator's interface to the collateral system.
    """
    
    def __init__(self, config=None, position_manager=None, ipc_manager=None, running_unit_tests=False, is_backtesting=False, metagraph=None):
        self.config = config
        self.position_manager = position_manager
        self.is_mothership = 'ms' in ValiUtils.get_secrets(running_unit_tests=running_unit_tests)
        self.is_backtesting = is_backtesting
        self.metagraph = metagraph
        self._account_sizes_lock = None
        
        # Store network type for dynamic max_theta property
        if config is not None:
            self.is_testnet = config.subtensor.network == "test"
        else:
            bt.logging.info("Config in contract manager is None")
            self.is_testnet = False

        if self.is_testnet:
            bt.logging.info("Using testnet collateral manager")
            self.collateral_manager = CollateralManager(Network.TESTNET)
        else:
            bt.logging.info("Using mainnet collateral manager")
            self.collateral_manager = CollateralManager(Network.MAINNET)

        # GCP secret manager
        self._gcp_secret_manager_client = None
        self.secrets = None
        # Initialize vault wallet as None for all validators
        self.vault_wallet = None

        # Initialize miner account sizes file location
        self.MINER_ACCOUNT_SIZES_FILE = ValiBkpUtils.get_miner_account_sizes_file_location(running_unit_tests=running_unit_tests)
        
        # Load existing data from disk or initialize empty
        if ipc_manager:
            self.miner_account_sizes = ipc_manager.dict()
        else:
            self.miner_account_sizes: Dict[str, List[CollateralRecord]] = {}
        self._load_miner_account_sizes_from_disk()

    @property
    def account_sizes_lock(self):
        if not self._account_sizes_lock:
            self._account_sizes_lock = threading.RLock()
        return self._account_sizes_lock

    @property
    def max_theta(self) -> float:
        """
        Get the current maximum collateral balance limit in theta tokens.

        Returns:
            float: Maximum balance limit based on network type and current date
        """
        if self.is_testnet:
            return ValiConfig.MAX_COLLATERAL_BALANCE_TESTNET
        else:
            return ValiConfig.MAX_COLLATERAL_BALANCE_THETA.value()

    @property
    def min_theta(self) -> float:
        """
        Get the current maximum collateral balance limit in theta tokens.

        Returns:
            float: Maximum balance limit based on network type and current date
        """
        if self.is_testnet:
            return ValiConfig.MIN_COLLATERAL_BALANCE_TESTNET
        else:
            return ValiConfig.MIN_COLLATERAL_BALANCE_THETA

    def load_contract_owner(self):
        """
        Load EVM contract owner secrets and vault wallet.
        This validator must be authorized to execute collateral operations.
        """
        if not self.is_mothership:
            return
        try:
            # Load from secrets.json using ValiUtils
            self.secrets = ValiUtils.get_secrets()
            self.vault_wallet = bt.wallet(config=self.config)
            bt.logging.info(f"Vault wallet loaded: {self.vault_wallet}")
        except Exception as e:
            bt.logging.warning(f"Failed to load vault wallet: {e}")

    def _load_miner_account_sizes_from_disk(self):
        """Load miner account sizes from disk during initialization"""
        try:
            disk_data = ValiUtils.get_vali_json_file_dict(self.MINER_ACCOUNT_SIZES_FILE)
            self.miner_account_sizes = self._parse_miner_account_sizes_dict(disk_data)
            bt.logging.info(f"Loaded {len(self.miner_account_sizes)} miner account size records from disk")
        except Exception as e:
            bt.logging.warning(f"Failed to load miner account sizes from disk: {e}")

    def _save_miner_account_sizes_to_disk(self):
        """Save miner account sizes to disk"""
        try:
            data_dict = self.miner_account_sizes_dict()
            ValiBkpUtils.write_file(self.MINER_ACCOUNT_SIZES_FILE, data_dict)
        except Exception as e:
            bt.logging.error(f"Failed to save miner account sizes to disk: {e}")

    def miner_account_sizes_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert miner account sizes to checkpoint format for backup/sync"""
        json_dict = {}
        for hotkey, records in self.miner_account_sizes.items():
            json_dict[hotkey] = [vars(record) for record in records]
        return json_dict

    @staticmethod
    def _parse_miner_account_sizes_dict(data_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[CollateralRecord]]:
        """Parse miner account sizes from disk format back to CollateralRecord objects"""
        parsed_dict = {}
        for hotkey, records_data in data_dict.items():
            try:
                parsed_records = []
                for record_data in records_data:
                    if isinstance(record_data, dict) and all(key in record_data for key in ["account_size", "update_time_ms"]):
                        record = CollateralRecord(record_data["account_size"], record_data["update_time_ms"])
                        parsed_records.append(record)

                if parsed_records:  # Only add if we have valid records
                    parsed_dict[hotkey] = parsed_records
            except Exception as e:
                bt.logging.warning(f"Failed to parse account size records for {hotkey}: {e}")

        return parsed_dict

    def sync_miner_account_sizes_data(self, account_sizes_data: Dict[str, List[Dict[str, Any]]]):
        """Sync miner account sizes data from external source (backup/sync)"""
        if not account_sizes_data:
            bt.logging.warning("miner_account_sizes_data appears empty or invalid")
            return

        try:
            with self.account_sizes_lock:
                synced_data = self._parse_miner_account_sizes_dict(account_sizes_data)
                self.miner_account_sizes.clear()
                self.miner_account_sizes.update(synced_data)
                self._save_miner_account_sizes_to_disk()
                bt.logging.info(f"Synced {len(self.miner_account_sizes)} miner account size records")
        except Exception as e:
            bt.logging.error(f"Failed to sync miner account sizes data: {e}")

    def get_secret(self, secret_name: str) -> Optional[str]:
        """
        Get secret with fallback to local secrets

        Args:
            secret_name (str): name of secret
        """
        secret = self._get_gcp_secret(secret_name)
        if secret is not None:
            return secret

        secret = self.secrets.get(secret_name)
        if secret is not None:
            bt.logging.info(f"{secret_name} retrieved from local secrets file")
        return secret

    def _get_gcp_secret(self, secret_name: str) -> Optional[str]:
        """
        Get vault password from Google Cloud Secret Manager.

        Args:
            secret_name (str): name of secret

        Returns:
            str: Vault password or None if not found
        """
        try:
            if self._gcp_secret_manager_client is None:
                # noinspection PyPackageRequirements
                from google.cloud import secretmanager

                self._gcp_secret_manager_client = secretmanager.SecretManagerServiceClient()

            secret_path = self._gcp_secret_manager_client.secret_version_path(
                self.secrets.get('gcp_project_name'), self.secrets.get(secret_name), "latest"
            )
            response = self._gcp_secret_manager_client.access_secret_version(name=secret_path)
            secret = response.payload.data.decode()

            if secret:
                bt.logging.info(f"{secret_name} retrieved from Google Cloud Secret Manager")
                return secret
            else:
                bt.logging.debug(f"{secret_name} not found in Google Cloud Secret Manager")
                return None
        except Exception as e:
            bt.logging.debug(f"Failed to retrieve {secret_name} from Google Cloud: {e}")

    def to_theta(self, rao_amount: int) -> float:
        """
        Convert rao_theta amount to theta tokens.

        Args:
            rao_amount (int): Amount in RAO units

        Returns:
            float: Amount in theta tokens
        """
        theta_amount = rao_amount / 10 ** 9  # Convert rao_theta to theta
        return theta_amount
    
    def process_deposit_request(self, extrinsic_hex: str) -> Dict[str, Any]:
        """
        Process a collateral deposit request using raw data.
        
        Args:
            extrinsic_hex (str): Hex-encoded extrinsic data
            amount (float): Amount in theta tokens
            miner_address (str): Miner's SS58 address
            
        Returns:
            Dict[str, Any]: Result of deposit operation
        """
        try:
            bt.logging.info("Received deposit request")
            # Decode and validate the extrinsic
            try:
                encoded_extrinsic = bytes.fromhex(extrinsic_hex)
                extrinsic = self.collateral_manager.decode_extrinsic(encoded_extrinsic)
                bt.logging.info("Extrinsic decoded successfully")
            except Exception as e:
                error_msg = f"Invalid extrinsic data: {str(e)}"
                bt.logging.error(error_msg)
                return {
                    "successfully_processed": False,
                    "error_message": error_msg
                }
            
            # Execute the deposit through the collateral manager
            try:
                miner_hotkey = next(arg["value"] for arg in extrinsic.value["call"]["call_args"] if arg["name"] == "hotkey")
                deposit_amount = next(arg["value"] for arg in extrinsic.value["call"]["call_args"] if arg["name"] == "alpha_amount")
                deposit_amount_theta = self.to_theta(deposit_amount)
                
                # Check collateral balance limit before processing
                try:
                    current_balance_theta = self.to_theta(self.collateral_manager.balance_of(miner_hotkey))
                    
                    if current_balance_theta + deposit_amount_theta > self.max_theta:
                        error_msg = (f"Deposit would exceed maximum balance limit. "
                                   f"Current: {current_balance_theta:.2f} Theta, "
                                   f"Deposit: {deposit_amount_theta:.2f} Theta, "
                                   f"Limit: {self.max_theta} Theta")
                        bt.logging.warning(error_msg)
                        return {
                            "successfully_processed": False,
                            "error_message": error_msg
                        }

                except Exception as e:
                    bt.logging.error(f"Failed to check balance limit: {e}")
                    return {
                        "successfully_processed": False,
                        "error_message": e
                    }

                # # All positions must be closed before a miner can deposit or withdraw
                # if len(self.position_manager.get_positions_for_one_hotkey(miner_hotkey, only_open_positions=True)) > 0:
                #     return {
                #         "successfully_processed": False,
                #         "error_message": "Miner has open positions, please close all positions before depositing or withdrawing collateral"
                #     }

                bt.logging.info(f"Processing deposit for: {deposit_amount_theta} Theta to miner: {miner_hotkey}")
                owner_address = self.get_secret("collateral_owner_address")
                owner_private_key = self.get_secret("collateral_owner_private_key")
                vault_password = self.get_secret("gcp_vali_pw_name")
                try:
                    deposited_balance = self.collateral_manager.deposit(
                        extrinsic=extrinsic,
                        source_hotkey=miner_hotkey,
                        vault_stake=self.vault_wallet.hotkey.ss58_address,
                        vault_wallet=self.vault_wallet,
                        owner_address=owner_address,
                        owner_private_key=owner_private_key,
                        wallet_password=vault_password
                    )
                finally:
                    del owner_address
                    del owner_private_key
                    del vault_password

                bt.logging.info(f"Deposit successful: {self.to_theta(deposited_balance.rao)} Theta deposited to miner: {miner_hotkey}")
                self.set_miner_account_size(miner_hotkey, TimeUtil.now_in_millis())
                return {
                    "successfully_processed": True,
                    "error_message": ""
                }
                
            except Exception as e:
                error_msg = f"Deposit execution failed: {str(e)}"
                bt.logging.error(error_msg)
                return {
                    "successfully_processed": False,
                    "error_message": error_msg
                }
                
        except Exception as e:
            error_msg = f"Deposit processing error: {str(e)}"
            bt.logging.error(error_msg)
            return {
                "successfully_processed": False,
                "error_message": error_msg
            }

    def process_withdrawal_request(self, amount: float, miner_coldkey: str, miner_hotkey: str) -> Dict[str, Any]:
        """
        Process a collateral withdrawal request using raw data.
        
        Args:
            amount (float): Amount to withdraw in theta tokens
            miner_coldkey (str): Miner's SS58 wallet coldkey address to return collateral to
            miner_hotkey (str): Miner's SS58 hotkey
            
        Returns:
            Dict[str, Any]: Result of withdrawal operation
        """
        try:
            bt.logging.info("Received withdrawal request")
            # Check current collateral balance
            try:
                current_balance = self.collateral_manager.balance_of(miner_hotkey)
                theta_current_balance = self.to_theta(current_balance)
                if amount > theta_current_balance:
                    error_msg = f"Insufficient collateral balance. Available: {theta_current_balance}, Requested: {amount}"
                    bt.logging.error(error_msg)
                    return {
                        "successfully_processed": False,
                        "error_message": error_msg,
                        "returned_amount": 0.0,
                        "returned_to": ""
                    }
            except Exception as e:
                error_msg = f"Failed to check collateral balance: {str(e)}"
                bt.logging.error(error_msg)
                return {
                    "successfully_processed": False,
                    "error_message": error_msg,
                    "returned_amount": 0.0,
                    "returned_to": ""
                }
            
            # Execute the withdrawal through the collateral manager
            try:
                # eligible_for_withdrawal = self.eligible_for_withdrawal(miner_hotkey)
                # if amount > eligible_for_withdrawal:
                #     error_msg = f"Withdrawal request exceeds eligible amount based on drawdown. Available: {eligible_for_withdrawal}, Requested: {amount}"
                #     bt.logging.error(error_msg)
                #     return {
                #         "successfully_processed": False,
                #         "error_message": error_msg,
                #         "returned_amount": 0.0,
                #         "returned_to": ""
                #     }
                #
                # # All positions must be closed before a miner can deposit or withdraw
                # if len(self.position_manager.get_positions_for_one_hotkey(miner_hotkey, only_open_positions=True)) > 0:
                #     return {
                #         "successfully_processed": False,
                #         "error_message": "Miner has open positions, please close all positions before depositing or withdrawing collateral"
                #     }

                bt.logging.info(f"Processing withdrawal request from {miner_hotkey} for {amount} Theta")
                owner_address = self.get_secret("collateral_owner_address")
                owner_private_key = self.get_secret("collateral_owner_private_key")
                vault_password = self.get_secret("gcp_vali_pw_name")
                try:
                    withdrawn_balance = self.collateral_manager.withdraw(
                        amount=int(amount * 10**9), # convert theta to rao_theta
                        source_coldkey=miner_coldkey,
                        source_hotkey=miner_hotkey,
                        vault_stake=self.vault_wallet.hotkey.ss58_address,
                        vault_wallet=self.vault_wallet,
                        owner_address=owner_address,
                        owner_private_key=owner_private_key,
                        wallet_password=vault_password
                    )
                finally:
                    del owner_address
                    del owner_private_key
                    del vault_password
                returned_theta = self.to_theta(withdrawn_balance.rao)
                bt.logging.info(f"Withdrawal successful: {returned_theta} Theta withdrawn for {miner_hotkey}, returned to {miner_coldkey}")
                self.set_miner_account_size(miner_hotkey, TimeUtil.now_in_millis())
                return {
                    "successfully_processed": True,
                    "error_message": "",
                    "returned_amount": returned_theta,
                    "returned_to": miner_coldkey
                }
                
            except Exception as e:
                error_msg = f"Withdrawal execution failed: {str(e)}"
                bt.logging.error(error_msg)
                return {
                    "successfully_processed": False,
                    "error_message": error_msg,
                    "returned_amount": 0.0,
                    "returned_to": ""
                }
                
        except Exception as e:
            error_msg = f"Withdrawal processing error: {str(e)}"
            bt.logging.error(error_msg)
            return {
                "successfully_processed": False,
                "error_message": error_msg,
                "returned_amount": 0.0,
                "returned_to": ""
            }

    def eligible_for_withdrawal(self, miner_hotkey: str) -> float:
        """
        Return the amount of collateral balance that is eligible for withdrawal.

        The miner is eligible to withdraw an amount proportional to 50% of their drawdown.
        For ex:
        10% drawdown (elimination) -> Eligible to withdraw 50%
        5% drawdown -> Eligible to withdraw 75%
        3% drawdown -> Eligible to withdraw 85%
        """
        balance = self.get_miner_collateral_balance(miner_hotkey)

        filtered_ledgers = self.position_manager.perf_ledger_manager.filtered_ledger_for_scoring(portfolio_only=True, hotkeys=[miner_hotkey])
        miner_ledger = filtered_ledgers.get(miner_hotkey)
        drawdown = LedgerUtils.instantaneous_max_drawdown(miner_ledger)

        drawdown_proportion = (drawdown - ValiConfig.MAX_TOTAL_DRAWDOWN) / (1 - ValiConfig.MAX_TOTAL_DRAWDOWN)
        eligible_proportion = ValiConfig.BASE_COLLATERAL_RETURNED + (1 - ValiConfig.BASE_COLLATERAL_RETURNED) * drawdown_proportion
        eligible_for_withdrawal = balance * eligible_proportion
        return eligible_for_withdrawal

    def compute_slash_amount(self, miner_hotkey: str) -> float:
        """
        Compute the amount of collateral balance to slash, depending on current drawdown.

        The amount slashed is 50% of the drawdown, scaled to the total collateral balance.
        For ex:
        10% drawdown (elimination) -> Slash 50%
        5% drawdown -> Slash 25%
        3% drawdown -> Slash 15%

        Args:
            miner_hotkey: miner hotkey to slash from

        Returns:
            float: amount to slash
        """
        filtered_ledgers = self.position_manager.perf_ledger_manager.filtered_ledger_for_scoring(portfolio_only=True, hotkeys=[miner_hotkey])
        miner_ledger = filtered_ledgers.get(miner_hotkey)

        try:
            # Get the portfolio ledger for the miner
            filtered_ledger = self.position_manager.perf_ledger_manager.filtered_ledger_for_scoring(
                portfolio_only=True,
                hotkeys=[miner_hotkey]
            )

            ledger = filtered_ledger.get(miner_hotkey)
            if not ledger or len(ledger.cps) == 0:
                bt.logging.warning(f"No ledger data found for {miner_hotkey}")
                return 0.0

            # Get current drawdown percentage
            max_drawdown = LedgerUtils.instantaneous_max_drawdown(ledger)

            # Get current balance
            current_balance_theta = self.get_miner_collateral_balance(miner_hotkey)
            if current_balance_theta is None or current_balance_theta <= 0:
                bt.logging.warning(f"No collateral balance for {miner_hotkey}")
                return 0.0

            # Calculate slash amount (50% of drawdown percentage)
            drawdown_proportion = 1 - ((max_drawdown - ValiConfig.MAX_TOTAL_DRAWDOWN) / (1 - ValiConfig.MAX_TOTAL_DRAWDOWN))  # scales x% drawdown to 100% of collateral
            slash_proportion = drawdown_proportion * ValiConfig.SLASH_PROPORTION
            slash_amount = current_balance_theta * slash_proportion

            bt.logging.info(f"Computed slashing for {miner_hotkey}: "
                            f"Drawdown: {max_drawdown:.2f}, "
                            f"Slash: {slash_proportion:.2f} = {slash_amount:.2f} Theta")

            return slash_amount

        except Exception as e:
            bt.logging.error(f"Failed to compute slash amount for {miner_hotkey}: {e}")
            return 0.0

    def slash_miner_collateral(self, miner_hotkey: str, slash_amount=None) -> bool:
        """
        Slash miner's collateral

        Args:
            miner_hotkey: miner hotkey to slash from
        """
        current_balance_theta = self.get_miner_collateral_balance(miner_hotkey)
        if current_balance_theta is None or current_balance_theta <= 0:
            return False

        if slash_amount is None:
            slash_amount = self.compute_slash_amount(miner_hotkey)

        # Ensure we don't slash more than the current balance
        slash_amount = min(slash_amount, current_balance_theta)
        if slash_amount <= 0:
            bt.logging.info(f"No slashing required for {miner_hotkey} (calculated amount: {slash_amount})")
            return True

        # Call collateral SDK slash method
        try:
            bt.logging.info(f"Processing slash of {slash_amount} Theta from {miner_hotkey}")
            owner_address = self.get_secret("collateral_owner_address")
            owner_private_key = self.get_secret("collateral_owner_private_key")
            try:
                self.collateral_manager.slash(
                    address=miner_hotkey,
                    amount=int(slash_amount * 10 ** 9),
                    owner_address=owner_address,
                    owner_private_key=owner_private_key,
                )
            finally:
                del owner_address
                del owner_private_key
            bt.logging.info(f"Successfully slashed {slash_amount} Theta from {miner_hotkey}")
            return True

        except Exception as e:
            bt.logging.error(f"Failed to execute slashing for {miner_hotkey}: {e}")
            return False

    def get_miner_collateral_balance(self, miner_address: str) -> Optional[float]:
        """
        Get a miner's current collateral balance in theta tokens.

        Args:
            miner_address (str): Miner's SS58 address

        Returns:
            Optional[float]: Balance in theta tokens, or None if error
        """
        try:
            rao_balance = self.collateral_manager.balance_of(miner_address)
            return self.to_theta(rao_balance)
        except Exception as e:
            bt.logging.error(f"Failed to get collateral balance for {miner_address}: {e}")
            return None

    def get_total_collateral(self) -> int:
        """Get total collateral in the contract in theta."""
        try:
            return self.collateral_manager.get_total_collateral()
        except Exception as e:
            bt.logging.error(f"Failed to get total collateral: {e}")
            return 0

    def get_slashed_collateral(self) -> int:
        """Get total slashed collateral in theta."""
        try:
            return self.collateral_manager.get_slashed_collateral()
        except Exception as e:
            bt.logging.error(f"Failed to get slashed collateral: {e}")
            return 0

    def set_miner_account_size(self, hotkey: str, timestamp_ms: int=None) -> None:
        """
        Set the account size for a miner. Saves to memory and disk.
        Records are kept in chronological order.

        Args:
            hotkey: Miner's hotkey (SS58 address)
            timestamp_ms: Timestamp for the record (defaults to now)
        """
        if timestamp_ms is None:
            timestamp_ms = TimeUtil.now_in_millis()

        collateral_balance = self.get_miner_collateral_balance(hotkey)
        if collateral_balance is None:
            bt.logging.warning(f"Could not retrieve collateral balance for {hotkey}")
            return

        account_size = collateral_balance * ValiConfig.COST_PER_THETA
        collateral_record = CollateralRecord(account_size, timestamp_ms)

        if hotkey not in self.miner_account_sizes:
            self.miner_account_sizes[hotkey] = []

        # Add the new record, IPC dict requires reassignment of entire k, v pair
        self.miner_account_sizes[hotkey] = self.miner_account_sizes[hotkey] + [collateral_record]

        # Save to disk
        self._save_miner_account_sizes_to_disk()

        # Broadcast collateral record to other validators
        if self.is_mothership:
            self._broadcast_collateral_record_update_to_validators(hotkey, collateral_record)

        if hasattr(account_size, '_mock_name'):  # It's a mock
            bt.logging.info(
                f"Updated account size for {hotkey}: ${account_size} (valid from {collateral_record.valid_date_str})")
        else:
            bt.logging.info(
                f"Updated account size for {hotkey}: ${account_size:,.2f} (valid from {collateral_record.valid_date_str})")

    def get_miner_account_size(self, hotkey: str, timestamp_ms: int=None, most_recent: bool=False, records_dict: dict=None) -> float | None:
        """
        Get the account size for a miner at a given timestamp. Iterate list in reverse chronological order, and return
        the first record whose valid_date_timestamp <= start_of_day_ms

        Args:
            hotkey: Miner's hotkey (SS58 address)
            timestamp_ms: Timestamp to query for (defaults to now)
            most_recent: If True, return most recent record regardless of timestamp
            records_dict: Optional dict to use instead of self.miner_account_sizes (for cached lookups)

        Returns:
            Account size in USD, or None if no applicable records
        """
        if timestamp_ms is None:
            timestamp_ms = TimeUtil.now_in_millis()

        # Use provided records_dict or default to self.miner_account_sizes
        source_records = records_dict if records_dict is not None else self.miner_account_sizes
        
        if hotkey not in source_records or not source_records[hotkey]:
            return None

        # Get start of the requested day
        start_of_day_ms = int(
            datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .timestamp() * 1000
        )

        # Return most recent record
        if most_recent:
            most_recent_record = source_records[hotkey][-1]
            return most_recent_record.account_size

        # Iteate in reversed order, and return the first record that is valid for or before the requested day
        for record in reversed(source_records[hotkey]):
            if record.valid_date_timestamp <= start_of_day_ms:
                return record.account_size

        # No applicable records found
        return None

    def _broadcast_collateral_record_update_to_validators(self, hotkey: str, collateral_record: CollateralRecord):
        """
        Broadcast CollateralRecord synapse to other validators.
        Runs in a separate thread to avoid blocking the main process.
        """
        def run_broadcast():
            try:
                asyncio.run(self._async_broadcast_collateral_record(hotkey, collateral_record))
            except Exception as e:
                bt.logging.error(f"Failed to broadcast collateral record for {hotkey}: {e}")
        
        thread = threading.Thread(target=run_broadcast, daemon=True)
        thread.start()

    async def _async_broadcast_collateral_record(self, hotkey: str, collateral_record: CollateralRecord):
        """
        Asynchronously broadcast CollateralRecord synapse to other validators.
        """
        try:
            # Get other validators to broadcast to
            if self.is_testnet:
                validator_axons = [n.axon_info for n in self.metagraph.neurons if n.axon_info.ip != ValiConfig.AXON_NO_IP and n.axon_info.hotkey != self.vault_wallet.hotkey.ss58_address]
            else:
                validator_axons = [n.axon_info for n in self.metagraph.neurons if n.stake > bt.Balance(ValiConfig.STAKE_MIN) and n.axon_info.ip != ValiConfig.AXON_NO_IP and n.axon_info.hotkey != self.vault_wallet.hotkey.ss58_address]

            if not validator_axons:
                bt.logging.debug("No other validators to broadcast CollateralRecord to")
                return
            
            # Create CollateralRecord synapse with the data
            collateral_record_data = {
                "hotkey": hotkey,
                "account_size": collateral_record.account_size,
                "update_time_ms": collateral_record.update_time_ms
            }
            
            collateral_synapse = template.protocol.CollateralRecord(
                collateral_record=collateral_record_data
            )
            
            bt.logging.info(f"Broadcasting CollateralRecord for {hotkey} to {len(validator_axons)} validators")
            
            # Send to other validators using dendrite
            async with bt.dendrite(wallet=self.vault_wallet) as dendrite:
                responses = await dendrite.aquery(validator_axons, collateral_synapse)
                
                # Log results
                success_count = 0
                for response in responses:
                    if response.successfully_processed:
                        success_count += 1
                    elif response.error_message:
                        bt.logging.warning(f"Failed to send CollateralRecord to {response.axon.hotkey}: {response.error_message}")
                
                bt.logging.info(f"CollateralRecord broadcast completed: {success_count}/{len(responses)} validators updated")
                
        except Exception as e:
            bt.logging.error(f"Error in async broadcast collateral record: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())

    def receive_collateral_record_update(self, collateral_record_data: dict) -> bool:
        """
        Process an incoming CollateralRecord synapse and update miner_account_sizes.
        
        Args:
            collateral_record_data: Dictionary containing hotkey, account_size, update_time_ms, valid_date_timestamp
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.is_mothership:
                return False
            with self.account_sizes_lock:
                # Extract data from the synapse
                hotkey = collateral_record_data.get("hotkey")
                account_size = collateral_record_data.get("account_size")
                update_time_ms = collateral_record_data.get("update_time_ms")
                bt.logging.info(f"Processing collateral record update for miner {hotkey}")

                if not all([hotkey, account_size is not None, update_time_ms]):
                    bt.logging.warning(f"Invalid collateral record data received: {collateral_record_data}")
                    return False

                # Create a CollateralRecord object
                collateral_record = CollateralRecord(account_size, update_time_ms)

                # Update miner account sizes
                if hotkey not in self.miner_account_sizes:
                    self.miner_account_sizes[hotkey] = []

                # Check if we already have this record (avoid duplicates)
                if self.get_miner_account_size(hotkey, most_recent=True) == account_size:
                    bt.logging.debug(f"Most recent collateral record for {hotkey} already exists")
                    return True

                # Add the new record, IPC dict requires reassignment of entire k, v pair
                self.miner_account_sizes[hotkey] = self.miner_account_sizes[hotkey] + [collateral_record]

                # Save to disk
                self._save_miner_account_sizes_to_disk()

                bt.logging.info(f"Updated miner account size for {hotkey}: ${account_size} (valid from {collateral_record.valid_date_str})")
                return True
            
        except Exception as e:
            bt.logging.error(f"Error processing collateral record update: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return False
