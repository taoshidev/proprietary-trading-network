import bittensor as bt
from bittensor_wallet import Wallet
from collateral_sdk import CollateralManager, Network
from typing import Dict, Any, Optional
import traceback
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig

class ValidatorContractManager:
    """
    Manages collateral contract interactions for validators.
    Handles deposit processing, withdrawal validation, and EVM contract operations.
    This class acts as the validator's interface to the collateral system.
    """
    
    def __init__(self, config, metagraph, running_unit_tests=False):
        self.config = config
        self.metagraph = metagraph
        self.is_mothership = 'ms' in ValiUtils.get_secrets(running_unit_tests=running_unit_tests)
        
        if config.subtensor.network == "test":
            bt.logging.info("Using testnet collateral manager")
            self.collateral_manager = CollateralManager(Network.TESTNET)
            self.max_theta = ValiConfig.MAX_COLLATERAL_BALANCE_TESTNET
        else:
            bt.logging.info("Using mainnet collateral manager")
            self.collateral_manager = CollateralManager(Network.MAINNET)
            self.max_theta = ValiConfig.MAX_COLLATERAL_BALANCE_THETA
        
        # Load contract owner credentials from environment or config
        if self.is_mothership:
            self._load_contract_owner_credentials()
        
    def _load_contract_owner_credentials(self):
        """
        Load EVM contract owner credentials from secrets.json file.
        This validator must be authorized to execute collateral operations.
        """
        try:
            # Load from secrets.json using ValiUtils
            secrets = ValiUtils.get_secrets()
            self.owner_address = secrets.get('collateral_owner_address')
            self.owner_private_key = secrets.get('collateral_owner_private_key')
            self.vault_password = secrets.get('vault_password', None)
            if not self.owner_address or not self.owner_private_key:
                bt.logging.warning("Contract owner credentials not found. Collateral operations will fail.")
                self.owner_address = None
                self.owner_private_key = None
            else:
                bt.logging.info("Contract owner credentials loaded successfully")
                
        except Exception as e:
            bt.logging.warning(f"Failed to load contract owner credentials: {e}")
            self.owner_address = None
            self.owner_private_key = None
            self.vault_password = None


    def get_theta_token_price(self) -> float:
        """
        Calculate the current theta token price in TAO.

        Returns:
            float: theta token price in TAO
        """
        theta_price = self.metagraph.pool.tao_in / self.metagraph.pool.alpha_in
        bt.logging.debug(f"theta token price: {theta_price} TAO")
        return theta_price

    def theta_to_rao(self, theta_amount: float) -> int:
        """
        Convert theta token amount to RAO units.

        Args:
            theta_amount (float): Amount in theta tokens

        Returns:
            int: Amount in RAO units
        """
        theta_price = self.get_theta_token_price()
        tao_amount = theta_amount * theta_price
        rao_amount = int(tao_amount * 10 ** 9)  # Convert TAO to RAO

        bt.logging.debug(f"Converted {theta_amount} theta tokens to {tao_amount} TAO")
        return rao_amount

    def rao_to_theta(self, rao_amount: int) -> float:
        """
        Convert RAO amount to theta tokens.

        Args:
            rao_amount (int): Amount in RAO units

        Returns:
            float: Amount in theta tokens
        """
        theta_price = self.get_theta_token_price()
        tao_amount = rao_amount / 10 ** 9  # Convert RAO to TAO
        theta_amount = tao_amount / theta_price

        bt.logging.debug(f"Converted {tao_amount} TAO to {theta_amount} theta tokens")
        return theta_amount
    
    def process_deposit_request(self, extrinsic_hex: str, vault_wallet: Wallet) -> Dict[str, Any]:
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
                stake_list = self.collateral_manager.subtensor_api.staking.get_stake_for_coldkey(vault_wallet.coldkeypub.ss58_address)
                vault_stake = next(
                    (stake for stake in stake_list if stake.hotkey_ss58 == vault_wallet.hotkey.ss58_address),
                    None
                )
                miner_coldkey = extrinsic.value["address"]
                miner_hotkey = next(arg["value"] for arg in extrinsic.value["call"]["call_args"] if arg["name"] == "hotkey")
                deposit_amount = next(arg["value"] for arg in extrinsic.value["call"]["call_args"] if arg["name"] == "alpha_amount")
                deposit_amount_theta = self.rao_to_theta(deposit_amount)
                
                # Check collateral balance limit before processing
                try:
                    current_balance_theta = self.rao_to_theta(self.collateral_manager.balance_of(miner_coldkey))
                    
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
                
                bt.logging.info(f"Processing deposit for: {deposit_amount_theta} Theta to miner hotkey: {miner_hotkey} coldkey: {miner_coldkey}")
                deposited_balance = self.collateral_manager.deposit(
                    extrinsic=extrinsic,
                    sender=extrinsic.value["address"],
                    vault_stake=vault_stake.hotkey_ss58,
                    vault_wallet=vault_wallet,
                    owner_address=self.owner_address,
                    owner_private_key=self.owner_private_key,
                    wallet_password=self.vault_password
                )
                
                bt.logging.info(f"Deposit successful: {self.rao_to_theta(deposited_balance.rao)} Theta deposited to miner hotkey: {miner_hotkey} coldkey: {miner_coldkey}")
                print("succesfully deposited")
                return {
                    "successfully_processed": True,
                    "error_message": ""
                }
                
            except Exception as e:
                error_msg = f"Deposit execution failed: {str(e)}"
                bt.logging.error(error_msg)
                bt.logging.error(traceback.format_exc())
                return {
                    "successfully_processed": False,
                    "error_message": error_msg
                }
                
        except Exception as e:
            error_msg = f"Deposit processing error: {str(e)}"
            bt.logging.error(error_msg)
            bt.logging.error(traceback.format_exc())
            return {
                "successfully_processed": False,
                "error_message": error_msg
            }

    def process_withdrawal_request(self, amount: float, miner_address: str, vault_wallet: Wallet) -> Dict[str, Any]:
        """
        Process a collateral withdrawal request using raw data.
        
        Args:
            amount (float): Amount to withdraw in theta tokens
            miner_address (str): Miner's SS58 address
            
        Returns:
            Dict[str, Any]: Result of withdrawal operation
        """
        try:
            bt.logging.info("Received withdrawal request")
            # Check current collateral balance
            try:
                current_balance = self.collateral_manager.balance_of(miner_address)
                theta_current_balance = self.rao_to_theta(current_balance)
                if amount > theta_current_balance:
                    error_msg = f"Insufficient collateral balance. Available: {theta_current_balance}, Requested: {amount}"
                    bt.logging.error(error_msg)
                    return {
                        "successfully_processed": False,
                        "error_message": error_msg,
                        "returned_amount": 0.0
                    }
            except Exception as e:
                error_msg = f"Failed to check collateral balance: {str(e)}"
                bt.logging.error(error_msg)
                return {
                    "successfully_processed": False,
                    "error_message": error_msg,
                    "returned_amount": 0.0
                }
            
            # Execute the withdrawal through the collateral manager
            try:
                stake_list = self.collateral_manager.subtensor_api.staking.get_stake_for_coldkey(vault_wallet.coldkeypub.ss58_address)
                vault_stake = next(
                    (stake for stake in stake_list if stake.hotkey_ss58 == vault_wallet.hotkey.ss58_address),
                    None
                )

                bt.logging.info(f"Processing withdrawal request from {miner_address} for {amount} Theta")
                withdrawn_balance = self.collateral_manager.withdraw(
                    amount=self.theta_to_rao(amount),
                    dest=miner_address,
                    vault_stake=vault_stake.hotkey_ss58,
                    vault_wallet=vault_wallet,
                    owner_address=self.owner_address,
                    owner_private_key=self.owner_private_key,
                    wallet_password=self.vault_password
                )
                returned_theta = self.rao_to_theta(withdrawn_balance.rao)
                bt.logging.info(f"Withdrawal successful: {returned_theta} Theta withdrawn for {miner_address}")
                print(f"Withdrawal successful: {returned_theta} Theta withdrawn for {miner_address}")
                return {
                    "successfully_processed": True,
                    "error_message": "",
                    "returned_amount": returned_theta
                }
                
            except Exception as e:
                error_msg = f"Withdrawal execution failed: {str(e)}"
                bt.logging.error(error_msg)
                bt.logging.error(traceback.format_exc())
                return {
                    "successfully_processed": False,
                    "error_message": error_msg,
                    "returned_amount": 0.0
                }
                
        except Exception as e:
            error_msg = f"Withdrawal processing error: {str(e)}"
            bt.logging.error(error_msg)
            bt.logging.error(traceback.format_exc())
            return {
                "successfully_processed": False,
                "error_message": error_msg,
                "returned_amount": 0.0
            }
    
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
            return self.rao_to_theta(rao_balance)
        except Exception as e:
            bt.logging.error(f"Failed to get collateral balance for {miner_address}: {e}")
            return None

