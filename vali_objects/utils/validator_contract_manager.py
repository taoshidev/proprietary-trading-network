import bittensor as bt
from bittensor_wallet import Wallet
from collateral_sdk import CollateralManager, Network
from template.protocol import DepositCollateral, WithdrawCollateral
from typing import Dict, Any, Optional, Tuple
import traceback
import os
from vali_objects.utils.vali_utils import ValiUtils

class ValidatorContractManager:
    """
    Manages collateral contract interactions for validators.
    Handles deposit processing, withdrawal validation, and EVM contract operations.
    This class acts as the validator's interface to the collateral system.
    """
    
    def __init__(self, config, wallet: Wallet, metagraph):
        self.config = config
        self.wallet = wallet
        self.metagraph = metagraph
        
        if config.subtensor.network == "test":
            self.network = Network.TESTNET
        else:
            self.network = Network.MAINNET
            
        self.collateral_manager = CollateralManager(self.network)
        
        # Load contract owner credentials from environment or config
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

        except Exception as e:
            bt.logging.error(f"Failed to load collateral owner credentials: {e}")
            self.owner_address = None
            self.owner_private_key = None


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
    
    def process_deposit_request(self, synapse: DepositCollateral) -> DepositCollateral:
        """
        Process a collateral deposit request from a miner.
        
        Args:
            synapse (DepositCollateral): The deposit request synapse
            
        Returns:
            DepositCollateral: Updated synapse with results
        """
        try:
            bt.logging.info(f"Processing deposit request from {synapse.miner_address} for {synapse.amount} RAO")
            
            # Decode and validate the extrinsic
            try:
                encoded_extrinsic = bytes.fromhex(synapse.extrinsic_data)
                extrinsic = self.collateral_manager.decode_extrinsic(encoded_extrinsic)
                bt.logging.debug("Extrinsic decoded successfully")
            except Exception as e:
                synapse.successfully_processed = False
                synapse.error_message = f"Invalid extrinsic data: {str(e)}"
                return synapse
            
            # Execute the deposit through the collateral manager
            try:
                vault_stake = manager.subtensor_api.staking.get_stake_for_coldkey(self.wallet.coldkeypub.ss58_address)[0]

                deposited_balance = self.collateral_manager.deposit(
                    extrinsic=extrinsic,
                    sender=synapse.miner_address,
                    vault_stake=vault_stake.hotkey_ss58,
                    vault_wallet=self.wallet,
                    owner_address=self.owner_address,
                    owner_private_key=self.owner_private_key
                )
                
                # todo: Get the transaction hash from the EVM/deposit operation
                # synapse.transaction_hash = "0x" + "0" * 64
                
                synapse.successfully_processed = True
                synapse.error_message = ""
                
                bt.logging.info(f"Deposit successful: {deposited_balance.rao} RAO deposited for {synapse.miner_address}")
                
            except Exception as e:
                synapse.successfully_processed = False
                synapse.error_message = f"Deposit execution failed: {str(e)}"
                bt.logging.error(f"Deposit execution failed: {e}")
                bt.logging.error(traceback.format_exc())
                
        except Exception as e:
            synapse.successfully_processed = False
            synapse.error_message = f"Deposit processing error: {str(e)}"
            bt.logging.error(f"Deposit processing error: {e}")
            bt.logging.error(traceback.format_exc())
            
        return synapse
    
    def process_withdrawal_request(self, synapse: WithdrawCollateral) -> WithdrawCollateral:
        """
        Process a collateral withdrawal request from a miner.
        
        Args:
            synapse (WithdrawCollateral): The withdrawal request synapse
            
        Returns:
            WithdrawCollateral: Updated synapse with results
        """
        try:
            bt.logging.info(f"Processing withdrawal request from {synapse.miner_address} for {synapse.amount} RAO")

            # Check current collateral balance
            try:
                current_balance = self.collateral_manager.balance_of(synapse.miner_address)
                if synapse.amount > current_balance:
                    synapse.successfully_processed = False
                    synapse.error_message = f"Insufficient collateral balance. Available: {current_balance}, Requested: {synapse.amount}"
                    return synapse
            except Exception as e:
                synapse.successfully_processed = False
                synapse.error_message = f"Failed to check collateral balance: {str(e)}"
                return synapse
            
            # Execute the withdrawal through the collateral manager
            try:
                vault_stake = manager.subtensor_api.staking.get_stake_for_coldkey(self.wallet.coldkeypub.ss58_address)[0]

                withdrawn_balance = self.collateral_manager.withdraw(
                    amount=self.theta_to_rao(synapse.amount),
                    dest=synapse.miner_address,
                    vault_stake=vault_stake.hotkey_ss58,
                    vault_wallet=self.wallet,
                    owner_address=self.owner_address,
                    owner_private_key=self.owner_private_key
                )
                
                synapse.returned_amount = self.rao_to_theta(withdrawn_balance.rao)
                synapse.successfully_processed = True
                synapse.error_message = ""
                
                bt.logging.info(f"Withdrawal successful: {self.rao_to_theta(withdrawn_balance.rao)} Theta withdrawn for {synapse.miner_address}")
                
            except Exception as e:
                synapse.successfully_processed = False
                synapse.error_message = f"Withdrawal execution failed: {str(e)}"
                bt.logging.error(f"Withdrawal execution failed: {e}")
                bt.logging.error(traceback.format_exc())
                
        except Exception as e:
            synapse.successfully_processed = False
            synapse.error_message = f"Withdrawal processing error: {str(e)}"
            bt.logging.error(f"Withdrawal processing error: {e}")
            bt.logging.error(traceback.format_exc())
            
        return synapse
