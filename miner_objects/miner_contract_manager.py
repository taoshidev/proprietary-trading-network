import asyncio
import bittensor as bt
from bittensor_wallet import Wallet
from collateral_sdk import CollateralManager, Network
from template.protocol import DepositCollateral, WithdrawCollateral
from typing import Dict, Any, Optional
import traceback

class MinerContractManager:
    """
    Manages collateral contract interactions for miners.
    Handles extrinsic creation, collateral queries, and validator communication.
    All operations work with theta tokens, converting to/from RAO internally.
    """
    
    def __init__(self, wallet: Wallet, config, dendrite: bt.dendrite, metagraph):
        self.wallet = wallet
        self.config = config
        self.dendrite = dendrite
        self.metagraph = metagraph

        if config.subtensor.network == "test":
            self.network = Network.TESTNET
        else:
            self.network = Network.MAINNET
            
        self.collateral_manager = CollateralManager(self.network)
    
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
        rao_amount = int(tao_amount * 10**9)  # Convert TAO to RAO
        
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
        tao_amount = rao_amount / 10**9  # Convert RAO to TAO
        theta_amount = tao_amount / theta_price
        
        bt.logging.debug(f"Converted {tao_amount} TAO to {theta_amount} theta tokens")
        return theta_amount
        
    def get_collateral_balance(self) -> float:
        """
        Get miner's current collateral balance in theta tokens.
        
        Returns:
            float: Current collateral balance in theta tokens
        """
        try:
            rao_balance = self.collateral_manager.balance_of(
                self.wallet.coldkeypub.ss58_address
            )
            return self.rao_to_theta(rao_balance)
        except Exception as e:
            bt.logging.error(f"Failed to get collateral balance: {e}")
            return 0.0
    
    def create_deposit_extrinsic(self, theta_amount: float, validator_vault_address: str) -> bytes:
        """
        Create signed extrinsic for collateral deposit.
        
        Args:
            theta_amount (float): Amount to deposit in theta tokens
            validator_vault_address (str): Validator's vault SS58 address
            
        Returns:
            bytes: Encoded extrinsic ready for transmission
        """
        try:
            bt.logging.info(f"Creating deposit extrinsic for {theta_amount} theta tokens")

            source_stake = manager.subtensor_api.staking.get_stake_for_coldkey(wallet.coldkeypub.ss58_address)[0]
            
            extrinsic = self.collateral_manager.create_stake_transfer_extrinsic(
                amount=self.theta_to_rao(theta_amount),
                dest=validator_vault_address,
                source_stake=source_stake.hotkey_ss58,  # self.wallet.hotkey.ss58_address,
                source_wallet=self.wallet
            )
            return self.collateral_manager.encode_extrinsic(extrinsic)
        except Exception as e:
            bt.logging.error(f"Failed to create deposit extrinsic: {e}")
            raise
    
    async def send_deposit_request(self, validator_hotkey: str, theta_amount: float,
                                   validator_vault_address: str) -> Dict[str, Any]:
        """
        Send deposit request to specific validator.
        
        Args:
            validator_hotkey (str): Validator's hotkey address
            theta_amount (float): Amount to deposit in theta tokens
            validator_vault_address (str): Validator's vault address
            
        Returns:
            Dict[str, Any]: Result of deposit operation
        """
        try:
            bt.logging.info(f"Creating deposit extrinsic for {theta_amount} theta tokens to validator {validator_hotkey}")
            encoded_extrinsic = self.create_deposit_extrinsic(theta_amount, validator_vault_address)

            synapse = DepositCollateral(
                extrinsic_data=encoded_extrinsic.hex(),
                amount=theta_amount,
                miner_address=self.wallet.coldkeypub.ss58_address
            )
            
            bt.logging.info(f"Sending deposit request to validator {validator_hotkey}")
            validator_axons = [n.axon_info for n in self.metagraph.neurons if n.hotkey == validator_hotkey]
            response = await self.dendrite.aquery(
                axons=validator_axons,
                synapse=synapse,
                deserialize=False,
                timeout=30
            )
            
            if response and len(response) > 0:
                result = {
                    "success": response[0].successfully_processed,
                    "error": response[0].error_message,
                    "transaction_hash": response[0].transaction_hash,
                    "theta_amount": theta_amount,
                    "rao_amount": self.theta_to_rao(theta_amount)
                }
                bt.logging.info(f"Deposit request result: {result}")
                return result
            else:
                return {"success": False, "error": "No response from validator"}
                
        except Exception as e:
            error_msg = f"Failed to send deposit request: {e}"
            bt.logging.error(error_msg)
            bt.logging.error(traceback.format_exc())
            return {"success": False, "error": error_msg}
    
    async def send_withdrawal_request(self, validator_hotkey: str,
                                      theta_amount: float) -> Dict[str, Any]:
        """
        Send withdrawal request to validator.
        
        Args:
            validator_hotkey (str): Validator's hotkey address
            theta_amount (float): Amount to withdraw in theta tokens
            
        Returns:
            Dict[str, Any]: Result of withdrawal operation
        """
        try:
            rao_amount = self.theta_to_rao(theta_amount)
            bt.logging.info(f"Sending withdrawal request for {theta_amount} theta token to validator {validator_hotkey}")
            
            synapse = WithdrawCollateral(
                amount=theta_amount,
                miner_address=self.wallet.coldkeypub.ss58_address
            )

            validator_axons = [n.axon_info for n in self.metagraph.neurons if n.hotkey == validator_hotkey]
            response = await self.dendrite.aquery(
                axons=validator_axons,
                synapse=synapse,
                deserialize=False,
                timeout=30
            )
            
            if response and len(response) > 0:
                returned_rao = response[0].returned_amount
                returned_theta = self.rao_to_theta(returned_rao) if returned_rao > 0 else 0.0
                
                result = {
                    "success": response[0].successfully_processed,
                    "error": response[0].error_message,
                    "returned_amount_theta": returned_theta,
                    "returned_amount_rao": returned_rao,
                    "requested_theta": theta_amount,
                    "requested_rao": rao_amount
                }
                bt.logging.info(f"Withdrawal request result: {result}")
                return result
            else:
                return {"success": False, "error": "No response from validator"}
                
        except Exception as e:
            error_msg = f"Failed to send withdrawal request: {e}"
            bt.logging.error(error_msg)
            bt.logging.error(traceback.format_exc())
            return {"success": False, "error": error_msg}
