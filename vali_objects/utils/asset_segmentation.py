import copy
import os
import math

from vali_objects.vali_dataclasses.perf_ledger import PerfLedger, PerfCheckpoint
from vali_objects.vali_config import ValiConfig, TradePair, TradePairCategory

from vali_objects.utils.logger_utils import LoggerUtils


class AssetSegmentation:
    def __init__(self, miner_ledgers: dict[str, dict[str, PerfLedger]]):
        self.overall_ledgers = miner_ledgers
        self.asset_breakdown: dict[str, float] = ValiConfig.ASSET_CLASS_BREAKDOWN
        self.asset_classes: list[str] = list(self.asset_breakdown.keys())

    def segmentation(self, asset_class: str) -> dict[str, PerfLedger]:
        """
        Segments the overall ledgers into asset classes and aggregates them.
        Returns a dictionary where keys are asset classes and values are aggregated PerfLedgers.
        """
        if asset_class not in self.asset_classes:
            raise ValueError(f"Asset class {asset_class} is not recognized.")

        # Initialize segmented ledgers for the specified asset class
        subset = self.ledger_subset(asset_class)

        total_miner_ledgers = {}
        for hotkey, full_ledger in subset.items():
            portfolio_ledger = self.overall_ledgers.get(hotkey, {}).get("portfolio", PerfLedger())
            total_miner_ledgers[hotkey] = AssetSegmentation.aggregate_miner_subledgers(
                portfolio_ledger,
                full_ledger,
            )

        return total_miner_ledgers

    def ledger_subset(self, asset_class: str) -> dict[str, dict[str, PerfLedger]]:
        """
        Only returns the subset of ledgers that match the specified asset class.
        """
        if asset_class not in self.asset_classes:
            raise ValueError(f"Asset class {asset_class} is not recognized.")

        subset_ledger = {}
        for hotkey, full_ledger in self.overall_ledgers.items():
            miner_subset_ledger = {}
            for asset_name, ledger in full_ledger.items():
                if asset_name == "portfolio":
                    continue

                trade_pair = TradePair.from_trade_pair_id(asset_name)
                trade_pair_category = trade_pair.trade_pair_category
                if trade_pair_category == asset_class:
                    miner_subset_ledger[asset_name] = ledger

            subset_ledger[hotkey] = miner_subset_ledger

        return subset_ledger

    @staticmethod
    def aggregate_miner_subledgers(
            default_ledger: PerfLedger,
            sub_ledgers: dict[str, PerfLedger]
    ) -> PerfLedger:
        """
        Aggregates the sub-ledgers for a specific miner into a single PerfLedger.
        """
        if len(sub_ledgers) == 0:
            return PerfLedger()

        default_ledger_copy = copy.deepcopy(default_ledger)
        default_ledger_copy.cps: list[PerfCheckpoint] = []

        aggregated_dict_ledger = {}
        for _, ledger in sub_ledgers.items():
            ledger_checkpoints = ledger.cps
            for checkpoint in ledger_checkpoints:
                if checkpoint.last_update_ms not in aggregated_dict_ledger:
                    aggregated_dict_ledger[checkpoint.last_update_ms] = copy.deepcopy(checkpoint)
                else:
                    existing_checkpoint = aggregated_dict_ledger.get(checkpoint.last_update_ms)

                    # Aggregate the values of the existing checkpoint with the new one
                    existing_checkpoint.n_updates += checkpoint.n_updates
                    existing_checkpoint.gain += checkpoint.gain
                    existing_checkpoint.loss += checkpoint.loss
                    existing_checkpoint.spread_fee_loss += checkpoint.spread_fee_loss
                    existing_checkpoint.carry_fee_loss += checkpoint.carry_fee_loss

                    aggregated_dict_ledger[checkpoint.last_update_ms] = existing_checkpoint

        default_ledger_copy.cps = sorted(list(aggregated_dict_ledger.values()), key=lambda x: x.last_update_ms)
        return default_ledger_copy

    @staticmethod
    def segment_competitiveness(incentive_distribution: list[float]) -> float:
        """
        Indicates the relative level of competitiveness for an asset class based on the incentive distribution.
        """
        # Placeholder logic for competitiveness segmentation
        vals = incentive_distribution
        n = len(vals)
        if n == 0:
            return math.nan
        if any(v < 0 for v in vals):
            raise ValueError("Gini coefficient is undefined for negative values")

        vals.sort()
        cumulative = 0.0
        cumulative_weighted = 0.0
        for i, v in enumerate(vals, 1):
            cumulative += v
            cumulative_weighted += i * v

        total = cumulative
        g = (2 * cumulative_weighted) / (n * total) - (n + 1) / n
        return g

    @staticmethod
    def asset_competitiveness_dictionary(
            asset_incentive_distributions: dict[str, list[tuple[str, float]]]
    ) -> dict[str, float]:
        """
        Returns a dictionary with asset classes as keys and their competitiveness as values.
        """
        competitiveness_dict = {}
        for asset_class, distribution in asset_incentive_distributions.items():
            if not distribution:
                competitiveness_dict[asset_class] = math.nan
            else:
                incentive_distribution = [value for _, value in distribution]
                competitiveness_dict[asset_class] = AssetSegmentation.segment_competitiveness(incentive_distribution)

        return competitiveness_dict
