from vali_objects.vali_config import ValiConfig, TradePairCategory
from vali_objects.vali_dataclasses.perf_ledger import PerfCheckpoint, PerfLedgerData

class Allocation():
    ### All allocation functions are defined against a single tradepair
    @staticmethod
    def hotkey_to_daycount(
            perf_ledger_bundles: dict[str, dict[str, list[PerfCheckpoint]]],
    ) -> dict[str, float]:
        """
        Returns the daycount associated with the trading behavior for each hotkey, based on their interaction with each asset
        """
        hotkey_daycounts = {}

        for hotkey, perf_ledger in perf_ledger_bundles.items():
            hotkey_allocations = Allocations.category_allocations(perf_ledger)
            hotkey_daycounts[hotkey] = Allocations.volume_normalized_to_daycount(hotkey_allocations)

        return hotkey_daycounts

    @staticmethod
    def category_allocations(
            perf_ledger_bundles: dict[str, list[PerfCheckpoint]],
    ) -> float:
        """
        Returns the allocations for each TradePairCategory
        """
        volume_dict = Allocations.transaction_volume(perf_ledger_bundles)
        normalized_volume_dict = Allocations.volume_normalization(volume_dict)

        return normalized_volume_dict

    @staticmethod
    def transaction_volume(
            perf_ledger_bundles: dict[str, list[PerfCheckpoint]],
    ) -> dict[str, float]:
        """
        Returns the volume for each TradePairCategory
        """
        volume_dict = {category.value: 0 for category in TradePairCategory}

        for bundle_name, perf_ledger in perf_ledger_bundles.items():
            for category in TradePairCategory:
                if bundle_name == category.value:
                    gain_volume = sum(
                        [perf_checkpoint.gain for perf_checkpoint in perf_ledger]
                    )
                    loss_volume = sum(
                        [perf_checkpoint.loss for perf_checkpoint in perf_ledger]
                    )

                    volume_dict += gain_volume + abs(loss_volume)

        return volume_dict

    @staticmethod
    def volume_normalization(
            volume_dict: dict[TradePairCategory, float]
    ) -> dict[TradePairCategory, float]:
        """
        Normalizes the volume_dict
        """
        volume_sum = sum(volume_dict.values())
        return {category: volume / volume_sum for category, volume in volume_dict.items()}

    @staticmethod
    def volume_normalized_to_daycount(
            volume_normalized: dict[TradePairCategory, float]
    ) -> dict[float]:
        """
        Normalizes the volume_dict
        """
        daycount_total = 0
        for category, normalized_volume in volume_normalized.items():
            daycount_total += normalized_volume * ValiConfig.DAYCOUNTS[category]

        return daycount_total / len(volume_normalized)
