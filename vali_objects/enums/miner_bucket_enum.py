from dataclasses import dataclass
from enum import Enum

from vali_objects.vali_config import ValiConfig


class MinerBucket(Enum):
    MAINCOMP = "MAINCOMP"
    CHALLENGE = "CHALLENGE"
    PROBATION = "PROBATION"
    PLAGIARISM = "PLAGIARISM"
    ELIMINATED = "ELIMINATED"
    UNKNOWN = "unknown"
    # Entity system buckets
    ENTITY = "ENTITY"
    SUBACCOUNT_CHALLENGE = "SUBACCOUNT_CHALLENGE"
    SUBACCOUNT_FUNDED = "SUBACCOUNT_FUNDED"
    SUBACCOUNT_ALPHA = "SUBACCOUNT_ALPHA"
    # Pro account buckets
    # A funded standard subaccount moving onto the pro track. Entered with in_transition set (see
    # BucketPhase): a wind-down week still trading the standard account. Clearing the flag moves it
    # onto the pro account, where it is paid on the standard size while passing the pro challenge.
    PRO_CHALLENGE_FUNDED = "PRO_CHALLENGE_FUNDED"
    PRO_CHALLENGE_DIRECT = "PRO_CHALLENGE_DIRECT"
    PRO_FUNDED = "PRO_FUNDED"

    @classmethod
    def _missing_(cls, value):
        """Parse the names PRO_CHALLENGE_FUNDED replaced. Drops the transition phase the old name
        carried; read it with BucketPhase.from_status where it matters."""
        if value in LEGACY_PRO_CHALLENGE_FUNDED_PHASES:
            return cls.PRO_CHALLENGE_FUNDED
        return None

    def intraday_drawdown_threshold(self, time_ms: int | None = None) -> float:
        """
        Returns the intraday drawdown threshold for this bucket from ValiConfig.
        For SUBACCOUNT_FUNDED (and the PRO_CHALLENGE_FUNDED transition, which keeps its rules - see
        BucketPhase), time_ms is the miner's SUBACCOUNT_CHALLENGE registration timestamp and
        determines which versioned threshold applies. Pro buckets are not versioned - they have no
        legacy tier.
        """
        if self in (MinerBucket.PRO_CHALLENGE_FUNDED, MinerBucket.PRO_CHALLENGE_DIRECT):
            return ValiConfig.PRO_CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD

        if self == MinerBucket.PRO_FUNDED:
            return ValiConfig.PRO_FUNDED_INTRADAY_DRAWDOWN_THRESHOLD

        if self in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.CHALLENGE):
            return ValiConfig.CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD

        if self == MinerBucket.SUBACCOUNT_FUNDED:
            if time_ms is not None and time_ms < ValiConfig.FUNDED_V0_CUTOFF_MS:
                return ValiConfig.FUNDED_INTRADAY_DRAWDOWN_THRESHOLD_V0
            if time_ms is not None and time_ms < ValiConfig.FUNDED_V1_CUTOFF_MS:
                return ValiConfig.FUNDED_INTRADAY_DRAWDOWN_THRESHOLD_V1
            return ValiConfig.FUNDED_INTRADAY_DRAWDOWN_THRESHOLD

        if self in (MinerBucket.MAINCOMP, MinerBucket.PROBATION, MinerBucket.SUBACCOUNT_ALPHA, MinerBucket.PLAGIARISM):
            return ValiConfig.FUNDED_INTRADAY_DRAWDOWN_THRESHOLD

        raise ValueError(f"No intraday drawdown threshold defined for bucket {self}")

    def eod_drawdown_threshold(self, time_ms: int | None = None) -> float:
        """
        Returns the intraday drawdown threshold for this bucket from ValiConfig.
        For SUBACCOUNT_FUNDED (and the PRO_CHALLENGE_FUNDED transition), time_ms is the miner's
        SUBACCOUNT_CHALLENGE registration timestamp and determines which versioned threshold
        applies.
        """
        if self in (MinerBucket.PRO_CHALLENGE_FUNDED, MinerBucket.PRO_CHALLENGE_DIRECT):
            return ValiConfig.PRO_CHALLENGE_EOD_DRAWDOWN_THRESHOLD

        if self == MinerBucket.PRO_FUNDED:
            return ValiConfig.PRO_FUNDED_EOD_DRAWDOWN_THRESHOLD

        if self in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.CHALLENGE):
            return ValiConfig.CHALLENGE_EOD_DRAWDOWN_THRESHOLD

        if self == MinerBucket.SUBACCOUNT_FUNDED:
            if time_ms is not None and time_ms < ValiConfig.FUNDED_V0_CUTOFF_MS:
                return ValiConfig.FUNDED_EOD_DRAWDOWN_THRESHOLD_V0
            return ValiConfig.FUNDED_EOD_DRAWDOWN_THRESHOLD

        if self in (MinerBucket.MAINCOMP, MinerBucket.PROBATION, MinerBucket.SUBACCOUNT_ALPHA, MinerBucket.PLAGIARISM):
            return ValiConfig.FUNDED_EOD_DRAWDOWN_THRESHOLD

        raise ValueError(f"No intraday drawdown threshold defined for bucket {self}")

    @property
    def is_regular_miner(self) -> bool:
        return self in (MinerBucket.CHALLENGE, MinerBucket.PROBATION, MinerBucket.MAINCOMP)

    @property
    def is_subaccount(self) -> bool:
        return self in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_ALPHA,
                        MinerBucket.PRO_CHALLENGE_FUNDED, MinerBucket.PRO_CHALLENGE_DIRECT,
                        MinerBucket.PRO_FUNDED)

    @property
    def is_pro(self) -> bool:
        """True for buckets trading the pro account. The PRO_CHALLENGE_FUNDED transition is still on
        the standard account and so is not pro - read it through BucketPhase."""
        return self in (MinerBucket.PRO_CHALLENGE_FUNDED, MinerBucket.PRO_CHALLENGE_DIRECT,
                        MinerBucket.PRO_FUNDED)

    @property
    def is_pro_track(self) -> bool:
        """True for every bucket on the pro journey, including the transition week."""
        return self.is_pro

    @property
    def calmar_threshold(self) -> float | None:
        """Minimum all-time calmar required for promotion. None for buckets with no calmar requirement."""
        return ValiConfig.PRO_CHALLENGE_CALMAR_THRESHOLD if self.is_pro else None

    @property
    def daily_consistency_threshold(self) -> float | None:
        """Maximum return consistency allowed for promotion. None for buckets with no consistency requirement."""
        return ValiConfig.PRO_CHALLENGE_DAILY_CONSISTENCY_THRESHOLD if self.is_pro else None

    @property
    def minimum_trading_days(self) -> int | None:
        """Full days of tracked returns required before promotion. None for buckets with no day requirement."""
        return ValiConfig.PRO_CHALLENGE_MINIMUM_DAYS if self.is_pro else None

    def returns_threshold(self, asset_class) -> float:
        """Return required for promotion out of this bucket, by the miner's asset class."""
        if self.is_pro:
            return ValiConfig.PRO_CHALLENGE_RETURNS_THRESHOLD.get(
                asset_class, ValiConfig.PRO_CHALLENGE_RETURNS_THRESHOLD_DEFAULT)
        return ValiConfig.SUBACCOUNT_CHALLENGE_RETURNS_THRESHOLD.get(
            asset_class, ValiConfig.SUBACCOUNT_CHALLENGE_RETURNS_THRESHOLD_DEFAULT)

    @property
    def soft_breach_applies(self) -> bool:
        """True for buckets where a pro-rule breach withholds the week's payout.
        Miners who already passed the standard challenge keep earning through a soft breach."""
        return self == MinerBucket.PRO_FUNDED

    @property
    def payout_scale_applies(self) -> bool:
        """True for buckets whose PnL is scaled by standard_account_size / pro_account_size."""
        return self == MinerBucket.PRO_CHALLENGE_FUNDED

    @property
    def is_subaccount_challenge(self) -> bool:
        """True for a subaccount's challenge bucket on either track."""
        return self in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.PRO_CHALLENGE_FUNDED,
                        MinerBucket.PRO_CHALLENGE_DIRECT)

    @property
    def is_subaccount_funded(self) -> bool:
        """True for either subaccount track's funded buckets."""
        return self in (MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.PRO_FUNDED)

    @property
    def is_subaccount_earning(self) -> bool:
        """True for subaccount buckets that earn payouts, carry margin requirements, and are
        subject to entity collateral slashing."""
        return self.is_subaccount_funded or self in (
            MinerBucket.SUBACCOUNT_ALPHA,
            MinerBucket.PRO_CHALLENGE_FUNDED,
        )

    @property
    def next_bucket(self) -> "MinerBucket | None":
        if self == MinerBucket.CHALLENGE:
            return MinerBucket.MAINCOMP
        elif self == MinerBucket.PROBATION:
            return MinerBucket.MAINCOMP
        elif self == MinerBucket.SUBACCOUNT_CHALLENGE:
            return MinerBucket.SUBACCOUNT_FUNDED
        elif self in (MinerBucket.PRO_CHALLENGE_FUNDED, MinerBucket.PRO_CHALLENGE_DIRECT):
            return MinerBucket.PRO_FUNDED
        # TODO determine if we need alpha or keep subaccounts as funded
        # elif self == MinerBucket.SUBACCOUNT_FUNDED:
        #     return MinerBucket.SUBACCOUNT_ALPHA
        return None

    @property
    def promotion_target(self) -> "MinerBucket | None":
        """Where a miner may promote itself with its own signed request (POST /entity/subaccount/promote).

        These hops onto the pro track are the only self-service bucket moves on the network; every
        other bucket returns None and cannot be promoted this way. SUBACCOUNT_FUNDED enters
        PRO_CHALLENGE_FUNDED in transition; leaving the transition early is BucketPhase.promotion_target.
        """
        if self == MinerBucket.SUBACCOUNT_CHALLENGE:
            return MinerBucket.PRO_CHALLENGE_DIRECT
        elif self == MinerBucket.SUBACCOUNT_FUNDED:
            return MinerBucket.PRO_CHALLENGE_FUNDED
        return None

    @property
    def switches_account(self) -> bool:
        """True when moving *into* this bucket requires closing
        positions, cancelling limit orders, and restarting the ledgers.

        PRO_FUNDED is deliberately absent: passing the pro challenge keeps the same account, so
        balance, equity, positions and the ledgers carry over and all-time calmar keeps the ratio
        it passed with. Nothing earned before the promotion is paid at pro scale - the payout paths
        exclude it by reading each checkpoint's own bucket instead of wiping the history.

        Entering the PRO_CHALLENGE_FUNDED transition keeps the standard account; this describes
        the pro phase. Use BucketPhase.switches_account for the phase being entered.
        """
        return self in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.SUBACCOUNT_FUNDED,
                        MinerBucket.PRO_CHALLENGE_FUNDED, MinerBucket.PRO_CHALLENGE_DIRECT)

    @property
    def max_time_ms(self) -> int | None:
        if self == MinerBucket.CHALLENGE:
            return ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS
        elif self == MinerBucket.PROBATION:
            return ValiConfig.PROBATION_MAXIMUM_MS
        elif self == MinerBucket.PLAGIARISM:
            return   ValiConfig.PLAGIARISM_REVIEW_PERIOD_MS
        else:
            return None

    @property
    def is_rank_based(self):
        return self in (
                MinerBucket.CHALLENGE,
                MinerBucket.MAINCOMP,
                MinerBucket.PROBATION,
                MinerBucket.SUBACCOUNT_ALPHA
                )

    @property
    def is_active(self):
        return self in (
                MinerBucket.CHALLENGE,
                MinerBucket.MAINCOMP,
                MinerBucket.PROBATION,
                MinerBucket.PLAGIARISM,
                MinerBucket.SUBACCOUNT_CHALLENGE,
                MinerBucket.SUBACCOUNT_FUNDED,
                MinerBucket.SUBACCOUNT_ALPHA,
                MinerBucket.PRO_CHALLENGE_FUNDED,
                MinerBucket.PRO_CHALLENGE_DIRECT,
                MinerBucket.PRO_FUNDED
                )

    def phase(self, in_transition: bool = False) -> "BucketPhase":
        return BucketPhase(self, in_transition)


# Names PRO_CHALLENGE_FUNDED replaced, mapped to the in_transition flag each one meant. Still read
# from checkpoints, disk state, ledger checkpoint statuses and request bodies written before the merge.
LEGACY_PRO_CHALLENGE_FUNDED_PHASES = {
    "PRO_CHALLENGE_TRANSITION": True,
    "PRO_CHALLENGE_FROM_STANDARD": False,
}

# Appended to the bucket value when a ledger checkpoint status is stamped during the transition
TRANSITION_STATUS_SUFFIX = ":in_transition"


@dataclass(frozen=True)
class BucketPhase:
    """A bucket together with the PRO_CHALLENGE_FUNDED in_transition flag.

    In transition, a PRO_CHALLENGE_FUNDED subaccount is still winding down its standard funded
    account: it trades and is paid under SUBACCOUNT_FUNDED rules (standard leverage, fees, trade
    pairs and versioned funded drawdown limits, no payout scale), may not add exposure, and must
    leave the transition by the first Monday 00:00 UTC after entering it. Leaving the transition
    switches onto the pro account. Out of transition every property reads straight off the bucket.

    The flag is meaningless for any other bucket and is normalized to False.
    """
    bucket: MinerBucket
    in_transition: bool = False

    def __post_init__(self):
        if self.in_transition and self.bucket != MinerBucket.PRO_CHALLENGE_FUNDED:
            object.__setattr__(self, "in_transition", False)

    @property
    def _rules(self) -> MinerBucket:
        """The bucket whose trading and payout rules apply in this phase."""
        return MinerBucket.SUBACCOUNT_FUNDED if self.in_transition else self.bucket

    @property
    def value(self) -> str:
        return self.bucket.value

    @property
    def status(self) -> str:
        """Ledger checkpoint status: the bucket value, suffixed while in transition."""
        return self.bucket.value + (TRANSITION_STATUS_SUFFIX if self.in_transition else "")

    @classmethod
    def from_status(cls, status: str | None) -> "BucketPhase":
        """Parse a status written by `status`, a plain bucket value, or a legacy bucket name."""
        if not status:
            return cls(MinerBucket.UNKNOWN)
        in_transition = status.endswith(TRANSITION_STATUS_SUFFIX)
        if in_transition:
            status = status[:-len(TRANSITION_STATUS_SUFFIX)]
        in_transition = LEGACY_PRO_CHALLENGE_FUNDED_PHASES.get(status, in_transition)
        try:
            bucket = MinerBucket(status)
        except ValueError:
            bucket = MinerBucket.UNKNOWN
        return cls(bucket, in_transition)

    @staticmethod
    def statuses_of(bucket: MinerBucket) -> set[str]:
        """Every ledger checkpoint status a bucket may have been stamped with, legacy names included."""
        statuses = {bucket.value}
        if bucket == MinerBucket.PRO_CHALLENGE_FUNDED:
            statuses.add(BucketPhase(bucket, True).status)
            statuses.update(LEGACY_PRO_CHALLENGE_FUNDED_PHASES)
        return statuses

    def __str__(self) -> str:
        return self.status

    # Properties that differ between the transition and the pro phase

    @property
    def is_pro(self) -> bool:
        return self._rules.is_pro

    @property
    def is_subaccount_challenge(self) -> bool:
        return self._rules.is_subaccount_challenge

    @property
    def is_subaccount_funded(self) -> bool:
        return self._rules.is_subaccount_funded

    @property
    def payout_scale_applies(self) -> bool:
        return self._rules.payout_scale_applies

    @property
    def soft_breach_applies(self) -> bool:
        return self._rules.soft_breach_applies

    @property
    def calmar_threshold(self) -> float | None:
        return self._rules.calmar_threshold

    @property
    def daily_consistency_threshold(self) -> float | None:
        return self._rules.daily_consistency_threshold

    @property
    def minimum_trading_days(self) -> int | None:
        return self._rules.minimum_trading_days

    def returns_threshold(self, asset_class) -> float:
        return self._rules.returns_threshold(asset_class)

    def intraday_drawdown_threshold(self, time_ms: int | None = None) -> float:
        return self._rules.intraday_drawdown_threshold(time_ms)

    def eod_drawdown_threshold(self, time_ms: int | None = None) -> float:
        return self._rules.eod_drawdown_threshold(time_ms)

    @property
    def uses_versioned_funded_thresholds(self) -> bool:
        """True when drawdown limits are versioned by the SUBACCOUNT_CHALLENGE registration time."""
        return self._rules == MinerBucket.SUBACCOUNT_FUNDED

    @property
    def switches_account(self) -> bool:
        """True when entering this phase closes positions and restarts the ledgers. Entering the
        transition keeps the standard account; leaving it switches onto the pro account."""
        return not self.in_transition and self.bucket.switches_account

    @property
    def next_phase(self) -> "BucketPhase | None":
        """Where the refresh loop promotes this phase to. The transition's exit is its Monday
        deadline (or a self-service promotion), never its returns."""
        if self.in_transition:
            return BucketPhase(MinerBucket.PRO_CHALLENGE_FUNDED)
        nxt = self.bucket.next_bucket
        return BucketPhase(nxt) if nxt is not None else None

    @property
    def promotion_target(self) -> "BucketPhase | None":
        """Self-service promotion target (POST /entity/subaccount/promote). SUBACCOUNT_FUNDED
        enters PRO_CHALLENGE_FUNDED in transition; a subaccount in transition may leave it early."""
        if self.in_transition:
            return BucketPhase(MinerBucket.PRO_CHALLENGE_FUNDED)
        target = self.bucket.promotion_target
        if target is None:
            return None
        return BucketPhase(target, in_transition=target == MinerBucket.PRO_CHALLENGE_FUNDED)

    # Properties shared by both phases

    @property
    def is_subaccount(self) -> bool:
        return self.bucket.is_subaccount

    @property
    def is_pro_track(self) -> bool:
        return self.bucket.is_pro_track

    @property
    def is_subaccount_earning(self) -> bool:
        return self.bucket.is_subaccount_earning

    @property
    def is_active(self) -> bool:
        return self.bucket.is_active

@dataclass
class BucketEntry:
    bucket: MinerBucket
    start_time_ms: int
    # When set, this span was undone (an elimination that was reverted). The entry is kept as the
    # record that it happened, but point-in-time lookups skip it: bucket history is replayed to
    # stamp historical checkpoints, so leaving a reverted span visible would permanently reclassify
    # every checkpoint inside it and change payout weeks that were already settled.
    reverted_ms: int | None = None
    # PRO_CHALLENGE_FUNDED only: this span is the transition week (see BucketPhase). Leaving the
    # transition appends a new PRO_CHALLENGE_FUNDED entry with the flag cleared.
    in_transition: bool = False

    def __post_init__(self):
        self.in_transition = BucketPhase(self.bucket, self.in_transition).in_transition

    @property
    def is_reverted(self) -> bool:
        return self.reverted_ms is not None

    @property
    def phase(self) -> BucketPhase:
        return BucketPhase(self.bucket, self.in_transition)

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            'bucket': self.bucket.value,
            'start_time_ms': self.start_time_ms,
            'bucket_start_time': self.start_time_ms,  # for backwards compatibility TODO remove
            'reverted_ms': self.reverted_ms,
            'in_transition': self.in_transition,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'BucketEntry':
        """Create from dict for deserialization. Entries written before the merge carry
        PRO_CHALLENGE_TRANSITION or PRO_CHALLENGE_FROM_STANDARD, which set the flag."""
        bucket = d.get('bucket', MinerBucket.UNKNOWN)
        in_transition = bool(d.get('in_transition', False))
        if isinstance(bucket, str):
            in_transition = LEGACY_PRO_CHALLENGE_FUNDED_PHASES.get(bucket, in_transition)
            bucket = MinerBucket(bucket)
        return cls(
            bucket=bucket,
            start_time_ms=d.get('start_time_ms') or d.get('bucket_start_time', 0),
            reverted_ms=d.get('reverted_ms'),
            in_transition=in_transition,
        )
