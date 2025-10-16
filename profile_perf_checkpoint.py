#!/usr/bin/env python3
"""
Profiling script to compare PerfCheckpoint with and without BaseModel inheritance.
Tests 1 million updates to measure performance impact.
"""

import time
import statistics
from pydantic import BaseModel, ConfigDict


# Original version with BaseModel (for comparison)
class PerfCheckpointBaseModel(BaseModel):
    last_update_ms: int
    prev_portfolio_ret: float
    prev_portfolio_spread_fee: float = 1.0
    prev_portfolio_carry_fee: float = 1.0
    accum_ms: int = 0
    open_ms: int = 0
    n_updates: int = 0
    gain: float = 0.0
    loss: float = 0.0
    spread_fee_loss: float = 0.0
    carry_fee_loss: float = 0.0
    mdd: float = 1.0
    mpv: float = 0.0
    pnl_gain: float = 0.0
    pnl_loss: float = 0.0

    model_config = ConfigDict(extra="allow")

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return self.__dict__


# Simple version without BaseModel
class PerfCheckpointSimple:
    def __init__(
        self,
        last_update_ms: int,
        prev_portfolio_ret: float,
        prev_portfolio_spread_fee: float = 1.0,
        prev_portfolio_carry_fee: float = 1.0,
        accum_ms: int = 0,
        open_ms: int = 0,
        n_updates: int = 0,
        gain: float = 0.0,
        loss: float = 0.0,
        spread_fee_loss: float = 0.0,
        carry_fee_loss: float = 0.0,
        mdd: float = 1.0,
        mpv: float = 0.0,
        pnl_gain: float = 0.0,
        pnl_loss: float = 0.0,
    ):
        self.last_update_ms = last_update_ms
        self.prev_portfolio_ret = prev_portfolio_ret
        self.prev_portfolio_spread_fee = prev_portfolio_spread_fee
        self.prev_portfolio_carry_fee = prev_portfolio_carry_fee
        self.accum_ms = accum_ms
        self.open_ms = open_ms
        self.n_updates = n_updates
        self.gain = gain
        self.loss = loss
        self.spread_fee_loss = spread_fee_loss
        self.carry_fee_loss = carry_fee_loss
        self.mdd = mdd
        self.mpv = mpv
        self.pnl_gain = pnl_gain
        self.pnl_loss = pnl_loss

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return self.__dict__


def profile_updates(checkpoint_class, n_iterations=1_000_000, name="Unknown"):
    """Profile attribute updates on a checkpoint object."""
    print(f"\n{'=' * 60}")
    print(f"Profiling {name}")
    print(f"{'=' * 60}")

    # Create instance
    start = time.perf_counter()
    obj = checkpoint_class(
        last_update_ms=1000000,
        prev_portfolio_ret=1.0,
    )
    creation_time = time.perf_counter() - start
    print(f"Object creation time: {creation_time * 1000:.4f} ms")

    # Test attribute updates
    start = time.perf_counter()
    for i in range(n_iterations):
        obj.last_update_ms += 1000
        obj.prev_portfolio_ret *= 1.001
        obj.gain += 0.01
        obj.loss += 0.005
        obj.n_updates += 1
        obj.accum_ms += 1000
        obj.mdd = min(obj.mdd, obj.prev_portfolio_ret)
        obj.mpv = max(obj.mpv, obj.prev_portfolio_ret)

    update_time = time.perf_counter() - start
    print(f"Total update time: {update_time:.4f} seconds")
    print(f"Average time per update: {(update_time / n_iterations) * 1_000_000:.4f} µs")
    print(f"Updates per second: {n_iterations / update_time:,.0f}")

    # Test attribute reads
    start = time.perf_counter()
    for i in range(n_iterations):
        _ = obj.last_update_ms
        _ = obj.prev_portfolio_ret
        _ = obj.gain
        _ = obj.loss
        _ = obj.n_updates

    read_time = time.perf_counter() - start
    print(f"Total read time: {read_time:.4f} seconds")
    print(f"Average time per read: {(read_time / n_iterations) * 1_000_000:.4f} µs")
    print(f"Reads per second: {n_iterations / read_time:,.0f}")

    # Test to_dict() calls
    start = time.perf_counter()
    for i in range(10_000):
        _ = obj.to_dict()

    dict_time = time.perf_counter() - start
    print(f"Total to_dict() time (10k calls): {dict_time:.4f} seconds")
    print(f"Average time per to_dict(): {(dict_time / 10_000) * 1_000_000:.4f} µs")

    return {
        "creation_time": creation_time,
        "update_time": update_time,
        "read_time": read_time,
        "dict_time": dict_time,
        "updates_per_sec": n_iterations / update_time,
        "reads_per_sec": n_iterations / read_time,
    }


def main():
    print("=" * 60)
    print("PerfCheckpoint Performance Profiling")
    print("=" * 60)
    print(f"Number of iterations: 1,000,000")

    # Profile BaseModel version
    results_basemodel = profile_updates(
        PerfCheckpointBaseModel,
        n_iterations=1_000_000,
        name="PerfCheckpoint (BaseModel - old)"
    )

    # Profile simple version (no type coercion)
    results_simple = profile_updates(
        PerfCheckpointSimple,
        n_iterations=1_000_000,
        name="PerfCheckpoint (Plain Class - no coercion)"
    )

    # Profile production version (from actual codebase)
    from vali_objects.vali_dataclasses.perf_ledger import PerfCheckpoint
    results_production = profile_updates(
        PerfCheckpoint,
        n_iterations=1_000_000,
        name="PerfCheckpoint (Production - with coercion)"
    )

    # Compare results
    print(f"\n{'=' * 60}")
    print("Performance Comparison")
    print(f"{'=' * 60}")

    print(f"\nCreation time:")
    print(f"  BaseModel (old):        {results_basemodel['creation_time'] * 1000:.4f} ms")
    print(f"  Simple (no coercion):   {results_simple['creation_time'] * 1000:.4f} ms")
    print(f"  Production (coercion):  {results_production['creation_time'] * 1000:.4f} ms")

    print(f"\nUpdate time (1M operations):")
    print(f"  BaseModel (old):        {results_basemodel['update_time']:.4f} seconds")
    print(f"  Simple (no coercion):   {results_simple['update_time']:.4f} seconds")
    print(f"  Production (coercion):  {results_production['update_time']:.4f} seconds")

    print(f"\nRead time (1M operations):")
    print(f"  BaseModel (old):        {results_basemodel['read_time']:.4f} seconds")
    print(f"  Simple (no coercion):   {results_simple['read_time']:.4f} seconds")
    print(f"  Production (coercion):  {results_production['read_time']:.4f} seconds")

    print(f"\nto_dict() time (10k operations):")
    print(f"  BaseModel (old):        {results_basemodel['dict_time']:.4f} seconds")
    print(f"  Simple (no coercion):   {results_simple['dict_time']:.4f} seconds")
    print(f"  Production (coercion):  {results_production['dict_time']:.4f} seconds")

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    baseline_total = results_basemodel['update_time'] + results_basemodel['read_time']
    production_total = results_production['update_time'] + results_production['read_time']
    speedup_vs_basemodel = baseline_total / production_total

    print(f"Production vs BaseModel speedup: {speedup_vs_basemodel:.2f}x")
    print(f"Time saved per 1M ops: {(baseline_total - production_total):.2f}s")

    if speedup_vs_basemodel > 5:
        print("\n✅ Significant performance improvement!")
    elif speedup_vs_basemodel > 2:
        print("\n✅ Good performance improvement")
    else:
        print("\n⚠️  Modest performance improvement")


if __name__ == "__main__":
    main()
