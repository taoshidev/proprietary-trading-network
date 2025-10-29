#!/usr/bin/env python3
"""
Demonstration of aggressive payout strategy.

Shows how the payout timeline changes across different days of the month.
"""

from vali_objects.scoring.debt_based_scoring import DebtBasedScoring

print("=" * 80)
print("AGGRESSIVE PAYOUT STRATEGY DEMONSTRATION")
print("=" * 80)
print()
print("This shows how the payout timeline becomes more aggressive early in the month")
print("and tapers off as we approach the day 25 deadline.")
print()

# Show examples for different days
example_days = [1, 5, 10, 15, 20, 21, 22, 23, 24, 25]

print(f"{'Day':<6} {'Actual Days':<15} {'Aggressive':<15} {'Strategy':<30}")
print(f"{'':6} {'to Deadline':<15} {'Days Target':<15} {'':30}")
print("-" * 80)

for day in example_days:
    # Calculate actual days until deadline
    if day > DebtBasedScoring.PAYOUT_TARGET_DAY:
        actual_days = 0
    else:
        actual_days = DebtBasedScoring.PAYOUT_TARGET_DAY - day + 1

    # Apply aggressive strategy
    aggressive_days = min(actual_days, DebtBasedScoring.AGGRESSIVE_PAYOUT_BUFFER_DAYS)

    # Ensure at least 1 day if not past deadline
    if actual_days > 0 and aggressive_days == 0:
        aggressive_days = 1

    # Determine strategy description
    if aggressive_days < actual_days:
        strategy = "AGGRESSIVE (4-day buffer)"
    elif aggressive_days == actual_days and actual_days > 0:
        strategy = "Tapering (use actual remaining)"
    else:
        strategy = "Past deadline"

    print(f"{day:<6} {actual_days:<15} {aggressive_days:<15} {strategy:<30}")

print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()
print("Early Month (Days 1-20):")
print("  - Aggressive 4-day buffer creates urgency")
print("  - Example: On day 1, we project only 4 days of emissions")
print("  - This will likely show insufficient emissions warning")
print("  - Encourages front-loaded payouts")
print()
print("Late Month (Days 21-24):")
print("  - Taper to actual remaining days")
print("  - Example: On day 23, we project 3 days of emissions")
print("  - Reduces pressure as we approach deadline")
print()
print("Deadline Day (Day 25):")
print("  - No buffer remaining")
print("  - Pay out everything with whatever emissions are available")
print()
print("=" * 80)
print("EMISSION PROJECTION EXAMPLE")
print("=" * 80)
print()
print("Assuming subnet emits 0.822 TAO/block (5,920 TAO/day):")
print()

tao_per_day = 5920
alpha_to_tao_rate = 0.5  # 1 ALPHA = 0.5 TAO
alpha_per_day = tao_per_day / alpha_to_tao_rate

for day in [1, 10, 23, 25]:
    if day > DebtBasedScoring.PAYOUT_TARGET_DAY:
        actual_days = 0
    else:
        actual_days = DebtBasedScoring.PAYOUT_TARGET_DAY - day + 1

    aggressive_days = min(actual_days, DebtBasedScoring.AGGRESSIVE_PAYOUT_BUFFER_DAYS)
    if actual_days > 0 and aggressive_days == 0:
        aggressive_days = 1

    projected_alpha = alpha_per_day * aggressive_days

    print(f"Day {day:2d}: Projects {aggressive_days} days = {projected_alpha:,.0f} ALPHA")
    print(f"       (Actual days to deadline: {actual_days})")
    print()

print("If total remaining payout is 50,000 ALPHA:")
print("  - Day 1 projects 47,360 ALPHA (INSUFFICIENT - warning shown)")
print("  - Day 10 projects 47,360 ALPHA (INSUFFICIENT - warning shown)")
print("  - Day 23 projects 35,520 ALPHA (INSUFFICIENT - warning shown)")
print("  - Day 25 projects 11,840 ALPHA (INSUFFICIENT - warning shown)")
print()
print("This aggressive strategy ensures validators are warned early and often,")
print("creating urgency to pay miners throughout the month.")
print()
