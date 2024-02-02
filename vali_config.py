# developer: Taoshi
# Copyright © 2023 Taoshi Inc


class ValiConfig:
    # fees take into account exiting and entering a position, liquidity, and futures fees
    TRADE_PAIR_FEES = {
        "BTC/USD": 0.3,
        "ETH/USD": 0.3
    }
    MIN_LEVERAGE = 0.001
    MAX_DAILY_DRAWDOWN = 0.05
    MAX_TOTAL_DRAWDOWN = 0.1
    MAX_ORDERS = 200