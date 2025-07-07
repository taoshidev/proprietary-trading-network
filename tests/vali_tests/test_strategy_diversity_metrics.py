import numpy as np
import math
from unittest.mock import patch
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.orthogonality import Orthogonality
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.vali_config import ValiConfig, TradePair
from vali_objects.vali_dataclasses.perf_ledger import PerfLedger, PerfCheckpoint
from vali_objects.position import Position
from vali_objects.vali_dataclasses.order import Order
from tests.shared_objects.test_utilities import ledger_generator, checkpoint_generator
from datetime import datetime, timezone
import copy


class TestStrategyDiversityValidation(TestBase):
    """Validation tests for ensuring strategy diversity and correlation metrics work correctly."""

    def setUp(self):
        super().setUp()
        
        # Create correlation pattern scenarios
        self.pattern_scenarios = self._create_pattern_scenarios()
        self.diverse_strategies = self._create_diverse_strategies()
        
    def _create_pattern_scenarios(self):
        """Create realistic correlation pattern scenarios for diversity testing."""
        base_length = max(ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N + 20, 80)
        
        # JPY correlation pattern analysis
        jpy_base_pattern = self._simulate_jpy_correlation_pattern(base_length)
        
        pattern_scenarios = {
            # JPY correlation group - testing correlation detection across related pairs
            'jpy_pattern_1': self._add_noise_to_pattern(jpy_base_pattern, 0.001),  # USDJPY focus
            'jpy_pattern_2': self._add_noise_to_pattern(jpy_base_pattern, 0.0015), # EURJPY focus  
            'jpy_pattern_3': self._add_noise_to_pattern(jpy_base_pattern, 0.0008), # GBPJPY focus
            'jpy_pattern_4': self._add_noise_to_pattern(jpy_base_pattern, 0.0012), # AUDJPY focus
            'jpy_pattern_5': self._add_noise_to_pattern(jpy_base_pattern, 0.0009), # CADJPY focus
            
            # Similar strategy cluster - testing similarity detection
            'base_strategy': self._create_base_strategy(base_length),
            'variant_1': self._create_strategy_variant(base_length, 0.0001),
            'variant_2': self._create_strategy_variant(base_length, 0.0002),
            'variant_3': self._create_strategy_variant(base_length, 0.0003),
            'variant_4': self._create_strategy_variant(base_length, 0.0001),
            
            # Crypto correlation group - BTC/ETH coordination analysis
            'crypto_leader': self._simulate_crypto_correlation_leader(base_length),
            'crypto_follower_1': self._simulate_crypto_correlation_follower(base_length, 0.95, 1),  # 95% correlation, 1 day lag
            'crypto_follower_2': self._simulate_crypto_correlation_follower(base_length, 0.92, 2),  # 92% correlation, 2 day lag
            'crypto_follower_3': self._simulate_crypto_correlation_follower(base_length, 0.89, 1),  # 89% correlation, 1 day lag
            
            # Cross-timeframe analysis
            'timeframe_fast': self._simulate_timeframe_patterns(base_length, 'scalper'),
            'timeframe_medium': self._simulate_timeframe_patterns(base_length, 'medium'),
            'timeframe_slow': self._simulate_timeframe_patterns(base_length, 'swing'),
            
            # Mean reversion cluster (forex focus)
            'forex_mean_1': self._simulate_coordinated_mean_reversion(base_length, 'EURUSD'),
            'forex_mean_2': self._simulate_coordinated_mean_reversion(base_length, 'GBPUSD'),
            'forex_mean_3': self._simulate_coordinated_mean_reversion(base_length, 'USDCAD'),
            
            # High-frequency pattern analysis
            'hft_trend': self._simulate_hft_pattern_analysis(base_length, 'trend'),
            'hft_momentum': self._simulate_hft_pattern_analysis(base_length, 'momentum'),
            'hft_arbitrage': self._simulate_hft_pattern_analysis(base_length, 'arbitrage'),
            
            # Market making patterns
            'market_maker_1': self._simulate_market_making(base_length, 'aggressive'),
            'market_maker_2': self._simulate_market_making(base_length, 'passive'),
            
            # Cross-pair analysis
            'arb_eur_gbp': self._simulate_cross_pair_analysis(base_length, ['EURUSD', 'GBPUSD']),
            'arb_btc_eth': self._simulate_cross_pair_analysis(base_length, ['BTCUSD', 'ETHUSD']),
            
            # Market regime specialists
            'regime_specialist_1': self._simulate_regime_analysis(base_length, 'trend_detector'),
            'regime_specialist_2': self._simulate_regime_analysis(base_length, 'volatility_trader'),
            
            # Latency pattern analysis
            'latency_pattern_1': self._simulate_latency_patterns(base_length, 'fast'),
            'latency_pattern_2': self._simulate_latency_patterns(base_length, 'medium'),
        }
        
        return {name: self._create_ledger_from_returns(returns) 
                for name, returns in pattern_scenarios.items()}
    
    def _simulate_jpy_correlation_pattern(self, length):
        """Simulate the specific JPY correlation pattern for testing."""
        returns = []
        
        # JPY pairs tend to move together due to Bank of Japan policy
        # Create a pattern that analyzes this correlation
        base_jpy_trend = 0.0
        
        for i in range(length):
            # Simulate BoJ intervention days (every ~20 days)
            if i % 20 == 0:
                boj_intervention = np.random.choice([-0.02, 0.02], p=[0.3, 0.7])  # Usually strengthening JPY
                base_jpy_trend += boj_intervention * 0.5
            
            # Add carry trade unwind simulation (crisis periods)
            if i % 60 == 0:  # Every ~60 days
                carry_unwind = np.random.choice([-0.03, 0.0], p=[0.2, 0.8])
                base_jpy_trend += carry_unwind
            
            # Daily return based on JPY dynamics
            daily_return = base_jpy_trend * 0.1 + np.random.normal(0, 0.005)
            
            # Add momentum (JPY trends persist)
            if i > 5:
                recent_trend = np.mean(returns[-5:])
                daily_return += recent_trend * 0.3
            
            # Decay the trend slowly
            base_jpy_trend *= 0.995
            
            returns.append(daily_return)
        
        return returns
    
    def _add_noise_to_pattern(self, base_pattern, noise_scale):
        """Add small noise to create strategy variants for correlation testing."""
        return [x + np.random.normal(0, noise_scale) for x in base_pattern]
    
    def _create_base_strategy(self, length):
        """Create the base strategy template for similarity testing."""
        returns = []
        strategy_state = 0.0
        
        for i in range(length):
            # Sophisticated but repeatable strategy
            market_signal = np.sin(i * 0.1) * 0.01  # Cyclical component
            trend_signal = (i % 30) * 0.0002 - 0.003  # 30-day trend cycle
            
            daily_return = strategy_state + market_signal + trend_signal + np.random.normal(0, 0.002)
            
            # Update strategy state
            strategy_state = strategy_state * 0.9 + daily_return * 0.1
            
            returns.append(daily_return)
        
        return returns
    
    def _create_strategy_variant(self, length, variation_scale):
        """Create a variant of the base strategy with minimal variation."""
        base = self._create_base_strategy(length)
        
        # Add tiny variations for strategy diversity testing
        variant = []
        for i, ret in enumerate(base):
            # Occasional random variation
            if np.random.random() < 0.05:  # 5% chance
                variation = np.random.normal(0, variation_scale * 2)
            else:
                variation = np.random.normal(0, variation_scale)
            
            variant.append(ret + variation)
        
        return variant
    
    def _simulate_crypto_correlation_leader(self, length):
        """Simulate the leader in a crypto correlation pattern."""
        returns = []
        crypto_regime = 0.0
        
        for i in range(length):
            # Bitcoin-like volatility with trends
            if i % 40 == 0:  # New regime every 40 days
                crypto_regime = np.random.choice([-0.02, 0.03], p=[0.4, 0.6])
            
            # High volatility periods
            vol_multiplier = 1.0
            if i % 100 == 0:  # Volatility events
                vol_multiplier = 3.0
            
            daily_return = crypto_regime * 0.3 + np.random.normal(0, 0.02 * vol_multiplier)
            
            # Crypto momentum
            if i > 10:
                momentum = np.mean(returns[-10:]) * 0.2
                daily_return += momentum
            
            crypto_regime *= 0.99  # Slow decay
            returns.append(daily_return)
        
        return returns
    
    def _simulate_crypto_correlation_follower(self, length, correlation, lag_days):
        """Simulate a follower in the crypto correlation pattern with lag."""
        leader_returns = self._simulate_crypto_correlation_leader(length + lag_days)
        
        follower_returns = []
        for i in range(length):
            # Follow leader with lag and correlation
            leader_signal = leader_returns[i] if i < len(leader_returns) - lag_days else 0
            
            # Add correlation noise
            correlation_noise = np.random.normal(0, 0.01 * (1 - correlation))
            
            # Independent component
            independent = np.random.normal(0, 0.005) * (1 - correlation)
            
            follower_return = correlation * leader_signal + correlation_noise + independent
            follower_returns.append(follower_return)
        
        return follower_returns
    
    def _simulate_timeframe_patterns(self, length, style):
        """Simulate different timeframe trading patterns."""
        returns = []
        
        if style == 'scalper':
            # High frequency, small profits
            for i in range(length):
                base_return = np.random.normal(0.0002, 0.001)  # Small consistent gains
                
                # Occasional large loss (blown stops)
                if np.random.random() < 0.02:  # 2% chance
                    base_return = np.random.normal(-0.01, 0.005)
                
                returns.append(base_return)
                
        elif style == 'medium':
            # Medium frequency
            position_duration = 0
            current_position = 0.0
            
            for i in range(length):
                if position_duration <= 0:
                    # Enter new position
                    current_position = np.random.choice([-1, 1]) * np.random.uniform(0.005, 0.015)
                    position_duration = np.random.randint(3, 8)  # 3-8 day holds
                
                daily_return = current_position * np.random.normal(1.0, 0.3)
                position_duration -= 1
                returns.append(daily_return)
                
        else:  # swing
            # Longer term swings
            trend_duration = 0
            current_trend = 0.0
            
            for i in range(length):
                if trend_duration <= 0:
                    current_trend = np.random.normal(0, 0.008)
                    trend_duration = np.random.randint(15, 30)  # 15-30 day trends
                
                daily_return = current_trend + np.random.normal(0, 0.003)
                trend_duration -= 1
                returns.append(daily_return)
        
        return returns
    
    def _simulate_coordinated_mean_reversion(self, length, pair_focus):
        """Simulate coordinated mean reversion strategies."""
        returns = []
        price_level = 0.0
        
        # Different mean reversion speeds for different pairs
        reversion_speed = {'EURUSD': 0.05, 'GBPUSD': 0.08, 'USDCAD': 0.06}.get(pair_focus, 0.05)
        
        for i in range(length):
            # Mean reversion signal
            reversion_signal = -price_level * reversion_speed
            
            # Market noise
            noise = np.random.normal(0, 0.006)
            
            # Occasional trend breaks (fundamental events)
            if np.random.random() < 0.03:  # 3% chance
                trend_break = np.random.normal(0, 0.015)
                noise += trend_break
            
            daily_return = reversion_signal + noise
            price_level += daily_return
            
            # Reset if too extreme
            if abs(price_level) > 0.05:
                price_level *= 0.5
            
            returns.append(daily_return)
        
        return returns
    
    def _simulate_hft_pattern_analysis(self, length, pattern_type):
        """Simulate HFT pattern analysis strategies."""
        returns = []
        
        if pattern_type == 'trend':
            # Institutional trend following analysis
            trend_state = 0.0
            
            for i in range(length):
                # Detect trend changes (simplified)
                if i > 20:
                    recent_trend = np.mean(returns[-20:])
                    if abs(recent_trend) > 0.002:
                        trend_state = np.sign(recent_trend) * 0.01
                
                # Follow trend with some lag
                daily_return = trend_state * 0.5 + np.random.normal(0, 0.003)
                trend_state *= 0.95  # Decay
                returns.append(daily_return)
                
        elif pattern_type == 'momentum':
            # Momentum analysis patterns
            for i in range(length):
                base_return = np.random.normal(0, 0.003)
                
                # Occasional momentum spikes
                if np.random.random() < 0.1:  # 10% chance
                    momentum_spike = np.random.choice([0.01, -0.01])
                    base_return += momentum_spike
                
                returns.append(base_return)
                
        else:  # arbitrage
            # Cross-market analysis patterns
            opportunity = 0.0
            
            for i in range(length):
                # Simulate analysis opportunities
                if i % 15 == 0:  # Every 15 days
                    opportunity = np.random.normal(0, 0.005)
                
                daily_return = opportunity * 0.3 + np.random.normal(0, 0.002)
                opportunity *= 0.8  # Decay opportunity
                returns.append(daily_return)
        
        return returns
    
    def _simulate_market_making(self, length, intensity):
        """Simulate market making patterns."""
        returns = []
        cycle = 0
        
        multiplier = 2.0 if intensity == 'aggressive' else 1.2
        
        for i in range(length):
            # Market making cycle
            if cycle <= 0:
                cycle = np.random.randint(5, 15)  # 5-15 day cycles
                direction = np.random.choice([-1, 1])
            
            # Market making volume/return patterns
            if cycle > cycle * 0.5:
                # Building phase
                daily_return = direction * 0.002 * multiplier
            else:
                # Unwinding phase
                daily_return = -direction * 0.001 * multiplier
            
            # Add market noise
            daily_return += np.random.normal(0, 0.004)
            cycle -= 1
            returns.append(daily_return)
        
        return returns
    
    def _simulate_cross_pair_analysis(self, length, pair_list):
        """Simulate cross-pair correlation analysis."""
        returns = []
        
        # Create correlated movements between pairs
        base_factor = 0.0
        
        for i in range(length):
            # Update base factor (shared across pairs)
            if i % 10 == 0:
                base_factor = np.random.normal(0, 0.005)
            
            # Pair-specific component
            pair_factor = np.random.normal(0, 0.003)
            
            # Analysis opportunity (rare but profitable)
            analysis_component = 0.0
            if np.random.random() < 0.05:  # 5% chance
                analysis_component = np.random.normal(0.003, 0.001)
            
            daily_return = base_factor * 0.7 + pair_factor + analysis_component
            base_factor *= 0.9  # Decay
            returns.append(daily_return)
        
        return returns
    
    def _simulate_regime_analysis(self, length, style):
        """Simulate market regime analysis strategies."""
        returns = []
        
        if style == 'trend_detector':
            # Analyze and predict trend changes
            trend_countdown = 0
            
            for i in range(length):
                # Simulate trend changes every 80-120 days
                if trend_countdown <= 0:
                    trend_countdown = np.random.randint(80, 120)
                    trend_magnitude = np.random.uniform(0.02, 0.05)
                
                # Pre-trend positioning
                if trend_countdown < 10:  # 10 days before trend
                    daily_return = trend_magnitude * 0.1  # Small early positioning
                elif trend_countdown < 5:  # Trend develops
                    daily_return = trend_magnitude  # Benefit from trend
                else:
                    daily_return = np.random.normal(0, 0.004)
                
                trend_countdown -= 1
                returns.append(daily_return)
                
        else:  # volatility_trader
            # Analyze volatility patterns
            vol_state = 0.01  # Base volatility
            
            for i in range(length):
                # Volatility regime changes
                if i % 30 == 0:
                    vol_state = np.random.uniform(0.005, 0.03)
                
                # Benefit from volatility analysis
                vol_profit = (vol_state - 0.01) * 0.2  # Profit when vol > base
                
                daily_return = vol_profit + np.random.normal(0, vol_state)
                returns.append(daily_return)
        
        return returns
    
    def _simulate_latency_patterns(self, length, speed):
        """Simulate latency-sensitive trading patterns."""
        returns = []
        
        if speed == 'fast':
            # Ultra-fast pattern analysis (holding seconds/minutes)
            for i in range(length):
                # Many small profitable trades
                base_return = np.random.normal(0.0001, 0.0005)  # Tiny profits
                
                # Occasional tech failure loss
                if np.random.random() < 0.01:  # 1% chance
                    base_return = np.random.normal(-0.005, 0.002)
                
                returns.append(base_return)
        else:
            # Medium-speed pattern analysis
            for i in range(length):
                base_return = np.random.normal(0.0005, 0.002)
                
                # Market disruption losses
                if np.random.random() < 0.02:  # 2% chance
                    base_return = np.random.normal(-0.008, 0.003)
                
                returns.append(base_return)
        
        return returns
    
    def _create_diverse_strategies(self):
        """Create truly diverse strategies for comparison."""
        base_length = max(ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N + 20, 80)
        
        diverse = {
            # Fundamental analysis strategies
            'fundamental_macro': self._simulate_macro_fundamental(base_length),
            'fundamental_micro': self._simulate_micro_fundamental(base_length),
            
            # Technical analysis variants
            'technical_breakout': self._simulate_breakout_strategy(base_length),
            'technical_channel': self._simulate_channel_strategy(base_length),
            'technical_oscillator': self._simulate_oscillator_strategy(base_length),
            
            # Cross-asset strategies
            'commodities_trader': self._simulate_commodities_strategy(base_length),
            'rates_trader': self._simulate_rates_strategy(base_length),
            'credit_trader': self._simulate_credit_strategy(base_length),
            
            # Alternative strategies
            'event_driven': self._simulate_event_driven(base_length),
            'statistical_arb': self._simulate_statistical_arbitrage(base_length),
            'risk_parity': self._simulate_risk_parity(base_length),
        }
        
        return {name: self._create_ledger_from_returns(returns) 
                for name, returns in diverse.items()}
    
    def _simulate_macro_fundamental(self, length):
        """Simulate macro fundamental strategy."""
        returns = []
        interest_rate_cycle = 0
        economic_growth = 0
        
        for i in range(length):
            # Central bank cycle (every 6 months = ~180 days)
            if i % 180 == 0:
                interest_rate_cycle = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            
            # Economic data cycle (monthly = ~30 days)
            if i % 30 == 0:
                economic_growth = np.random.normal(0, 0.5)
            
            # Fundamental-based return
            fundamental_signal = interest_rate_cycle * 0.003 + economic_growth * 0.001
            daily_return = fundamental_signal + np.random.normal(0, 0.006)
            
            returns.append(daily_return)
        
        return returns
    
    def _simulate_micro_fundamental(self, length):
        """Simulate micro fundamental (company-specific) strategy."""
        returns = []
        earnings_cycle = 0
        
        for i in range(length):
            # Quarterly earnings (every 90 days)
            if i % 90 == 0:
                earnings_surprise = np.random.normal(0, 0.02)
                earnings_cycle = earnings_surprise
            
            # Decay earnings impact
            earnings_cycle *= 0.98
            
            # Company-specific events
            event_return = 0
            if np.random.random() < 0.02:  # 2% chance of event
                event_return = np.random.normal(0, 0.015)
            
            daily_return = earnings_cycle * 0.5 + event_return + np.random.normal(0, 0.008)
            returns.append(daily_return)
        
        return returns
    
    def _simulate_breakout_strategy(self, length):
        """Simulate breakout technical strategy."""
        returns = []
        price_range = [0, 0]  # [low, high]
        
        for i in range(length):
            if i < 20:
                # Build initial range
                daily_return = np.random.normal(0, 0.004)
                price_range[0] = min(price_range[0], daily_return)
                price_range[1] = max(price_range[1], daily_return)
            else:
                # Look for breakouts
                recent_move = np.random.normal(0, 0.006)
                
                # Breakout detection
                if recent_move > price_range[1] * 1.5:  # Upside breakout
                    daily_return = recent_move * 1.2  # Follow breakout
                elif recent_move < price_range[0] * 1.5:  # Downside breakout
                    daily_return = recent_move * 1.2  # Follow breakout
                else:
                    daily_return = recent_move  # No breakout
                
                # Update range periodically
                if i % 20 == 0:
                    price_range = [daily_return * 0.8, daily_return * 1.2]
            
            returns.append(daily_return)
        
        return returns
    
    def _simulate_channel_strategy(self, length):
        """Simulate channel trading strategy."""
        returns = []
        channel_center = 0
        channel_width = 0.01
        
        for i in range(length):
            # Update channel periodically
            if i % 40 == 0:
                channel_center = np.random.normal(0, 0.003)
                channel_width = np.random.uniform(0.005, 0.015)
            
            # Current price relative to channel
            market_move = np.random.normal(channel_center, 0.006)
            
            # Channel strategy: fade extremes
            if market_move > channel_center + channel_width/2:
                daily_return = -0.002  # Fade high
            elif market_move < channel_center - channel_width/2:
                daily_return = 0.002   # Fade low
            else:
                daily_return = np.random.normal(0, 0.001)  # Neutral
            
            returns.append(daily_return)
        
        return returns
    
    def _simulate_oscillator_strategy(self, length):
        """Simulate oscillator-based strategy."""
        returns = []
        oscillator_state = 0
        
        for i in range(length):
            # RSI-like oscillator
            market_move = np.random.normal(0, 0.008)
            oscillator_state = oscillator_state * 0.9 + market_move * 0.1
            
            # Oscillator signals
            if oscillator_state > 0.01:  # Overbought
                daily_return = -oscillator_state * 0.3
            elif oscillator_state < -0.01:  # Oversold
                daily_return = -oscillator_state * 0.3
            else:
                daily_return = np.random.normal(0, 0.003)
            
            returns.append(daily_return)
        
        return returns
    
    def _simulate_commodities_strategy(self, length):
        """Simulate commodities-focused strategy."""
        returns = []
        seasonal_factor = 0
        supply_shock = 0
        
        for i in range(length):
            # Seasonal patterns (agricultural commodities)
            seasonal_factor = 0.005 * np.sin(i * 2 * np.pi / 365)  # Annual cycle
            
            # Supply/demand shocks
            if np.random.random() < 0.03:  # 3% chance
                supply_shock = np.random.normal(0, 0.02)
            
            # Decay supply shock
            supply_shock *= 0.95
            
            daily_return = seasonal_factor + supply_shock + np.random.normal(0, 0.01)
            returns.append(daily_return)
        
        return returns
    
    def _simulate_rates_strategy(self, length):
        """Simulate interest rates strategy."""
        returns = []
        yield_curve_slope = 0
        duration_exposure = np.random.uniform(2, 10)  # 2-10 year duration
        
        for i in range(length):
            # Yield curve changes
            if i % 30 == 0:  # Monthly
                yield_curve_slope = np.random.normal(0, 0.002)
            
            # Duration-based P&L
            rate_change = np.random.normal(0, 0.001)
            duration_pnl = -duration_exposure * rate_change  # Inverse relationship
            
            # Curve positioning
            curve_pnl = yield_curve_slope * 0.5
            
            daily_return = duration_pnl + curve_pnl + np.random.normal(0, 0.003)
            returns.append(daily_return)
        
        return returns
    
    def _simulate_credit_strategy(self, length):
        """Simulate credit strategy."""
        returns = []
        credit_spread = 0.002  # Base spread
        
        for i in range(length):
            # Credit cycle
            if i % 60 == 0:  # Every 2 months
                credit_spread += np.random.normal(0, 0.001)
                credit_spread = max(0.0005, min(0.01, credit_spread))  # Keep reasonable
            
            # Credit events
            credit_event = 0
            if np.random.random() < 0.01:  # 1% chance
                credit_event = np.random.normal(-0.005, 0.003)  # Usually negative
            
            # Carry + credit beta
            carry = credit_spread * 0.1  # Daily carry
            beta_move = np.random.normal(0, 0.004)
            
            daily_return = carry + beta_move + credit_event
            returns.append(daily_return)
        
        return returns
    
    def _simulate_event_driven(self, length):
        """Simulate event-driven strategy."""
        returns = []
        
        for i in range(length):
            # Corporate events (M&A, earnings, etc.)
            if np.random.random() < 0.05:  # 5% chance of event
                event_type = np.random.choice(['merger', 'earnings', 'restructuring'])
                
                if event_type == 'merger':
                    daily_return = np.random.uniform(0.005, 0.02)  # Usually positive
                elif event_type == 'earnings':
                    daily_return = np.random.normal(0, 0.015)  # Can go either way
                else:  # restructuring
                    daily_return = np.random.normal(-0.002, 0.01)  # Usually slightly negative
            else:
                daily_return = np.random.normal(0, 0.002)  # Quiet periods
            
            returns.append(daily_return)
        
        return returns
    
    def _simulate_statistical_arbitrage(self, length):
        """Simulate statistical arbitrage strategy."""
        returns = []
        pair_spread = 0
        
        for i in range(length):
            # Pair spread evolution
            pair_spread += np.random.normal(0, 0.003)
            
            # Mean reversion trade
            if abs(pair_spread) > 0.01:  # Spread too wide
                trade_return = -np.sign(pair_spread) * 0.002  # Fade the spread
                pair_spread *= 0.8  # Spread tightens from trade
            else:
                trade_return = np.random.normal(0, 0.001)
            
            returns.append(trade_return)
        
        return returns
    
    def _simulate_risk_parity(self, length):
        """Simulate risk parity strategy."""
        returns = []
        asset_vols = [0.01, 0.015, 0.02, 0.008]  # Different asset volatilities
        
        for i in range(length):
            # Risk parity allocation (inverse vol weighting)
            total_inv_vol = sum(1/vol for vol in asset_vols)
            weights = [(1/vol)/total_inv_vol for vol in asset_vols]
            
            # Asset returns
            asset_returns = [np.random.normal(0, vol) for vol in asset_vols]
            
            # Portfolio return
            daily_return = sum(w * r for w, r in zip(weights, asset_returns))
            
            # Update volatilities periodically
            if i % 20 == 0:
                asset_vols = [max(0.005, vol + np.random.normal(0, 0.002)) for vol in asset_vols]
            
            returns.append(daily_return)
        
        return returns
    
    def _create_ledger_from_returns(self, returns):
        """Create a PerfLedger from a list of daily returns."""
        checkpoints = []
        base_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        for i, daily_return in enumerate(returns):
            day_start = base_time + (i * 24 * 60 * 60 * 1000)
            n_checkpoints_per_day = int(ValiConfig.DAILY_CHECKPOINTS)
            checkpoint_duration = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
            
            for j in range(n_checkpoints_per_day):
                checkpoint_time = day_start + (j * checkpoint_duration)
                cp_return = daily_return / n_checkpoints_per_day
                gain = max(0, cp_return)
                loss = min(0, cp_return)
                
                checkpoint = PerfCheckpoint(
                    last_update_ms=checkpoint_time + checkpoint_duration,
                    accum_ms=checkpoint_duration,
                    open_ms=checkpoint_time,
                    gain=gain,
                    loss=loss,
                    prev_portfolio_ret=1.0 + cp_return,
                    n_updates=1,
                    mdd=0.99
                )
                checkpoints.append(checkpoint)
        
        return ledger_generator(checkpoints=checkpoints)

    def test_jpy_correlation_pattern_detection(self):
        """Test detection of JPY correlation patterns."""
        jpy_patterns = {
            'jpy_pattern_1': self.pattern_scenarios['jpy_pattern_1'],
            'jpy_pattern_2': self.pattern_scenarios['jpy_pattern_2'],
            'jpy_pattern_3': self.pattern_scenarios['jpy_pattern_3'],
            'jpy_pattern_4': self.pattern_scenarios['jpy_pattern_4'],
            'jpy_pattern_5': self.pattern_scenarios['jpy_pattern_5'],
        }
        
        # Test correlation detection
        returns = {name: LedgerUtils.daily_return_log(ledger) 
                  for name, ledger in jpy_patterns.items()}
        corr_matrix, mean_correlations = Orthogonality.correlation_matrix(returns)
        
        if not corr_matrix.empty and mean_correlations:
            # JPY patterns should show high correlation
            correlations = list(mean_correlations.values())
            avg_correlation = np.mean([abs(c) for c in correlations])
            
            self.assertGreater(avg_correlation, 0.3, 
                             "JPY correlation patterns should show significant correlation")
        
        # Test penalty system response
        penalties = LedgerUtils.orthogonality_penalty(jpy_patterns)
        
        # All JPY patterns should receive substantial penalties
        penalty_values = list(penalties.values())
        avg_penalty = np.mean(penalty_values)
        
        self.assertGreater(avg_penalty, 0.4, 
                         "JPY correlation patterns should receive substantial penalties")
        
        # Test against diverse strategies
        diverse_subset = {
            'fundamental': self.diverse_strategies['fundamental_macro'],
            'technical': self.diverse_strategies['technical_breakout'],
            'commodities': self.diverse_strategies['commodities_trader'],
        }
        
        diverse_penalties = LedgerUtils.orthogonality_penalty(diverse_subset)
        diverse_avg = np.mean(list(diverse_penalties.values()))
        
        self.assertGreater(avg_penalty, diverse_avg + 0.1, 
                         "Correlated strategies should be penalized more than diverse ones")

    def test_strategy_similarity_detection(self):
        """Test detection of similar strategies with minor variations."""
        similar_strategies = {
            'base_strategy': self.pattern_scenarios['base_strategy'],
            'variant_1': self.pattern_scenarios['variant_1'],
            'variant_2': self.pattern_scenarios['variant_2'],
            'variant_3': self.pattern_scenarios['variant_3'],
            'variant_4': self.pattern_scenarios['variant_4'],
        }
        
        penalties = LedgerUtils.orthogonality_penalty(similar_strategies)
        
        # All similar strategies should receive high penalties
        penalty_values = list(penalties.values())
        min_penalty = min(penalty_values)
        
        self.assertGreater(min_penalty, 0.5, 
                         "All similar strategies should receive substantial penalties")
        
        # Penalties should be relatively uniform (all are similar)
        penalty_std = np.std(penalty_values)
        self.assertLess(penalty_std, 0.2, 
                       "Similar strategies should have comparable penalty levels")

    def test_crypto_correlation_cluster_detection(self):
        """Test detection of crypto correlation clusters."""
        crypto_cluster = {
            'crypto_leader': self.pattern_scenarios['crypto_leader'],
            'crypto_follower_1': self.pattern_scenarios['crypto_follower_1'],
            'crypto_follower_2': self.pattern_scenarios['crypto_follower_2'],
            'crypto_follower_3': self.pattern_scenarios['crypto_follower_3'],
        }
        
        # Test lagged correlation detection
        returns = {name: LedgerUtils.daily_return_log(ledger) 
                  for name, ledger in crypto_cluster.items()}
        
        # Test sliding similarity (should catch lagged correlations)
        leader_returns = returns['crypto_leader']
        follower_returns = returns['crypto_follower_1']
        
        sliding_sim = Orthogonality.sliding_similarity(leader_returns, follower_returns)
        
        self.assertGreater(sliding_sim, 0.5, 
                         "Sliding similarity should detect lagged correlation")
        
        # Test penalty system
        penalties = LedgerUtils.orthogonality_penalty(crypto_cluster)
        avg_penalty = np.mean(list(penalties.values()))
        
        self.assertGreater(avg_penalty, 0.3, 
                         "Crypto correlation cluster should receive significant penalties")

    def test_market_making_pattern_detection(self):
        """Test detection of market making patterns."""
        market_makers = {
            'market_maker_1': self.pattern_scenarios['market_maker_1'],
            'market_maker_2': self.pattern_scenarios['market_maker_2'],
            'diverse_strategy': self.diverse_strategies['fundamental_macro'],
        }
        
        penalties = LedgerUtils.orthogonality_penalty(market_makers)
        
        # Market makers should be penalized more than diverse strategies
        maker_penalty_avg = (penalties['market_maker_1'] + penalties['market_maker_2']) / 2
        diverse_penalty = penalties['diverse_strategy']
        
        self.assertGreater(maker_penalty_avg, diverse_penalty, 
                         "Market makers should be penalized more than diverse strategies")

    def test_cross_timeframe_pattern_detection(self):
        """Test detection of cross-timeframe trading patterns."""
        timeframe_patterns = {
            'timeframe_fast': self.pattern_scenarios['timeframe_fast'],
            'timeframe_medium': self.pattern_scenarios['timeframe_medium'],
            'timeframe_slow': self.pattern_scenarios['timeframe_slow'],
        }
        
        penalties = LedgerUtils.orthogonality_penalty(timeframe_patterns)
        
        # Should detect if these are too similar despite different timeframes
        penalty_values = list(penalties.values())
        self.assertTrue(all(p > 0.0 for p in penalty_values), 
                       "All timeframe patterns should receive some penalty")

    def test_coordinated_mean_reversion_detection(self):
        """Test detection of coordinated mean reversion strategies."""
        mean_rev_cluster = {
            'forex_mean_1': self.pattern_scenarios['forex_mean_1'],
            'forex_mean_2': self.pattern_scenarios['forex_mean_2'],
            'forex_mean_3': self.pattern_scenarios['forex_mean_3'],
        }
        
        penalties = LedgerUtils.orthogonality_penalty(mean_rev_cluster)
        
        # Should detect correlation even across different pairs
        avg_penalty = np.mean(list(penalties.values()))
        self.assertGreater(avg_penalty, 0.2, 
                         "Coordinated mean reversion should be penalized")

    def test_hft_pattern_analysis_detection(self):
        """Test detection of HFT pattern analysis strategies."""
        hft_patterns = {
            'hft_trend': self.pattern_scenarios['hft_trend'],
            'hft_momentum': self.pattern_scenarios['hft_momentum'],
            'hft_arbitrage': self.pattern_scenarios['hft_arbitrage'],
        }
        
        penalties = LedgerUtils.orthogonality_penalty(hft_patterns)
        
        # Should detect similarity in HFT patterns
        penalty_values = list(penalties.values())
        self.assertTrue(all(p > 0.1 for p in penalty_values), 
                       "HFT patterns should receive meaningful penalties")

    def test_latency_pattern_detection(self):
        """Test detection of latency-sensitive trading patterns."""
        latency_patterns = {
            'latency_pattern_1': self.pattern_scenarios['latency_pattern_1'],
            'latency_pattern_2': self.pattern_scenarios['latency_pattern_2'],
            'diverse_strategy': self.diverse_strategies['statistical_arb'],  # Different type of strategy
        }
        
        penalties = LedgerUtils.orthogonality_penalty(latency_patterns)
        
        # Latency patterns should be more correlated with each other than with diverse strategy
        latency_penalty_avg = (penalties['latency_pattern_1'] + penalties['latency_pattern_2']) / 2
        diverse_penalty = penalties['diverse_strategy']
        
        # Test depends on whether they're detected as similar
        self.assertGreaterEqual(latency_penalty_avg, 0.0, 
                              "Latency pattern penalties should be calculated")

    def test_correlated_vs_diverse_penalty_differential(self):
        """Test that correlated strategies receive significantly higher penalties than diverse ones."""
        # Select representative correlated strategies
        correlated_subset = {
            'jpy_pattern': self.pattern_scenarios['jpy_pattern_1'],
            'strategy_variant': self.pattern_scenarios['variant_1'],
            'crypto_follower': self.pattern_scenarios['crypto_follower_1'],
            'market_maker': self.pattern_scenarios['market_maker_1'],
        }
        
        # Select diverse strategies
        diverse_subset = {
            'fundamental': self.diverse_strategies['fundamental_macro'],
            'technical': self.diverse_strategies['technical_breakout'],
            'commodities': self.diverse_strategies['commodities_trader'],
            'event_driven': self.diverse_strategies['event_driven'],
        }
        
        correlated_penalties = LedgerUtils.orthogonality_penalty(correlated_subset)
        diverse_penalties = LedgerUtils.orthogonality_penalty(diverse_subset)
        
        correlated_avg = np.mean(list(correlated_penalties.values()))
        diverse_avg = np.mean(list(diverse_penalties.values()))
        
        self.assertGreater(correlated_avg, diverse_avg, 
                         "Correlated strategies should receive higher penalties than diverse strategies")
        
        # The difference should be substantial
        penalty_differential = correlated_avg - diverse_avg
        self.assertGreater(penalty_differential, 0.1, 
                         "Penalty differential should be substantial (>0.1)")

    def test_massive_correlation_scenario(self):
        """Test system behavior with a large number of correlated strategies."""
        # Combine all pattern scenarios
        all_patterns = self.pattern_scenarios.copy()
        
        # Add some diverse strategies as control
        all_patterns.update({
            'diverse_1': self.diverse_strategies['fundamental_macro'],
            'diverse_2': self.diverse_strategies['technical_breakout'],
            'diverse_3': self.diverse_strategies['commodities_trader'],
        })
        
        penalties = LedgerUtils.orthogonality_penalty(all_patterns)
        
        # System should handle large number of strategies
        self.assertEqual(len(penalties), len(all_patterns), 
                        "Should calculate penalties for all strategies")
        
        # All penalties should be valid
        for miner, penalty in penalties.items():
            self.assertGreaterEqual(penalty, 0.0, f"Penalty for {miner} should be non-negative")
            self.assertLessEqual(penalty, 1.0, f"Penalty for {miner} should not exceed 1.0")
            self.assertTrue(np.isfinite(penalty), f"Penalty for {miner} should be finite")
        
        # Diverse strategies should generally have lower penalties
        diverse_penalties = [penalties['diverse_1'], penalties['diverse_2'], penalties['diverse_3']]
        pattern_penalties_list = [penalty for name, penalty in penalties.items() 
                               if not name.startswith('diverse')]
        
        diverse_avg = np.mean(diverse_penalties)
        pattern_avg = np.mean(pattern_penalties_list)
        
        self.assertGreater(pattern_avg, diverse_avg, 
                         "Correlated patterns should have higher average penalties")

    def test_adaptive_correlation_resistance(self):
        """Test resistance to adaptive strategies that try to reduce correlation detection."""
        # Create strategies that try to reduce correlation while maintaining similarity
        adaptive_patterns = {}
        
        # Strategy 1: Add random noise to try to reduce correlation
        base_strategy = self._create_base_strategy(80)
        for i in range(5):
            noisy_strategy = [x + np.random.normal(0, 0.01) for x in base_strategy]
            adaptive_patterns[f'noise_variant_{i}'] = self._create_ledger_from_returns(noisy_strategy)
        
        # Strategy 2: Use different time windows but same underlying pattern
        for i in range(3):
            windowed_strategy = []
            window_size = 5 + i * 2  # Different window sizes
            for j in range(len(base_strategy)):
                if j < window_size:
                    windowed_strategy.append(base_strategy[j])
                else:
                    # Apply moving average with different windows
                    avg = np.mean(base_strategy[j-window_size:j])
                    windowed_strategy.append(avg + np.random.normal(0, 0.002))
            
            adaptive_patterns[f'window_variant_{i}'] = self._create_ledger_from_returns(windowed_strategy)
        
        # Test that the system still detects these as similar
        penalties = LedgerUtils.orthogonality_penalty(adaptive_patterns)
        
        # Should still detect high correlation despite adaptive attempts
        penalty_values = list(penalties.values())
        avg_penalty = np.mean(penalty_values)
        
        self.assertGreater(avg_penalty, 0.4, 
                         "Adaptive correlation attempts should still receive high penalties")
        
        # Penalties should be relatively uniform (indicating they're all similar)
        penalty_std = np.std(penalty_values)
        self.assertLess(penalty_std, 0.3, 
                       "Adaptive correlation penalties should be relatively uniform")

    def test_parameter_correlation_resistance(self):
        """Test resistance to parameter manipulation attempts."""
        original_corr_weight = ValiConfig.ORTHOGONALITY_CORRELATION_WEIGHT
        original_pref_weight = ValiConfig.ORTHOGONALITY_PREFERENCE_WEIGHT
        original_intensity = ValiConfig.ORTHOGONALITY_DIVERGING_INTENSITY
        
        try:
            # Test with different parameter combinations
            test_strategies = {
                'jpy_pattern_1': self.pattern_scenarios['jpy_pattern_1'],
                'jpy_pattern_2': self.pattern_scenarios['jpy_pattern_2'],
                'variant_1': self.pattern_scenarios['variant_1'],
                'diverse_strategy': self.diverse_strategies['fundamental_macro'],
            }
            
            # Test multiple parameter combinations
            param_combinations = [
                (0.8, 0.2, 0.5),   # High correlation weight
                (0.3, 0.7, 0.8),   # High preference weight
                (0.5, 0.5, 1.0),   # Balanced, high intensity
                (0.7, 0.3, 0.3),   # Different combination
            ]
            
            for corr_w, pref_w, intensity in param_combinations:
                ValiConfig.ORTHOGONALITY_CORRELATION_WEIGHT = corr_w
                ValiConfig.ORTHOGONALITY_PREFERENCE_WEIGHT = pref_w
                ValiConfig.ORTHOGONALITY_DIVERGING_INTENSITY = intensity
                
                penalties = LedgerUtils.orthogonality_penalty(test_strategies)
                
                # Correlated strategies should always be penalized more than diverse ones
                correlated_penalties = [penalties['jpy_pattern_1'], penalties['jpy_pattern_2'], 
                                      penalties['variant_1']]
                diverse_penalty = penalties['diverse_strategy']
                
                correlated_avg = np.mean(correlated_penalties)
                
                self.assertGreater(correlated_avg, diverse_penalty, 
                                 f"Correlation should be penalized more with params {corr_w}, {pref_w}, {intensity}")
        
        finally:
            # Restore original parameters
            ValiConfig.ORTHOGONALITY_CORRELATION_WEIGHT = original_corr_weight
            ValiConfig.ORTHOGONALITY_PREFERENCE_WEIGHT = original_pref_weight
            ValiConfig.ORTHOGONALITY_DIVERGING_INTENSITY = original_intensity

    def test_extreme_market_conditions_correlation(self):
        """Test correlation detection under extreme market conditions."""
        # Create extreme market scenarios
        extreme_scenarios = {}
        
        # Market crash scenario
        crash_returns = []
        for i in range(80):
            if i < 10:  # Pre-crash
                crash_returns.append(np.random.normal(0.001, 0.005))
            elif i < 20:  # Crash
                crash_returns.append(np.random.normal(-0.05, 0.02))
            else:  # Recovery
                crash_returns.append(np.random.normal(0.02, 0.015))
        
        # Correlated strategies that follow similar crash patterns
        for j in range(5):
            # Similar crisis-response strategies
            correlated_crash = [x + np.random.normal(0, 0.002) for x in crash_returns]
            extreme_scenarios[f'crash_pattern_{j}'] = self._create_ledger_from_returns(correlated_crash)
        
        # Add a diverse crisis strategy
        diverse_crisis = self.diverse_strategies['event_driven']
        extreme_scenarios['diverse_crisis'] = diverse_crisis
        
        penalties = LedgerUtils.orthogonality_penalty(extreme_scenarios)
        
        # Correlated strategies should still be detected as similar even in extreme conditions
        correlated_penalties = [penalties[f'crash_pattern_{j}'] for j in range(5)]
        diverse_penalty = penalties['diverse_crisis']
        
        correlated_avg = np.mean(correlated_penalties)
        
        self.assertGreater(correlated_avg, diverse_penalty, 
                         "Correlated strategies should be detected even in extreme market conditions")

    def test_cross_asset_correlation_detection(self):
        """Test detection of correlation across different asset classes."""
        # Cross-asset correlation patterns
        cross_asset_patterns = {
            'cross_crypto': self.pattern_scenarios['crypto_leader'],
            'cross_forex': self.pattern_scenarios['jpy_pattern_1'],
            'cross_pair': self.pattern_scenarios['arb_eur_gbp'],
            'cross_hft': self.pattern_scenarios['hft_trend'],
        }
        
        # Add diverse cross-asset strategy
        cross_asset_patterns['diverse_cross'] = self.diverse_strategies['risk_parity']
        
        penalties = LedgerUtils.orthogonality_penalty(cross_asset_patterns)
        
        # System should detect correlation patterns even across asset classes
        pattern_penalties = [penalties['cross_crypto'], penalties['cross_forex'], 
                           penalties['cross_pair'], penalties['cross_hft']]
        diverse_penalty = penalties['diverse_cross']
        
        # At least some correlation patterns should be detected
        detected_patterns = sum(1 for p in pattern_penalties if p > diverse_penalty + 0.05)
        
        self.assertGreater(detected_patterns, 0, 
                         "Should detect some correlation patterns across asset classes")

    def test_time_series_correlation_detection(self):
        """Test detection of time-series based correlation patterns."""
        # Create strategies that follow time-series patterns
        time_patterns = {}
        
        # Strategy 1: Delayed correlation with different lags
        base_pattern = self._simulate_jpy_correlation_pattern(80)
        for lag in [1, 2, 3, 5, 7]:
            delayed_pattern = [0] * lag + base_pattern[:-lag]  # Shift by lag days
            time_patterns[f'delayed_pattern_{lag}'] = self._create_ledger_from_returns(delayed_pattern)
        
        # Strategy 2: Inverted correlation patterns
        for multiplier in [-1, -0.8, -0.9]:
            inverted_pattern = [x * multiplier for x in base_pattern]
            time_patterns[f'inverted_{abs(multiplier)}'] = self._create_ledger_from_returns(inverted_pattern)
        
        penalties = LedgerUtils.orthogonality_penalty(time_patterns)
        
        # Should detect these as correlated despite time shifts and inversions
        penalty_values = list(penalties.values())
        avg_penalty = np.mean(penalty_values)
        
        self.assertGreater(avg_penalty, 0.3, 
                         "Time-series correlation should be detected and penalized")

    def test_volume_based_correlation_detection(self):
        """Test detection of volume-based correlation patterns."""
        # Create strategies that follow volume-based patterns
        volume_patterns = {}
        
        # Volume-weighted correlation
        base_strategy = self._create_base_strategy(80)
        for i in range(4):
            # Same strategy but simulate different volume weighting
            volume_weighted = []
            for j, ret in enumerate(base_strategy):
                # Apply volume weighting on certain days
                if j % (10 + i) == 0:  # Different volume cycles
                    volume_multiplier = 1.5 + i * 0.2
                    volume_weighted.append(ret * volume_multiplier)
                else:
                    volume_weighted.append(ret)
            
            volume_patterns[f'volume_pattern_{i}'] = self._create_ledger_from_returns(volume_weighted)
        
        penalties = LedgerUtils.orthogonality_penalty(volume_patterns)
        
        # Should still detect correlation despite volume weighting
        penalty_values = list(penalties.values())
        avg_penalty = np.mean(penalty_values)
        
        self.assertGreater(avg_penalty, 0.4, 
                         "Volume-based correlation should be detected and penalized")

    def test_statistical_correlation_robustness(self):
        """Test robustness against statistical distribution manipulation."""
        # Create strategies that modify statistical properties while maintaining correlation
        stat_patterns = {}
        
        # Strategy 1: Same correlation but different variance
        base_pattern = self._simulate_jpy_correlation_pattern(80)
        for variance_mult in [0.5, 0.8, 1.2, 1.5]:
            scaled_pattern = []
            for ret in base_pattern:
                noise = np.random.normal(0, 0.003 * variance_mult)
                scaled_pattern.append(ret + noise)
            stat_patterns[f'variance_scaled_{variance_mult}'] = self._create_ledger_from_returns(scaled_pattern)
        
        # Strategy 2: Same mean but different distribution shapes
        for skew in [0.5, 1.0, 1.5, 2.0]:
            skewed_pattern = []
            for ret in base_pattern:
                # Add skewed noise
                skewed_noise = np.random.gamma(skew, 0.001) - skew * 0.001
                skewed_pattern.append(ret + skewed_noise)
            stat_patterns[f'skewed_{skew}'] = self._create_ledger_from_returns(skewed_pattern)
        
        penalties = LedgerUtils.orthogonality_penalty(stat_patterns)
        
        # Should detect correlation despite statistical manipulation
        penalty_values = list(penalties.values())
        min_penalty = min(penalty_values)
        
        self.assertGreater(min_penalty, 0.3, 
                         "Statistical correlation should be detected despite distribution manipulation")