import numpy as np
import math
from unittest.mock import patch
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.orthogonality import Orthogonality
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.vali_config import ValiConfig, TradePair
from vali_objects.vali_dataclasses.perf_ledger import PerfLedger, PerfCheckpoint
from tests.shared_objects.test_utilities import ledger_generator, checkpoint_generator
from datetime import datetime, timezone
import copy


class TestPatternAnalysisComprehensive(TestBase):
    """Advanced tests for pattern analysis and correlation detection in trading strategies."""

    def setUp(self):
        super().setUp()
        
        # Create sophisticated pattern analysis scenarios
        self.pattern_configurations = self._create_pattern_configurations()
        
    def _create_pattern_configurations(self):
        """Create advanced pattern configurations for comprehensive correlation analysis."""
        base_length = max(ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N + 30, 90)
        
        pattern_configurations = {
            # Advanced JPY correlation cluster - cross-currency analysis
            'jpy_cluster_primary': self._create_jpy_cluster_primary(base_length),
            'jpy_cluster_secondary_1': self._create_jpy_cluster_secondary(base_length, 0.95, 'USDJPY'),
            'jpy_cluster_secondary_2': self._create_jpy_cluster_secondary(base_length, 0.92, 'EURJPY'),
            'jpy_cluster_secondary_3': self._create_jpy_cluster_secondary(base_length, 0.89, 'GBPJPY'),
            'jpy_cluster_secondary_4': self._create_jpy_cluster_secondary(base_length, 0.86, 'AUDJPY'),
            'jpy_cluster_secondary_5': self._create_jpy_cluster_secondary(base_length, 0.93, 'CADJPY'),
            'jpy_cluster_secondary_6': self._create_jpy_cluster_secondary(base_length, 0.91, 'CHFJPY'),
            'jpy_cluster_secondary_7': self._create_jpy_cluster_secondary(base_length, 0.88, 'NZDJPY'),
            
            # Multi-tier correlation hierarchy
            'correlation_tier_1_primary': self._create_correlation_tier_primary(base_length),
            'correlation_tier_1_derived_1': self._create_correlation_tier_derived(base_length, 1, 0.94),
            'correlation_tier_1_derived_2': self._create_correlation_tier_derived(base_length, 1, 0.91),
            'correlation_tier_2_derived_1': self._create_correlation_tier_derived(base_length, 2, 0.87),
            'correlation_tier_2_derived_2': self._create_correlation_tier_derived(base_length, 2, 0.85),
            'correlation_tier_3_derived_1': self._create_correlation_tier_derived(base_length, 3, 0.82),
            
            # Signal transformation chain - processing through intermediaries
            'signal_source': self._create_signal_source(base_length),
            'transformer_1': self._create_signal_transformer(base_length, 1, 0.9),
            'transformer_2': self._create_signal_transformer(base_length, 2, 0.85),
            'transformer_3': self._create_signal_transformer(base_length, 3, 0.8),
            'terminal_processor_1': self._create_signal_terminal_processor(base_length, 4, 0.75),
            'terminal_processor_2': self._create_signal_terminal_processor(base_length, 5, 0.7),
            
            # Variant strategies - same logic, different implementations
            'strategy_base': self._create_strategy_base(base_length),
            'strategy_variant_1': self._create_strategy_variant(base_length, 'invert_signals'),
            'strategy_variant_2': self._create_strategy_variant(base_length, 'delay_signals'),
            'strategy_variant_3': self._create_strategy_variant(base_length, 'amplify_signals'),
            'strategy_variant_4': self._create_strategy_variant(base_length, 'compress_signals'),
            'strategy_variant_5': self._create_strategy_variant(base_length, 'noise_inject'),
            
            # Cross-validation adaptation - phase-specific optimization
            'cv_optimizer_train': self._create_cv_optimization_strategy(base_length, 'train'),
            'cv_optimizer_val_1': self._create_cv_optimization_strategy(base_length, 'validate'),
            'cv_optimizer_val_2': self._create_cv_optimization_strategy(base_length, 'validate'),
            'cv_optimizer_test': self._create_cv_optimization_strategy(base_length, 'test'),
            
            # Regime switching specialists
            'regime_specialist_1': self._create_regime_specialist(base_length, 'bull_specialist'),
            'regime_specialist_2': self._create_regime_specialist(base_length, 'bear_specialist'),
            'regime_specialist_3': self._create_regime_specialist(base_length, 'vol_specialist'),
            'regime_specialist_4': self._create_regime_specialist(base_length, 'crisis_specialist'),
            
            # Temporal arbitrage strategies
            'temporal_strategy_1': self._create_temporal_strategy(base_length, 'microsecond'),
            'temporal_strategy_2': self._create_temporal_strategy(base_length, 'millisecond'),
            'temporal_strategy_3': self._create_temporal_strategy(base_length, 'second'),
            'temporal_strategy_4': self._create_temporal_strategy(base_length, 'minute'),
            
            # Information advantage strategies
            'info_strategy_1': self._create_info_strategy(base_length, 'news_front_run'),
            'info_strategy_2': self._create_info_strategy(base_length, 'insider_mimic'),
            'info_strategy_3': self._create_info_strategy(base_length, 'order_flow_read'),
            
            # Network correlation patterns
            'network_hub': self._create_network_hub(base_length),
            'network_node_1': self._create_network_node(base_length, 1, 0.88),
            'network_node_2': self._create_network_node(base_length, 2, 0.85),
            'network_node_3': self._create_network_node(base_length, 3, 0.82),
            'network_node_4': self._create_network_node(base_length, 4, 0.79),
            'network_node_5': self._create_network_node(base_length, 5, 0.76),
            
            # Adaptive learning strategies
            'adaptive_learner_1': self._create_adaptive_learner(base_length, 'fast_adapt'),
            'adaptive_learner_2': self._create_adaptive_learner(base_length, 'medium_adapt'),
            'adaptive_learner_3': self._create_adaptive_learner(base_length, 'slow_adapt'),
            
            # Market microstructure strategies
            'microstructure_1': self._create_microstructure_strategy(base_length, 'bid_ask_spread'),
            'microstructure_2': self._create_microstructure_strategy(base_length, 'order_book_depth'),
            'microstructure_3': self._create_microstructure_strategy(base_length, 'tick_size_optimization'),
        }
        
        return {name: self._create_ledger_from_returns(returns) 
                for name, returns in pattern_configurations.items()}
    
    def _create_jpy_cluster_primary(self, length):
        """Create the primary JPY cluster strategy - the source of coordinated JPY trading."""
        returns = []
        
        # Advanced JPY strategy that analyzes:
        # 1. BoJ intervention patterns
        # 2. Carry trade flows
        # 3. Risk-on/risk-off dynamics
        # 4. Cross-currency arbitrage
        
        jpy_strength_cycle = 0.0
        carry_trade_flow = 0.0
        intervention_expectation = 0.0
        
        for i in range(length):
            # BoJ intervention cycle (they intervene when USD/JPY gets too strong)
            if i % 25 == 0:  # Every ~25 days
                usd_jpy_level = 140 + np.random.normal(0, 10)  # Simulated USD/JPY level
                if usd_jpy_level > 155:  # BoJ intervention threshold
                    intervention_expectation = -0.02  # Expect JPY strengthening
                elif usd_jpy_level < 130:
                    intervention_expectation = 0.01   # Expect JPY weakening
                else:
                    intervention_expectation *= 0.8   # Decay expectation
            
            # Carry trade dynamics (risk-on/risk-off)
            if i % 15 == 0:  # Every ~15 days
                risk_sentiment = np.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
                if risk_sentiment == 1:  # Risk-on
                    carry_trade_flow = 0.005  # JPY weakens (carry funding)
                elif risk_sentiment == -1:  # Risk-off
                    carry_trade_flow = -0.008  # JPY strengthens (carry unwind)
                else:
                    carry_trade_flow *= 0.9
            
            # Seasonal patterns (fiscal year end, etc.)
            seasonal_factor = 0.003 * np.sin(i * 2 * np.pi / 365 * 4)  # Quarterly pattern
            
            # Cross-JPY arbitrage opportunities
            cross_jpy_arb = 0.0
            if np.random.random() < 0.1:  # 10% chance
                cross_jpy_arb = np.random.normal(0, 0.004)
            
            # Economic data releases (Tokyo CPI, Tankan, etc.)
            econ_data_impact = 0.0
            if i % 30 == 0:  # Monthly data
                data_surprise = np.random.normal(0, 0.006)
                econ_data_impact = data_surprise * 0.5
            
            # Combine all factors
            daily_return = (intervention_expectation * 0.3 + 
                          carry_trade_flow * 0.25 +
                          seasonal_factor * 0.15 +
                          cross_jpy_arb * 0.2 +
                          econ_data_impact * 0.1 +
                          np.random.normal(0, 0.003))
            
            # Add momentum and mean reversion
            if i > 10:
                momentum = np.mean(returns[-5:]) * 0.15
                mean_reversion = -np.mean(returns[-20:]) * 0.05 if i > 20 else 0
                daily_return += momentum + mean_reversion
            
            # Decay factors
            intervention_expectation *= 0.98
            carry_trade_flow *= 0.95
            
            returns.append(daily_return)
        
        return returns
    
    def _create_jpy_cluster_secondary(self, length, correlation, pair_focus):
        """Create JPY cluster secondary strategies - variants that follow the primary with pair-specific variations."""
        primary_returns = self._create_jpy_cluster_primary(length)
        
        secondary_returns = []
        
        # Pair-specific factors
        pair_volatility = {
            'USDJPY': 0.008, 'EURJPY': 0.012, 'GBPJPY': 0.015, 
            'AUDJPY': 0.018, 'CADJPY': 0.014, 'CHFJPY': 0.010, 'NZDJPY': 0.020
        }
        
        pair_beta = {
            'USDJPY': 1.0, 'EURJPY': 1.1, 'GBPJPY': 1.3, 
            'AUDJPY': 1.4, 'CADJPY': 1.2, 'CHFJPY': 0.9, 'NZDJPY': 1.5
        }
        
        vol = pair_volatility.get(pair_focus, 0.012)
        beta = pair_beta.get(pair_focus, 1.0)
        
        # Add execution delays and slippage
        execution_delay = np.random.randint(0, 3)  # 0-2 day delay
        
        for i in range(length):
            # Get delayed primary signal
            primary_signal = primary_returns[max(0, i - execution_delay)]
            
            # Apply correlation and pair-specific beta
            correlated_signal = correlation * beta * primary_signal
            
            # Add pair-specific noise
            pair_noise = np.random.normal(0, vol * (1 - correlation))
            
            # Add execution costs and slippage
            execution_cost = -abs(correlated_signal) * 0.0001  # Bid-ask spread cost
            
            # Add occasional strategy drift
            strategy_drift = 0.0
            if np.random.random() < 0.02:  # 2% chance
                strategy_drift = np.random.normal(0, 0.003)
            
            secondary_return = correlated_signal + pair_noise + execution_cost + strategy_drift
            secondary_returns.append(secondary_return)
        
        return secondary_returns
    
    def _create_correlation_tier_primary(self, length):
        """Create the top-tier correlation primary strategy."""
        returns = []
        
        # Multi-asset, multi-timeframe correlation strategy
        equity_factor = 0.0
        bond_factor = 0.0
        currency_factor = 0.0
        commodity_factor = 0.0
        
        for i in range(length):
            # Rotate factor focus every 30 days
            if i % 30 == 0:
                dominant_factor = np.random.choice(['equity', 'bond', 'currency', 'commodity'])
                
                if dominant_factor == 'equity':
                    equity_factor = np.random.normal(0.01, 0.005)
                elif dominant_factor == 'bond':
                    bond_factor = np.random.normal(0.008, 0.003)
                elif dominant_factor == 'currency':
                    currency_factor = np.random.normal(0.006, 0.004)
                else:  # commodity
                    commodity_factor = np.random.normal(0.012, 0.006)
            
            # Cross-asset momentum
            if i > 20:
                cross_momentum = np.mean(returns[-10:]) * 0.3
            else:
                cross_momentum = 0
            
            # Factor rotation signal
            rotation_signal = (equity_factor * 0.3 + bond_factor * 0.25 + 
                             currency_factor * 0.25 + commodity_factor * 0.2)
            
            daily_return = rotation_signal + cross_momentum + np.random.normal(0, 0.004)
            
            # Decay factors
            equity_factor *= 0.97
            bond_factor *= 0.98
            currency_factor *= 0.96
            commodity_factor *= 0.95
            
            returns.append(daily_return)
        
        return returns
    
    def _create_correlation_tier_derived(self, length, tier_level, correlation):
        """Create tiered derived strategies that follow the primary with increasing lag and decreasing correlation."""
        primary_returns = self._create_correlation_tier_primary(length + tier_level * 2)
        
        derived_returns = []
        lag = tier_level  # Increasing lag with tier level
        
        for i in range(length):
            # Get lagged primary signal
            primary_idx = max(0, i - lag)
            if primary_idx < len(primary_returns):
                primary_signal = primary_returns[primary_idx]
            else:
                primary_signal = 0
            
            # Apply tier-specific correlation decay
            effective_correlation = correlation * (0.95 ** (tier_level - 1))
            
            # Add tier-specific noise
            tier_noise = np.random.normal(0, 0.003 * tier_level)
            
            # Add signal decay
            signal_decay = primary_signal * effective_correlation
            
            derived_return = signal_decay + tier_noise
            derived_returns.append(derived_return)
        
        return derived_returns
    
    def _create_signal_source(self, length):
        """Create the original signal source for transformation chain analysis."""
        returns = []
        
        # High-alpha strategy that feeds into transformation chain
        alpha_signal = 0.0
        
        for i in range(length):
            # Generate high-quality alpha
            if i % 10 == 0:  # New alpha every 10 days
                alpha_signal = np.random.normal(0.015, 0.008)  # High expected return
            
            # Alpha decay
            alpha_signal *= 0.92
            
            # Add implementation costs (this signal has high visibility)
            implementation_cost = -abs(alpha_signal) * 0.02  # High cost due to visibility
            
            daily_return = alpha_signal + implementation_cost + np.random.normal(0, 0.006)
            returns.append(daily_return)
        
        return returns
    
    def _create_signal_transformer(self, length, transform_level, correlation):
        """Create signal transformers that process signals through multiple transformations."""
        source_returns = self._create_signal_source(length + transform_level * 3)
        
        transformer_returns = []
        
        for i in range(length):
            # Get source signal with lag
            source_idx = max(0, i - transform_level)
            if source_idx < len(source_returns):
                source_signal = source_returns[source_idx]
            else:
                source_signal = 0
            
            # Apply signal transformations
            if transform_level == 1:
                # Simple inversion + noise
                transformed_signal = -source_signal * correlation + np.random.normal(0, 0.004)
            elif transform_level == 2:
                # Moving average + scaling
                if i > 5:
                    ma_signal = np.mean([source_returns[max(0, i-j-transform_level)] for j in range(5)])
                    transformed_signal = ma_signal * correlation * 0.8 + np.random.normal(0, 0.005)
                else:
                    transformed_signal = source_signal * correlation * 0.5
            else:  # transform_level == 3
                # Complex transformation
                if i > 10:
                    ema_signal = 0
                    alpha = 0.3
                    for j in range(10):
                        weight = alpha * ((1 - alpha) ** j)
                        signal_idx = max(0, i - j - transform_level)
                        if signal_idx < len(source_returns):
                            ema_signal += weight * source_returns[signal_idx]
                    transformed_signal = ema_signal * correlation + np.random.normal(0, 0.006)
                else:
                    transformed_signal = source_signal * correlation * 0.3
            
            transformer_returns.append(transformed_signal)
        
        return transformer_returns
    
    def _create_signal_terminal_processor(self, length, chain_length, correlation):
        """Create terminal processors of transformed signals."""
        # Chain through multiple transformers
        current_signal = self._create_signal_source(length + chain_length * 2)
        
        for level in range(1, chain_length):
            current_signal = self._create_signal_transformer(length + (chain_length - level) * 2, 
                                                           level, correlation + 0.05 * level)
        
        terminal_returns = []
        
        for i in range(length):
            if i < len(current_signal):
                # Final transformation with additional noise
                end_signal = current_signal[i] * correlation + np.random.normal(0, 0.008)
            else:
                end_signal = np.random.normal(0, 0.008)
            
            terminal_returns.append(end_signal)
        
        return terminal_returns
    
    def _create_strategy_base(self, length):
        """Create the base strategy for variant analysis."""
        returns = []
        
        # Complex but identifiable pattern
        state_a = 0.0
        state_b = 0.0
        
        for i in range(length):
            # Two-state system
            if i % 20 == 0:
                state_a = np.random.normal(0.01, 0.005)
                state_b = np.random.normal(-0.005, 0.003)
            
            # State transition logic
            if state_a > 0 and state_b < 0:
                daily_return = state_a * 0.7 + state_b * 0.3
            else:
                daily_return = state_a * 0.3 + state_b * 0.7
            
            # Add momentum
            if i > 5:
                momentum = np.mean(returns[-3:]) * 0.2
                daily_return += momentum
            
            # State decay
            state_a *= 0.95
            state_b *= 0.97
            
            daily_return += np.random.normal(0, 0.003)
            returns.append(daily_return)
        
        return returns
    
    def _create_strategy_variant(self, length, variant_type):
        """Create strategy variants that implement the same logic differently."""
        base_returns = self._create_strategy_base(length)
        
        if variant_type == 'invert_signals':
            # Invert signals but keep same logic
            return [-x * 0.8 + np.random.normal(0, 0.002) for x in base_returns]
        
        elif variant_type == 'delay_signals':
            # Delay signals but keep same logic
            delayed = [0, 0] + base_returns[:-2]
            return [x + np.random.normal(0, 0.002) for x in delayed]
        
        elif variant_type == 'amplify_signals':
            # Amplify signals but keep same logic
            return [x * 1.3 + np.random.normal(0, 0.003) for x in base_returns]
        
        elif variant_type == 'compress_signals':
            # Compress signals but keep same logic
            return [x * 0.6 + np.random.normal(0, 0.001) for x in base_returns]
        
        else:  # noise_inject
            # Inject noise but keep same logic
            noisy_returns = []
            for i, x in enumerate(base_returns):
                if i % 5 == 0:  # Every 5th day
                    noise = np.random.normal(0, 0.01)
                else:
                    noise = np.random.normal(0, 0.002)
                noisy_returns.append(x + noise)
            return noisy_returns
    
    def _create_cv_optimization_strategy(self, length, phase):
        """Create strategies that optimize for specific cross-validation phases."""
        returns = []
        
        if phase == 'train':
            # Overfit to training period
            for i in range(length):
                # Create overfitted pattern
                pattern_signal = 0.01 * np.sin(i * 0.1) + 0.005 * np.cos(i * 0.05)
                daily_return = pattern_signal + np.random.normal(0, 0.001)
                returns.append(daily_return)
        
        elif phase == 'validate':
            # Optimize for validation by being similar to training but not identical
            for i in range(length):
                pattern_signal = 0.008 * np.sin(i * 0.11) + 0.004 * np.cos(i * 0.048)
                daily_return = pattern_signal + np.random.normal(0, 0.002)
                returns.append(daily_return)
        
        else:  # test
            # Different pattern optimized for test phase
            for i in range(length):
                pattern_signal = 0.006 * np.sin(i * 0.09) + 0.003 * np.cos(i * 0.052)
                daily_return = pattern_signal + np.random.normal(0, 0.003)
                returns.append(daily_return)
        
        return returns
    
    def _create_regime_specialist(self, length, regime_type):
        """Create regime-specific specialists."""
        returns = []
        
        if regime_type == 'bull_specialist':
            # Only profitable in bull markets
            for i in range(length):
                market_regime = 1 if np.sin(i * 0.02) > 0 else -1
                if market_regime == 1:  # Bull market
                    daily_return = np.random.normal(0.015, 0.008)
                else:  # Bear market
                    daily_return = np.random.normal(-0.005, 0.012)
                returns.append(daily_return)
        
        elif regime_type == 'bear_specialist':
            # Only profitable in bear markets
            for i in range(length):
                market_regime = 1 if np.sin(i * 0.02) > 0 else -1
                if market_regime == -1:  # Bear market
                    daily_return = np.random.normal(0.012, 0.006)
                else:  # Bull market
                    daily_return = np.random.normal(-0.003, 0.008)
                returns.append(daily_return)
        
        elif regime_type == 'vol_specialist':
            # Only profitable in high volatility
            for i in range(length):
                vol_regime = abs(np.sin(i * 0.03))
                if vol_regime > 0.7:  # High vol
                    daily_return = np.random.normal(0.01, 0.02)
                else:  # Low vol
                    daily_return = np.random.normal(-0.002, 0.003)
                returns.append(daily_return)
        
        else:  # crisis_specialist
            # Only profitable during crises
            for i in range(length):
                if i % 100 < 10:  # Crisis period (10 days every 100)
                    daily_return = np.random.normal(0.02, 0.015)
                else:  # Normal period
                    daily_return = np.random.normal(-0.001, 0.005)
                returns.append(daily_return)
        
        return returns
    
    def _create_temporal_strategy(self, length, speed):
        """Create temporal-based trading strategies."""
        returns = []
        
        speed_multipliers = {
            'microsecond': 0.0001, 'millisecond': 0.0005, 
            'second': 0.002, 'minute': 0.008
        }
        
        base_return = speed_multipliers.get(speed, 0.002)
        
        for i in range(length):
            # High frequency, small profits
            daily_return = np.random.normal(base_return, base_return * 0.5)
            
            # Occasional tech failures
            if np.random.random() < 0.01:  # 1% chance
                daily_return = np.random.normal(-base_return * 10, base_return * 5)
            
            returns.append(daily_return)
        
        return returns
    
    def _create_info_strategy(self, length, strategy_type):
        """Create information advantage strategies."""
        returns = []
        
        if strategy_type == 'news_front_run':
            # Front-run news releases
            for i in range(length):
                if i % 7 == 0:  # Weekly news
                    # Front-run positive news
                    daily_return = np.random.normal(0.008, 0.003)
                elif (i + 1) % 7 == 0:  # Day after news
                    # Reverse position
                    daily_return = np.random.normal(-0.004, 0.002)
                else:
                    daily_return = np.random.normal(0, 0.003)
                returns.append(daily_return)
        
        elif strategy_type == 'insider_mimic':
            # Mimic insider trading patterns
            for i in range(length):
                if i % 30 == 0:  # Monthly insider activity
                    daily_return = np.random.normal(0.012, 0.005)
                elif i % 30 < 5:  # Follow-up trading
                    daily_return = np.random.normal(0.004, 0.003)
                else:
                    daily_return = np.random.normal(0, 0.002)
                returns.append(daily_return)
        
        else:  # order_flow_read
            # Read order flow patterns
            for i in range(length):
                # Simulate order flow reading
                flow_signal = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
                daily_return = flow_signal * 0.003 + np.random.normal(0, 0.004)
                returns.append(daily_return)
        
        return returns
    
    def _create_network_hub(self, length):
        """Create the central hub of a trading network."""
        returns = []
        
        # Hub generates original signals
        signal_strength = 0.0
        
        for i in range(length):
            # Generate network signals
            if i % 15 == 0:  # New signal every 15 days
                signal_strength = np.random.normal(0.012, 0.006)
            
            # Network amplification effect
            network_effect = signal_strength * 1.2 if signal_strength > 0 else signal_strength * 0.8
            
            daily_return = network_effect + np.random.normal(0, 0.004)
            
            # Signal decay
            signal_strength *= 0.92
            
            returns.append(daily_return)
        
        return returns
    
    def _create_network_node(self, length, node_id, correlation):
        """Create network nodes that follow the hub."""
        hub_returns = self._create_network_hub(length + node_id)
        
        node_returns = []
        
        for i in range(length):
            # Follow hub with delay and correlation
            hub_idx = max(0, i - node_id)
            if hub_idx < len(hub_returns):
                hub_signal = hub_returns[hub_idx]
            else:
                hub_signal = 0
            
            # Network distance decay
            distance_decay = correlation * (0.95 ** node_id)
            
            # Node-specific noise
            node_noise = np.random.normal(0, 0.003 * (1 + node_id * 0.2))
            
            node_return = hub_signal * distance_decay + node_noise
            node_returns.append(node_return)
        
        return node_returns
    
    def _create_adaptive_learner(self, length, adapt_speed):
        """Create adaptive learning strategies that evolve over time."""
        returns = []
        
        # Learning parameters
        if adapt_speed == 'fast_adapt':
            learning_rate = 0.1
            memory_decay = 0.8
        elif adapt_speed == 'medium_adapt':
            learning_rate = 0.05
            memory_decay = 0.9
        else:  # slow_adapt
            learning_rate = 0.02
            memory_decay = 0.95
        
        strategy_weights = np.random.random(5)  # 5 different sub-strategies
        strategy_weights /= np.sum(strategy_weights)
        
        for i in range(length):
            # Generate returns from multiple strategies
            sub_strategies = [
                np.random.normal(0.002, 0.008),  # Strategy 1
                np.random.normal(0.001, 0.006),  # Strategy 2
                np.random.normal(0.003, 0.010),  # Strategy 3
                np.random.normal(-0.001, 0.005), # Strategy 4
                np.random.normal(0.004, 0.012),  # Strategy 5
            ]
            
            # Combined return based on current weights
            daily_return = sum(w * s for w, s in zip(strategy_weights, sub_strategies))
            
            # Update weights based on performance (simplified learning)
            if i > 10:
                recent_performance = np.mean(returns[-10:])
                if recent_performance > 0:
                    # Increase weights of profitable strategies
                    performance_gradient = [s - np.mean(sub_strategies) for s in sub_strategies]
                    strategy_weights += learning_rate * np.array(performance_gradient)
                    strategy_weights = np.maximum(strategy_weights, 0.01)  # Minimum weight
                    strategy_weights /= np.sum(strategy_weights)  # Normalize
            
            # Memory decay
            strategy_weights = memory_decay * strategy_weights + (1 - memory_decay) * np.ones(5) / 5
            
            returns.append(daily_return)
        
        return returns
    
    def _create_microstructure_strategy(self, length, strategy_type):
        """Create market microstructure strategies."""
        returns = []
        
        if strategy_type == 'bid_ask_spread':
            # Exploit bid-ask spread patterns
            for i in range(length):
                # Simulate spread capture
                spread_capture = np.random.uniform(0.0001, 0.0005)
                
                # Volume dependency
                volume_factor = np.random.lognormal(0, 0.5)
                adjusted_capture = spread_capture * min(volume_factor, 2.0)
                
                # Market impact cost
                market_impact = -adjusted_capture * 0.3
                
                daily_return = adjusted_capture + market_impact + np.random.normal(0, 0.001)
                returns.append(daily_return)
        
        elif strategy_type == 'order_book_depth':
            # Exploit order book depth patterns
            for i in range(length):
                # Deep book advantage
                depth_advantage = np.random.normal(0.0003, 0.0002)
                
                # Liquidity provision reward
                lp_reward = np.random.exponential(0.0002)
                
                # Adverse selection cost
                adverse_selection = -np.random.exponential(0.0001)
                
                daily_return = depth_advantage + lp_reward + adverse_selection
                returns.append(daily_return)
        
        else:  # tick_size_optimization
            # Exploit tick size constraints
            for i in range(length):
                # Sub-penny pricing advantage
                tick_advantage = np.random.uniform(0.00005, 0.0002)
                
                # Queue position advantage
                queue_advantage = np.random.normal(0.0001, 0.00005)
                
                # Regulatory risk
                regulatory_risk = -np.random.exponential(0.00003)
                
                daily_return = tick_advantage + queue_advantage + regulatory_risk
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

    def test_jpy_cluster_correlation_detection(self):
        """Test complete detection of the JPY cluster correlation patterns."""
        jpy_cluster = {f'jpy_cluster_{name}': ledger for name, ledger in self.pattern_configurations.items() 
                       if name.startswith('jpy_cluster')}
        
        penalties = LedgerUtils.orthogonality_penalty(jpy_cluster)
        
        # ALL JPY cluster members should receive substantial penalties
        penalty_values = list(penalties.values())
        min_penalty = min(penalty_values)
        avg_penalty = np.mean(penalty_values)
        
        self.assertGreater(min_penalty, 0.5, 
                         "Even the least penalized JPY cluster member should receive substantial penalty")
        self.assertGreater(avg_penalty, 0.6, 
                         "Average JPY cluster penalty should be very high")
        
        # Test correlation detection
        returns = {name: LedgerUtils.daily_return_log(ledger) 
                  for name, ledger in jpy_cluster.items()}
        corr_matrix, mean_correlations = Orthogonality.correlation_matrix(returns)
        
        if not corr_matrix.empty and mean_correlations:
            correlations = [abs(c) for c in mean_correlations.values()]
            avg_correlation = np.mean(correlations)
            
            self.assertGreater(avg_correlation, 0.4, 
                             "JPY cluster should show high average correlation")

    def test_multi_tier_correlation_detection(self):
        """Test detection of multi-tier correlation hierarchies."""
        tier_correlations = {name: ledger for name, ledger in self.pattern_configurations.items() 
                            if name.startswith('correlation_tier')}
        
        penalties = LedgerUtils.orthogonality_penalty(tier_correlations)
        
        # All tiers should be detected
        self.assertEqual(len(penalties), 6)  # 1 primary + 5 derived
        
        # Primary should have high penalty (most correlated with derived)
        primary_penalty = penalties['correlation_tier_1_primary']
        self.assertGreater(primary_penalty, 0.4, 
                         "Tier primary should receive substantial penalty")
        
        # All derived strategies should also be penalized
        derived_penalties = [penalties[name] for name in penalties.keys() 
                           if 'derived' in name]
        avg_derived_penalty = np.mean(derived_penalties)
        
        self.assertGreater(avg_derived_penalty, 0.3, 
                         "Tier derived strategies should receive meaningful penalties")

    def test_signal_transformation_detection(self):
        """Test detection of signal transformation through intermediaries."""
        transformation_chain = {name: ledger for name, ledger in self.pattern_configurations.items() 
                               if any(x in name for x in ['signal_source', 'transformer', 'terminal_processor'])}
        
        penalties = LedgerUtils.orthogonality_penalty(transformation_chain)
        
        # Should detect the transformation chain
        source_penalty = penalties['signal_source']
        transformer_penalties = [penalties[name] for name in penalties.keys() if 'transformer' in name]
        terminal_penalties = [penalties[name] for name in penalties.keys() if 'terminal_processor' in name]
        
        # Source should be detected
        self.assertGreater(source_penalty, 0.2, 
                         "Signal source should be penalized")
        
        # At least some transformers should be detected
        detected_transformers = sum(1 for p in transformer_penalties if p > 0.2)
        self.assertGreater(detected_transformers, 0, 
                         "Should detect some signal transformers")

    def test_strategy_variant_detection(self):
        """Test detection of strategy variants (same logic, different implementations)."""
        strategy_variants = {name: ledger for name, ledger in self.pattern_configurations.items() 
                           if name.startswith('strategy')}
        
        penalties = LedgerUtils.orthogonality_penalty(strategy_variants)
        
        # Should detect that these are all variants of the same strategy
        penalty_values = list(penalties.values())
        avg_penalty = np.mean(penalty_values)
        
        self.assertGreater(avg_penalty, 0.3, 
                         "Strategy variants should be detected as similar")
        
        # Base strategy should be detected as correlated with variants
        base_penalty = penalties['strategy_base']
        self.assertGreater(base_penalty, 0.3, 
                         "Base strategy should be penalized")

    def test_cross_validation_optimization_detection(self):
        """Test detection of cross-validation optimization patterns."""
        cv_optimizers = {name: ledger for name, ledger in self.pattern_configurations.items() 
                        if name.startswith('cv_optimizer')}
        
        penalties = LedgerUtils.orthogonality_penalty(cv_optimizers)
        
        # Should detect correlation despite different phases
        penalty_values = list(penalties.values())
        avg_penalty = np.mean(penalty_values)
        
        self.assertGreater(avg_penalty, 0.2, 
                         "Cross-validation optimization should be detected")

    def test_regime_specialization_detection(self):
        """Test detection of regime-specific specialization patterns."""
        regime_specialists = {name: ledger for name, ledger in self.pattern_configurations.items() 
                             if name.startswith('regime_specialist')}
        
        penalties = LedgerUtils.orthogonality_penalty(regime_specialists)
        
        # Should detect if these are coordinated despite different regimes
        penalty_values = list(penalties.values())
        self.assertTrue(all(p >= 0.0 for p in penalty_values), 
                       "All regime specialists should have valid penalties")

    def test_temporal_strategy_detection(self):
        """Test detection of temporal strategy coordination patterns."""
        temporal_strategies = {name: ledger for name, ledger in self.pattern_configurations.items() 
                              if name.startswith('temporal_strategy')}
        
        penalties = LedgerUtils.orthogonality_penalty(temporal_strategies)
        
        # Should detect similarity in temporal arbitrage patterns
        penalty_values = list(penalties.values())
        avg_penalty = np.mean(penalty_values)
        
        self.assertGreater(avg_penalty, 0.1, 
                         "Temporal strategies should show some correlation")

    def test_information_advantage_detection(self):
        """Test detection of information advantage strategy patterns."""
        info_strategies = {name: ledger for name, ledger in self.pattern_configurations.items() 
                          if name.startswith('info_strategy')}
        
        penalties = LedgerUtils.orthogonality_penalty(info_strategies)
        
        # Should detect coordination in information exploitation
        penalty_values = list(penalties.values())
        self.assertTrue(all(p >= 0.0 for p in penalty_values), 
                       "All info strategies should have valid penalties")

    def test_network_correlation_detection(self):
        """Test detection of network-based correlation patterns."""
        network_patterns = {name: ledger for name, ledger in self.pattern_configurations.items() 
                           if name.startswith('network')}
        
        penalties = LedgerUtils.orthogonality_penalty(network_patterns)
        
        # Hub should be highly penalized (most correlated)
        hub_penalty = penalties['network_hub']
        self.assertGreater(hub_penalty, 0.4, 
                         "Network hub should receive substantial penalty")
        
        # Nodes should also be penalized
        node_penalties = [penalties[name] for name in penalties.keys() if 'node' in name]
        avg_node_penalty = np.mean(node_penalties)
        
        self.assertGreater(avg_node_penalty, 0.2, 
                         "Network nodes should receive meaningful penalties")

    def test_adaptive_learning_detection(self):
        """Test detection of adaptive learning correlation patterns."""
        adaptive_learners = {name: ledger for name, ledger in self.pattern_configurations.items() 
                            if name.startswith('adaptive_learner')}
        
        penalties = LedgerUtils.orthogonality_penalty(adaptive_learners)
        
        # Should detect if adaptive learners converge to similar strategies
        penalty_values = list(penalties.values())
        avg_penalty = np.mean(penalty_values)
        
        self.assertGreater(avg_penalty, 0.1, 
                         "Adaptive learners should show some correlation")

    def test_microstructure_strategy_detection(self):
        """Test detection of market microstructure strategy patterns."""
        microstructure_strategies = {name: ledger for name, ledger in self.pattern_configurations.items() 
                                     if name.startswith('microstructure')}
        
        penalties = LedgerUtils.orthogonality_penalty(microstructure_strategies)
        
        # Should detect similarity in microstructure exploitation
        penalty_values = list(penalties.values())
        avg_penalty = np.mean(penalty_values)
        
        self.assertGreater(avg_penalty, 0.15, 
                         "Microstructure strategies should show correlation")

    def test_comprehensive_pattern_scenario(self):
        """Test system behavior with all pattern configurations simultaneously."""
        # Use all pattern configurations
        all_patterns = self.pattern_configurations.copy()
        
        penalties = LedgerUtils.orthogonality_penalty(all_patterns)
        
        # System should handle comprehensive pattern analysis
        self.assertEqual(len(penalties), len(all_patterns), 
                        "Should calculate penalties for all patterns")
        
        # All penalties should be valid
        for name, penalty in penalties.items():
            self.assertGreaterEqual(penalty, 0.0, f"Penalty for {name} should be non-negative")
            self.assertLessEqual(penalty, 1.0, f"Penalty for {name} should not exceed 1.0")
            self.assertTrue(np.isfinite(penalty), f"Penalty for {name} should be finite")
        
        # Should detect substantial correlation patterns
        penalty_values = list(penalties.values())
        avg_penalty = np.mean(penalty_values)
        
        self.assertGreater(avg_penalty, 0.3, 
                         "Average penalty should be substantial in comprehensive pattern scenario")

    def test_correlation_vs_innovation_differentiation(self):
        """Test that the system can differentiate between correlation patterns and legitimate innovation."""
        # Mix correlation patterns and innovation
        mixed_strategies = {}
        
        # Add some correlation patterns
        mixed_strategies.update({
            'jpy_pattern': self.pattern_configurations['jpy_cluster_primary'],
            'network_pattern': self.pattern_configurations['network_hub'],
            'signal_pattern': self.pattern_configurations['signal_source'],
        })
        
        # Add legitimate innovation (create new strategies)
        innovation_patterns = self._create_innovation_patterns()
        mixed_strategies.update(innovation_patterns)
        
        penalties = LedgerUtils.orthogonality_penalty(mixed_strategies)
        
        # Pattern correlations should have higher penalties than innovators
        pattern_penalties = [penalties['jpy_pattern'], penalties['network_pattern'], 
                           penalties['signal_pattern']]
        innovator_penalties = [penalties[name] for name in innovation_patterns.keys()]
        
        pattern_avg = np.mean(pattern_penalties)
        innovator_avg = np.mean(innovator_penalties)
        
        self.assertGreater(pattern_avg, innovator_avg, 
                         "Correlation patterns should be penalized more than innovators")

    def _create_innovation_patterns(self):
        """Create legitimate innovation patterns for comparison."""
        base_length = max(ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N + 30, 90)
        
        innovations = {
            'quantum_inspired': self._create_quantum_inspired_strategy(base_length),
            'esg_momentum': self._create_esg_momentum_strategy(base_length),
            'crypto_defi': self._create_crypto_defi_strategy(base_length),
            'climate_risk': self._create_climate_risk_strategy(base_length),
            'social_sentiment': self._create_social_sentiment_strategy(base_length),
        }
        
        return {name: self._create_ledger_from_returns(returns) 
                for name, returns in innovations.items()}
    
    def _create_quantum_inspired_strategy(self, length):
        """Create quantum-inspired trading strategy."""
        returns = []
        
        # Quantum superposition-like states
        state_vectors = [np.random.random(3) for _ in range(5)]
        
        for i in range(length):
            # Quantum interference pattern
            interference = sum(np.sin(i * 0.01 * j) * state_vectors[j % 5][0] 
                             for j in range(5)) / 5
            
            # Entanglement-like correlation
            entanglement = np.prod([state_vectors[j % 5][1] for j in range(3)]) ** (1/3)
            
            # Uncertainty principle
            momentum = np.random.normal(0, 0.005)
            position = interference * 0.01
            uncertainty = -abs(momentum * position) * 100  # Uncertainty cost
            
            daily_return = position + momentum + uncertainty
            
            # Update quantum states
            if i % 10 == 0:
                state_vectors = [np.random.random(3) for _ in range(5)]
            
            returns.append(daily_return)
        
        return returns
    
    def _create_esg_momentum_strategy(self, length):
        """Create ESG momentum strategy."""
        returns = []
        
        esg_score = 0.5
        
        for i in range(length):
            # ESG events
            if i % 20 == 0:  # ESG reporting
                esg_surprise = np.random.normal(0, 0.2)
                esg_score = np.clip(esg_score + esg_surprise, 0, 1)
            
            # ESG momentum
            esg_momentum = (esg_score - 0.5) * 0.02
            
            # Regulatory impact
            regulatory_impact = 0
            if np.random.random() < 0.05:  # 5% chance
                regulatory_impact = np.random.normal(0.003, 0.002)
            
            # Consumer sentiment
            consumer_sentiment = np.random.normal(0, 0.003)
            
            daily_return = esg_momentum + regulatory_impact + consumer_sentiment
            
            # ESG score decay
            esg_score = 0.9 * esg_score + 0.1 * 0.5
            
            returns.append(daily_return)
        
        return returns
    
    def _create_crypto_defi_strategy(self, length):
        """Create crypto DeFi strategy."""
        returns = []
        
        liquidity_pool_yield = 0.05  # Annual yield
        impermanent_loss_risk = 0.0
        
        for i in range(length):
            # Yield farming returns
            daily_yield = liquidity_pool_yield / 365
            
            # Impermanent loss
            if i % 30 == 0:  # Monthly rebalancing
                price_divergence = np.random.normal(0, 0.15)
                impermanent_loss_risk = abs(price_divergence) ** 2 / 8  # Simplified IL formula
            
            # Smart contract risk
            smart_contract_risk = 0
            if np.random.random() < 0.001:  # 0.1% chance of exploit
                smart_contract_risk = -np.random.uniform(0.1, 0.5)  # 10-50% loss
            
            # Gas fee costs
            gas_costs = -np.random.exponential(0.0001)
            
            # Token rewards
            token_rewards = np.random.exponential(0.001)
            
            daily_return = daily_yield - impermanent_loss_risk + smart_contract_risk + gas_costs + token_rewards
            returns.append(daily_return)
        
        return returns
    
    def _create_climate_risk_strategy(self, length):
        """Create climate risk strategy."""
        returns = []
        
        climate_trend = 0.0
        
        for i in range(length):
            # Climate events
            if i % 45 == 0:  # Seasonal climate events
                climate_event = np.random.choice([
                    ('drought', -0.02), ('flood', -0.015), ('hurricane', -0.025),
                    ('heatwave', -0.01), ('normal', 0.005)
                ], p=[0.15, 0.15, 0.1, 0.15, 0.45])
                
                climate_trend += climate_event[1]
            
            # Carbon pricing
            carbon_price_change = np.random.normal(0, 0.002)
            
            # Green tech innovation
            green_innovation = 0
            if np.random.random() < 0.03:  # 3% chance
                green_innovation = np.random.normal(0.008, 0.004)
            
            # Physical risk materialization
            physical_risk = np.random.exponential(0.001) * (-1 if np.random.random() < 0.3 else 0)
            
            daily_return = climate_trend * 0.1 + carbon_price_change + green_innovation + physical_risk
            
            # Climate trend persistence
            climate_trend *= 0.98
            
            returns.append(daily_return)
        
        return returns
    
    def _create_social_sentiment_strategy(self, length):
        """Create social sentiment strategy."""
        returns = []
        
        sentiment_momentum = 0.0
        
        for i in range(length):
            # Social media sentiment
            if i % 7 == 0:  # Weekly sentiment update
                sentiment_change = np.random.normal(0, 0.3)
                sentiment_momentum += sentiment_change * 0.2
            
            # Viral events
            viral_impact = 0
            if np.random.random() < 0.02:  # 2% chance
                viral_impact = np.random.normal(0, 0.015)
            
            # Influencer effects
            influencer_effect = 0
            if np.random.random() < 0.05:  # 5% chance
                influencer_effect = np.random.normal(0, 0.008)
            
            # Sentiment decay
            sentiment_decay = -abs(sentiment_momentum) * 0.1
            
            daily_return = sentiment_momentum * 0.01 + viral_impact + influencer_effect + sentiment_decay
            
            # Momentum decay
            sentiment_momentum *= 0.95
            
            returns.append(daily_return)
        
        return returns

    def test_stress_test_parameter_variation(self):
        """Test system robustness against parameter variation scenarios."""
        original_params = {
            'corr_weight': ValiConfig.ORTHOGONALITY_CORRELATION_WEIGHT,
            'pref_weight': ValiConfig.ORTHOGONALITY_PREFERENCE_WEIGHT,
            'intensity': ValiConfig.ORTHOGONALITY_DIVERGING_INTENSITY,
        }
        
        try:
            # Test extreme parameter combinations
            extreme_params = [
                (0.99, 0.01, 0.01),  # Almost all correlation
                (0.01, 0.99, 0.01),  # Almost all preference
                (0.5, 0.5, 0.99),    # High intensity
                (0.1, 0.1, 10.0),    # Extreme intensity (invalid)
            ]
            
            jpy_cluster = {f'jpy_cluster_{name}': ledger for name, ledger in self.pattern_configurations.items() 
                          if name.startswith('jpy_cluster_primary') or name.startswith('jpy_cluster_secondary')}
            
            for corr_w, pref_w, intensity in extreme_params:
                ValiConfig.ORTHOGONALITY_CORRELATION_WEIGHT = corr_w
                ValiConfig.ORTHOGONALITY_PREFERENCE_WEIGHT = pref_w
                ValiConfig.ORTHOGONALITY_DIVERGING_INTENSITY = intensity
                
                try:
                    penalties = LedgerUtils.orthogonality_penalty(jpy_cluster)
                    
                    # System should remain stable
                    penalty_values = list(penalties.values())
                    self.assertTrue(all(0 <= p <= 1 for p in penalty_values), 
                                  f"Penalties should remain valid with params {corr_w}, {pref_w}, {intensity}")
                    
                    # Should still detect JPY cluster
                    avg_penalty = np.mean(penalty_values)
                    self.assertGreater(avg_penalty, 0.1, 
                                     f"Should still detect JPY cluster with params {corr_w}, {pref_w}, {intensity}")
                
                except Exception as e:
                    # System should handle invalid parameters gracefully
                    self.assertIn("intensity", str(e).lower(), 
                                f"Should handle invalid intensity parameter gracefully")
        
        finally:
            # Restore original parameters
            ValiConfig.ORTHOGONALITY_CORRELATION_WEIGHT = original_params['corr_weight']
            ValiConfig.ORTHOGONALITY_PREFERENCE_WEIGHT = original_params['pref_weight']
            ValiConfig.ORTHOGONALITY_DIVERGING_INTENSITY = original_params['intensity']

    def test_stress_test_data_quality_scenarios(self):
        """Test resilience against data quality edge cases."""
        # Create data quality edge case scenarios
        data_scenarios = {}
        
        # NaN injection scenario
        nan_scenario = [0.01, 0.02, float('nan'), 0.015, float('inf'), 0.01] * 15
        data_scenarios['nan_generator'] = self._create_ledger_from_returns(nan_scenario)
        
        # Extreme outlier scenario
        outlier_scenario = [0.01] * 80
        outlier_scenario[40] = 100.0  # Extreme outlier
        data_scenarios['outlier_generator'] = self._create_ledger_from_returns(outlier_scenario)
        
        # Zero variance scenario
        zero_var_scenario = [0.01] * 90
        data_scenarios['zero_var_generator'] = self._create_ledger_from_returns(zero_var_scenario)
        
        # Add a normal strategy for comparison
        normal_strategy = [np.random.normal(0.01, 0.005) for _ in range(90)]
        data_scenarios['normal_trader'] = self._create_ledger_from_returns(normal_strategy)
        
        # System should handle gracefully
        try:
            penalties = LedgerUtils.orthogonality_penalty(data_scenarios)
            
            # Should return valid penalties for all miners
            self.assertEqual(len(penalties), 4)
            
            for name, penalty in penalties.items():
                self.assertTrue(np.isfinite(penalty), f"Penalty for {name} should be finite")
                self.assertGreaterEqual(penalty, 0.0, f"Penalty for {name} should be non-negative")
                self.assertLessEqual(penalty, 1.0, f"Penalty for {name} should not exceed 1.0")
        
        except Exception as e:
            self.fail(f"System should handle data quality scenarios gracefully, but got: {e}")

    def test_stress_test_computational_complexity_scenario(self):
        """Test resilience against computational complexity scenarios."""
        # Create many highly correlated strategies to stress the system
        complexity_scenario = {}
        
        base_strategy = self._create_jpy_cluster_primary(100)
        
        # Create 50 nearly identical strategies
        for i in range(50):
            # Add tiny variations to create computational burden
            variant = [x + np.random.normal(0, 0.0001) for x in base_strategy]
            complexity_scenario[f'complexity_generator_{i}'] = self._create_ledger_from_returns(variant)
        
        import time
        start_time = time.time()
        
        try:
            penalties = LedgerUtils.orthogonality_penalty(complexity_scenario)
            execution_time = time.time() - start_time
            
            # Should complete in reasonable time
            self.assertLess(execution_time, 60.0, "Should complete complexity scenario test within 60 seconds")
            
            # Should detect all as highly correlated
            penalty_values = list(penalties.values())
            avg_penalty = np.mean(penalty_values)
            
            self.assertGreater(avg_penalty, 0.5, "Should detect complexity scenario as highly correlated")
        
        except Exception as e:
            self.fail(f"System should handle computational complexity scenarios, but got: {e}")

    def test_final_pattern_analysis_validation(self):
        """Final comprehensive test of pattern analysis capabilities."""
        # Combine the most sophisticated patterns
        comprehensive_patterns = {
            'jpy_cluster_primary': self.pattern_configurations['jpy_cluster_primary'],
            'jpy_cluster_secondary_1': self.pattern_configurations['jpy_cluster_secondary_1'],
            'jpy_cluster_secondary_2': self.pattern_configurations['jpy_cluster_secondary_2'],
            'network_hub': self.pattern_configurations['network_hub'],
            'network_node_1': self.pattern_configurations['network_node_1'],
            'signal_source': self.pattern_configurations['signal_source'],
            'transformer_1': self.pattern_configurations['transformer_1'],
            'strategy_base': self.pattern_configurations['strategy_base'],
            'strategy_variant_1': self.pattern_configurations['strategy_variant_1'],
            'adaptive_learner_1': self.pattern_configurations['adaptive_learner_1'],
        }
        
        penalties = LedgerUtils.orthogonality_penalty(comprehensive_patterns)
        
        # This is the comprehensive test - the system should detect and heavily penalize
        # the most sophisticated correlation patterns
        
        penalty_values = list(penalties.values())
        min_penalty = min(penalty_values)
        avg_penalty = np.mean(penalty_values)
        max_penalty = max(penalty_values)
        
        # All sophisticated patterns should receive substantial penalties
        self.assertGreater(min_penalty, 0.3, 
                         "Even least penalized sophisticated pattern should receive substantial penalty")
        self.assertGreater(avg_penalty, 0.5, 
                         "Average penalty for sophisticated patterns should be very high")
        self.assertGreater(max_penalty, 0.7, 
                         "Maximum penalty should indicate strong correlation detection")
        
        # Standard deviation should not be too high (all should be similarly penalized)
        penalty_std = np.std(penalty_values)
        self.assertLess(penalty_std, 0.3, 
                       "Sophisticated patterns should be similarly penalized")
        
        print(f"\n=== FINAL PATTERN ANALYSIS VALIDATION ===")
        print(f"Penalties - Min: {min_penalty:.3f}, Avg: {avg_penalty:.3f}, Max: {max_penalty:.3f}, Std: {penalty_std:.3f}")
        print(f"✅ Comprehensive pattern correlation analysis successfully completed!")