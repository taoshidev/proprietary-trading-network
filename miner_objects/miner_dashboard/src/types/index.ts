export interface Challenge {
  percentile: number;
  target_score: number;
  value: number;
  target_percentile: number;
}

export interface ChallengeMetric {
  omega: Challenge;
  overall: Challenge;
  return_long: Challenge;
  return_short: Challenge;
  sharpe_ratio: Challenge;
  sortino: Challenge;
  statistical_confidence: Challenge;
}

export interface ChallengePeriod {
  remaining_time_ms: number;
  start_time_ms: number;
  status: "testing" | "success";
  scores: ChallengeMetric;
}

export interface Source {
  close: number;
  high: number;
  lag_ms: number;
  low: number;
  open: number;
  source: string;
  start_ms: number;
  timespan_ms: number;
  volume: number;
}

export interface Order {
  leverage: number;
  order_type: string;
  order_uuid: string;
  price: number;
  price_sources: Source[];
  processed_ms: number;
  trade_pair: TradePair;
}

export type TradePair = [string, string, number, number, number, string];

export interface Position {
  average_entry_price: number;
  close_ms: number;
  current_return: number;
  is_closed_position: boolean;
  miner_hotkey: string;
  net_leverage: number;
  open_ms: number;
  orders: Order[];
  position_type: string;
  position_uuid: string;
  return_at_close: number;
  trade_pair: TradePair[];
}

export interface Score {
  value: number;
  rank: number;
  percentile: number;
  overall_contribution: number;
}

export interface Scores {
  return: Score;
  omega: Score;
  sortino: Score;
  statistical_confidence: Score;
  sharpe: Score;
  calmar: Score;
}

// unused
export interface Penalties {
  drawdown_threshold: number;
  martingale: number;
  total: number;
}

export interface Engagement {
  n_positions: number;
  position_duration: number;
  checkpoint_durations: number;
}

export interface Checkpoint {
  mdd: number;
}

export interface PenalizedScores {
  omega: Score;
  sharpe: Score;
  risk_adjusted_return: Score;
  short_risk_adjusted_return: Score;
  calmar: Score;
  "short-calmar": Score;
}

export interface Drawdowns {
  approximate: number;
  effective: number;
  recent: number;
}

export interface StatisticsData {
  hotkey: string;
  penalties: Penalties;
  penalized_scores: PenalizedScores;
  engagement: Engagement;
  drawdowns: Drawdowns;
  weight: Score;
  scores: Scores;
  checkpoints: Checkpoint[];
  challengeperiod: ChallengePeriod;
}

export interface Statistics {
  data: StatisticsData[];
}

export interface Positions {
  positions: Position[];
  thirty_day_returns: number;
  all_time_returns: number;
  n_positions: number;
  percentage_profitable: number;
}

export interface MinerData {
  statistics: Statistics;
  positions: Record<string, Positions>;
}
