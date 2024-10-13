export interface Challenge {
  passing: boolean,
  target: number,
  value: number
}

export interface ChallengePeriod {
  remaining_time_ms: number;
  start_time_ms: number;
  status: "testing" | "success";
  positions: Challenge;
  return: Challenge;
  return_ratio: Challenge;
  unrealized_ratio: Challenge;
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
}

export interface Scores {
  risk_adjusted_return: Score;
  short_risk_adjusted_return: Score;
  omega: Score;
  sharpe: Score;
}

export interface Penalties {
  biweekly: number;
  daily: number;
  drawdown: number;
  returns_ratio: number;
  time_consistency: number;
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
  positions: Position[]
  thirty_day_returns: number;
  all_time_returns: number;
  n_positions: number;
  percentage_profitable: number;
}

export interface MinerData {
  statistics: Statistics;
  positions: Record<string, Positions>;
}