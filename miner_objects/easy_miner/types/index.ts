export interface Config {
  exchange: string;
  apiKey: string;
  secret: string;
  demo: boolean;
}

export interface Signal {
  trade_pair: string;
  order_type: string;
  leverage: number;
}
