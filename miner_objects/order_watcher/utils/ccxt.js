import { pro as ccxt } from "ccxt";

export async function initializeExchange(config) {
  const { apiKey, secret, demo } = config;

  if (!apiKey || !secret ) return null;

  const exchange = new ccxt.bybit({
    apiKey: config.apiKey,
    secret: config.secret,
  });

  await exchange.enableDemoTrading(demo);

  exchange.enableRateLimit = true;

  return exchange;
}