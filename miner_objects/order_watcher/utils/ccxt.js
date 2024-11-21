import { pro as ccxt } from "ccxt";

export async function initializeExchange(exchangeId, config) {
  const { apiKey, secret, demo } = config;

  if (!apiKey || !secret) return null;

  const exchange = new ccxt[exchangeId]({
    apiKey: config.apiKey,
    secret: config.secret,
  });

  if (exchange && typeof exchange.enableDemoTrading === "function") {
    await exchange.enableDemoTrading(demo);
  } else {
    console.error(
      "enableDemoTrading is not a function or exchange is undefined",
    );
  }

  exchange.enableRateLimit = true;

  return exchange;
}
