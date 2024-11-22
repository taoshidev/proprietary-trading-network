import { pro as ccxt } from "ccxt";

export async function initializeExchange(exchangeId, config) {
  const { apiKey, secret, password, demo } = config;

  if (!apiKey || !secret) return null;

  const exchange = new ccxt[exchangeId]({
    apiKey,
    secret,
    ...(password && { password }),
    options: { defaultType: "spot" },
  });

  exchange.set_sandbox_mode(demo);

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
