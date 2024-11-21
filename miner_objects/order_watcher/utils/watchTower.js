import _ from "lodash";

import { initializeExchange } from "./ccxt.js";
import { broadcast } from "./websocketService.js";
import { orderToSignal } from "./orderToSignal.js";
import { calculateLeverage } from "./calculateLeverage.js";
import { sendToPTN } from "./sendToPTN.js";
import { logUserBalance } from "./logs.js";

import config from "../config.json" assert { type: "json" };

export const watchTower = async (id, wss) => {
  let exchange = await initializeExchange(id, config[id]);

  if (!exchange) return null;

  await exchange.loadMarkets();

  console.info(`Watching orders for ${id}...`);

  while (true) {
    try {
      const order = await exchange.watchOrders();
      const recentOrder = _.last(order);

      if (recentOrder.info.orderStatus === "Filled") {
        // on new order get balance
        const balance = await exchange.fetchBalance();

        const leverage = calculateLeverage(recentOrder, balance);
        const signal = orderToSignal(recentOrder, leverage);

        const response = await sendToPTN(signal);

        console.info(`${signal.trade_pair}:${signal.order_type} sent to PTN`);

        logUserBalance(balance);
        broadcast(wss, response.data);
      }
    } catch (error) {
      console.error("Error watching orders:", error);
    }
  }
};
