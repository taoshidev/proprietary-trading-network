import express from "express";
import { createServer } from "http";
import _ from "lodash";

import { initializeExchange } from './lib/ccxt.js';
import { orderToSignal } from "./lib/orderToSignal.js";
import {initializeRoutes} from "./routes/index.js";
import { initializeWebSocketServer, broadcast } from "./utils/websocketService.js";
import {calculateLeverage} from "./utils/calculateLeverage.js";
import { sendToPTN } from './lib/sendToPTN.js';

import config from "./config.json" assert { type: 'json' };

const app = express();
const server = createServer(app);
const wss = initializeWebSocketServer(server);

app.use(express.json());
app.use("/api", initializeRoutes(wss));

(async () => {
	let exchange = await initializeExchange(config);

	if (!exchange) return null;

	await exchange.loadMarkets();
	const balance = await exchange.fetchBalance();

	console.log(balance);
	console.log("Starting WebSocket server...");

	while (true) {
		try {
			const order = await exchange.watchOrders();
			const recentOrder = _.last(order);

			console.log("Recent Order Status: ", recentOrder.info.orderStatus);

			// if recent order is filled format, send to ptn, and broadcast
			if (recentOrder.info.orderStatus === 'Filled') {
				const leverage = calculateLeverage(recentOrder, balance);
				const signal = orderToSignal(recentOrder, leverage);

				const response = await sendToPTN(signal);

				// broadcast to frontend
				broadcast(wss, response.data);
			}
		} catch (error) {
			console.error("Error watching orders:", error);
		}
	}
})();

server.listen(8080, () => {
	console.log("WebSocket server running on ws://localhost:8080");
});