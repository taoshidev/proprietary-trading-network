import { TradePair, OrderType } from "../constants/index.js";

export async function sendToPTN(signal) {
	const data = {
		"trade_pair": TradePair[signal.trade_pair],
		"order_type": OrderType[signal.order_type],
		"leverage": signal.leverage,
		"api_key": 'xxxx'
	};

	try {
		const response = await fetch(`http://127.0.0.1:3005/api/receive-signal`, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify(data),
		});

		const result = await response.json();

		return {
			data,
			status: response.status,
			message: result.message || "Send to PTN",
			error: !response.ok ? (result.error || "Failed to send to PTN") : undefined
		};
	} catch (error) {
		return {
			status: 500,
			message: "Failed to send to PTN",
			error: error instanceof Error ? error.message : "An unexpected error occurred"
		};
	}
}