import {startMinerProcess, startSignalServer} from "../lib/miner.js";

export const minerRoutes = (router, wss) => {
	router.post("/start-miner", async (req, res) => {
		const { testnet, wallet, debug } = req.body;

		try {
			const output = await startMinerProcess(wss, { testnet, wallet, debug });
			res.json({
				status: "success",
				message: "Miner started successfully",
				output,
			});
		} catch (error) {
			console.error("Error in start-miner route:", error.message);
			res.status(500).json({
				status: "error",
				message: error.message || "Failed to start miner",
			});
		}
	});

	router.post("/start-server", async (_, res) => {
		try {
			const output = await startSignalServer(wss);
			console.log("Starting server:", output);
			res.json({
				status: "success",
				message: "Miner started successfully",
				output,
			});
		} catch (error) {
			console.error("Error in start-miner route:", error.message);
			res.status(500).json({
				status: "error",
				message: error.message || "Failed to start miner",
			});
		}
	});

	return router;
};
