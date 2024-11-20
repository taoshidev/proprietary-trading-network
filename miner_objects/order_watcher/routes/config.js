import { JSONFilePreset } from "lowdb/node";

import config from "../config.json" assert { type: "json" };

export const configRoutes = (router, wss) => {

	router.post("/save-config", async (req, res) => {
		console.log(req.body);
		const db = await JSONFilePreset("config.json", config);
		const { apiKey, secret, demo } = req.body;

		if (!apiKey || !secret || !demo) {
			return res.status(400).json({ error: "Missing required data" });
		}

		const newConfig = { apiKey, secret, demo };

		try {
			db.data = {...db.data, ...newConfig };

			await db.write();

			res.status(200).json({
				status: 200,
				message: "Configuration saved successfully",
				data: db.data,
			});
		} catch (error) {
			console.error("Error saving config file:", error);
			return res.status(500).json({ error: "Failed to save config file" });
		}
	});

	router.get("/get-config", async (_, res) => {
		const db = await JSONFilePreset("config.json", config);
		const { apiKey, secret, demo } = db.data;

		if (!apiKey || !secret || !demo) {
			return res.status(400).json({ error: "Missing required data" });
		}

		return res.status(200).json({
			message: "Configuration retrieved successfully",
			data: db.data,
		});
	});

	return router;
};
