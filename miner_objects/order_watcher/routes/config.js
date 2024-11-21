import { JSONFilePreset } from "lowdb/node";

import config from "../config.json" assert { type: "json" };

export const configRoutes = (router, wss) => {
  router.post("/save-config", async (req, res) => {
    const db = await JSONFilePreset("config.json", config);
    const { exchange, apiKey, secret, demo } = req.body;

    if (!apiKey || !secret || !demo) {
      return res.status(400).json({ error: "Missing required data" });
    }

    try {
      db.data = {
        ...db.data,
        [exchange]: { apiKey, secret, demo },
      };

      await db.write();

      res.status(200).json({
        status: 200,
        message: `Configuration for ${exchange} saved successfully`,
        data: db.data[exchange],
      });
    } catch (error) {
      console.error("Error saving config file:", error);
      return res.status(500).json({ error: "Failed to save config file" });
    }
  });

  router.get("/get-config", async (req, res) => {
    const db = await JSONFilePreset("config.json", config);
    const { exchange } = req.query;

    try {
      if (exchange) {
        const exchangeConfig = db.data[exchange];

        if (!exchangeConfig) {
          return res
            .status(404)
            .json({ error: `Configuration for ${exchange} not found` });
        }

        return res.status(200).json({
          message: `Configuration for ${exchange} retrieved successfully`,
          data: exchangeConfig,
        });
      }

      return res.status(200).json({
        message: "All configurations retrieved successfully",
        data: db.data,
      });
    } catch (error) {
      console.error("Error retrieving config file:", error);
      return res.status(500).json({ error: "Failed to retrieve config file" });
    }
  });

  return router;
};
