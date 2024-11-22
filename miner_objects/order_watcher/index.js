import express from "express";
import { createServer } from "http";
import _ from "lodash";
import { Command } from "commander";

import { watchTower } from "./modules/app/index.js";
import { startWebSocket } from "./modules/websocket/start.js";
import { initializeRoutes } from "./routes/index.js";
import config from "./config.json" assert { type: "json" };

const app = express();
const server = createServer(app);
const wss = startWebSocket(server);
const program = new Command();

app.use(express.json());
app.use("/api", initializeRoutes(wss));

program
  .option("--exchange <string>", "specify the exchange to use", "")
  .option("--port <number>", "specify the port to use", config.port)
  .parse(process.argv);

const options = program.opts();

(async () => {
  try {
    console.log(`Starting server with options: ${JSON.stringify(options)}`);

    if (options.exchange) {
      console.log(`Initializing watchTower for exchange: ${options.exchange}`);

      await watchTower(options.exchange, wss);

      server.listen(options.port, () => {
        console.log(
          `WebSocket server running on ws://localhost:${options.port}`,
        );
      });
    } else {
      console.warn(
        "No exchange specified. Skipping watchTower initialization.",
      );
    }
  } catch (error) {
    console.warn("Error starting Order Watcher");
  }
})();
