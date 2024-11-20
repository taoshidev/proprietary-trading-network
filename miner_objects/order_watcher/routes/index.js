import express from "express";
import { minerRoutes } from "./miner.js";
import { configRoutes } from "./config.js";

export const initializeRoutes = (wss) => {
  const router = express.Router();

  minerRoutes(router, wss);
  configRoutes(router, wss);

  return router;
};