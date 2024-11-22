import { spawn } from "child_process";
import path from "path";

import { broadcast } from "../websocket/broadcast.js";

export const startSignalServer = async (wss) => {
  console.log("Starting server...");
  const scriptPath = path.resolve("../../mining/run_receive_signals_server.py");

  return new Promise((resolve, reject) => {
    const pythonProcess = spawn("python3", [scriptPath]);

    let stdout = "";
    let stderr = "";

    pythonProcess.stdout.on("data", (data) => {
      stdout += data.toString();
      console.log("Python STDOUT:", data.toString());
      broadcast(wss, { type: "log", message: data.toString() });
    });

    pythonProcess.stderr.on("data", (data) => {
      stderr += data.toString();
      console.error("Python STDERR:", data.toString());
      broadcast(wss, { type: "error", message: data.toString() });
    });

    pythonProcess.on("close", (code) => {
      if (code === 0) {
        broadcast(wss, {
          type: "status",
          message: "Miner started successfully",
        });
        resolve(stdout);
      } else {
        broadcast(wss, { type: "status", message: "Miner failed to start" });
        reject(new Error(stderr || `Python process exited with code ${code}`));
      }
    });

    pythonProcess.on("error", (error) => {
      reject(error);
    });
  });
};
