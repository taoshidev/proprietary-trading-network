import { WebSocketServer } from "ws";

let wss = null;

export function startWebSocket(server) {
  wss = new WebSocketServer({ server });
  console.log("WebSocket server initialized");
  return wss;
}
