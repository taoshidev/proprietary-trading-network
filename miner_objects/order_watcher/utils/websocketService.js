import { WebSocket, WebSocketServer } from "ws";

let wss = null;

export function initializeWebSocketServer(server) {
  wss = new WebSocketServer({ server });
  console.log("WebSocket server initialized");
  return wss;
}

export function broadcast(wss, message) {
  if (!wss) {
    console.error("WebSocket server is not initialized.");
    return;
  }
  const data = JSON.stringify(message);
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(data);
    }
  });
}
