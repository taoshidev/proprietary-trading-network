import { WebSocket, WebSocketServer } from "ws";

let wss = null; // Keep a reference to the WebSocket server

/**
 * Initializes the WebSocket server
 * @param {Object} server - The HTTP server instance
 * @returns {WebSocketServer} - The WebSocket server instance
 */
export function initializeWebSocketServer(server) {
	wss = new WebSocketServer({ server });
	console.log("WebSocket server initialized");
	return wss;
}

/**
 * Broadcasts a message to all connected WebSocket clients
 * @param {Object} wss - The WebSocket server instance
 * @param {Object} message - The message to broadcast
 */
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
