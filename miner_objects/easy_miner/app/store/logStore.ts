import { create } from "zustand";
import { WEBSOCKET_URL } from "@/constants";
import { Config } from '@/types'

interface WebSocketMessage {
  type: "orders" | "status" | "error";
  message?: string;
  data?: Config;
}

interface LogStore {
  logs: string[];
  isWatching: boolean;
  error: string | null;
  initialize: () => void;
  stopWebSocket: () => void;
  clearLogs: () => void;
  clearError: () => void;
}

const useLogStore = create<LogStore>((set, get) => {
  let ws: WebSocket | null = null;

  return {
    logs: [],
    isWatching: false,
    error: null,

    initialize: () => {
      if (get().isWatching) return;

      ws = new WebSocket(WEBSOCKET_URL);

      ws.onopen = () => {
        console.log("WebSocket connection opened");
        set({ isWatching: true, error: null });
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          const log = message.message || JSON.stringify(message.data) || "Unknown message";

          set((state) => ({
            logs: [...state.logs, log],
            error: null,
          }));
        } catch (error) {
          console.error("Error parsing WebSocket message:", error);
          set((state) => ({
            logs: [...state.logs, "Error: Failed to parse server message"],
            error: "Failed to parse server message",
          }));
        }
      };

      ws.onclose = () => {
        console.log("WebSocket connection closed");
        set({ isWatching: false });
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        set((state) => ({
          logs: [...state.logs, "WebSocket encountered an error"],
          error: "WebSocket encountered an error",
        }));
      };
    },

    stopWebSocket: () => {
      if (ws) {
        ws.close();
        ws = null;
        set({ isWatching: false });
      }
    },

    clearLogs: () => {
      set({ logs: [] });
    },

    clearError: () => {
      set({ error: null });
    },
  };
});

export default useLogStore;