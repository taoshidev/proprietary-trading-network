import { create } from "zustand";
import { WEBSOCKET_URL } from "@/constants";
import { Signal } from '@/types'

interface SignalsState {
  signals: Signal[];
  isWatching: boolean;
  error: string | null;
  startWebSocket: () => void;
  stopWebSocket: () => void;
  addSignal: (newSignal: Signal[]) => void;
  clearError: () => void;
}

const useOrdersStore = create<SignalsState>((set, get) => {
  let ws: WebSocket | null = null;

  return {
    signals: [],
    isWatching: false,
    error: null,

    startWebSocket: () => {
      if (get().isWatching) return;

      ws = new WebSocket(WEBSOCKET_URL);

      ws.onopen = () => {
        console.log("Socket WebSocket connection opened");
        set({ isWatching: true, error: null });
      };

      ws.onmessage = (event) => {
        console.log("WebSocket connection message", event.data);

        try {
          const message: Signal = JSON.parse(event.data);

          console.log(message);
          
          set((state) => {
            const newSignal: Signal[] = [message];
            return {
              signals: [...state.signals, ...newSignal],
              error: null,
            };
          });
        } catch (error) {
          console.error("Error parsing WebSocket message:", error);
          set({ error: "Failed to parse server message" });
        }
      };

      ws.onclose = () => {
        console.log("WebSocket connection closed");
        set({ isWatching: false });
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        set({
          error: 'WebSocket connection error',
          isWatching: false
        });
      };
    },

    stopWebSocket: () => {
      if (ws) {
        ws.close();
        ws = null;
        set({ isWatching: false });
      }
    },

    addSignal: (newSignal: Signal[]) => {
      set((state) => ({ signals: [...state.signals, ...newSignal] }));
    },

    clearError: () => {
      set({ error: null });
    },
  };
});

export default useOrdersStore;