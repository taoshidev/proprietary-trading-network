import axios, { AxiosError } from "axios";

const MINER_URL = import.meta.env.VITE_MINER_URL;

const api = axios.create({
  baseURL: MINER_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export async function getMinerData() {
  try {
    const response = await api.get(`/miner-data`);
    
    return response.data;
  } catch (error) {
    if (error instanceof AxiosError) {
      throw new Error(error.message);
    } else {
      throw new Error(`Unknown error: ${(error as Error).message}`);
    }
  }
}