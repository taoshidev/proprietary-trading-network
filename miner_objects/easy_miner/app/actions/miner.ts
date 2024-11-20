"use server";

import { API_URL } from '@/constants'

export async function startMiner(data) {
  console.log("Received Data:", data);

  try {
    const response = await fetch(`${API_URL}/start-miner`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    const result = await response.json();

    return {
      status: response.status,
      message: result.message || "Miner started successfully",
      data: result.data,
      error: !response.ok ? (result.error || "Failed to start miner") : undefined
    };
  } catch (error: unknown) {
    console.error("Error in startMiner:", error);

    return {
      status: 500,
      message: "Failed to start miner",
      error: error instanceof Error ? error.message : "An unexpected error occurred"
    };
  }
}

export async function startServer() {
  console.log("Start Server");

  try {
    const response = await fetch(`${API_URL}/start-server`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const result = await response.json();

    return {
      status: response.status,
      message: result.message || "Server started successfully",
      data: result.data,
      error: !response.ok ? (result.error || "Failed to start server") : undefined
    };
  } catch (error: unknown) {
    console.error("Error in startServer:", error);

    return {
      status: 500,
      message: "Failed to start server",
      error: error instanceof Error ? error.message : "An unexpected error occurred"
    };
  }
}