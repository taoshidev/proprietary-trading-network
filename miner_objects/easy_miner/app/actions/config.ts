"use server";

import { API_URL } from "@/constants";
import { type Config } from "@/types";

export async function saveConfig(data: Config) {
  try {
    const response = await fetch(`${API_URL}/save-config`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    const result = await response.json();

    return {
      status: response.status,
      message: result.message || "Configuration saved",
      data: result.data,
      error: !response.ok
        ? result.error || "Failed to save configuration"
        : undefined,
    };
  } catch (error) {
    return {
      status: 500,
      message: "Failed to save configuration",
      error:
        error instanceof Error ? error.message : "An unexpected error occurred",
    };
  }
}

export async function getConfig(exchange?: string) {
  try {
    const url = exchange
      ? `${API_URL}/get-config?exchange=${exchange}`
      : `${API_URL}/get-config`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const result = await response.json();

    return {
      status: response.status,
      message: result.message || "Configuration retrieved",
      data: result.data,
      error: !response.ok
        ? result.error || "Failed to get configuration"
        : undefined,
    };
  } catch (error) {
    return {
      status: 500,
      message: "Failed to get configuration",
      error:
        error instanceof Error ? error.message : "An unexpected error occurred",
    };
  }
}
