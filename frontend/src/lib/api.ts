import { API_BASE_URL } from "../config";
import type { AgentTool, ChatMessage, ChatSession } from "../types";

const defaultHeaders = {
  "Content-Type": "application/json"
};

const handleResponse = async (resp: Response) => {
  if (!resp.ok) {
    const message = await resp.text();
    throw new Error(message || "Unexpected API error");
  }
  return resp.json();
};

export async function sendMessage(
  sessionId: string,
  payload: { message: string; tools: string[]; temperature: number }
): Promise<{ messages: ChatMessage[] }> {
  const resp = await fetch(`${API_BASE_URL}/api/chat/${sessionId}`, {
    method: "POST",
    headers: defaultHeaders,
    body: JSON.stringify(payload)
  });
  return handleResponse(resp);
}

export async function createSession(): Promise<ChatSession> {
  const resp = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: defaultHeaders,
    body: JSON.stringify({})
  });
  return handleResponse(resp);
}

export async function fetchSessions(): Promise<ChatSession[]> {
  const resp = await fetch(`${API_BASE_URL}/api/chat`);
  return handleResponse(resp);
}

export async function fetchMessages(sessionId: string): Promise<ChatMessage[]> {
  const resp = await fetch(`${API_BASE_URL}/api/chat/${sessionId}`);
  return handleResponse(resp);
}

export async function uploadDocument(file: File): Promise<{ id: string }> {
  const formData = new FormData();
  formData.append("file", file);
  const resp = await fetch(`${API_BASE_URL}/api/rag/upload`, {
    method: "POST",
    body: formData
  });
  return handleResponse(resp);
}

export async function fetchTools(): Promise<AgentTool[]> {
  const resp = await fetch(`${API_BASE_URL}/api/tools`);
  return handleResponse(resp);
}

export async function toggleTool(toolId: string, enabled: boolean) {
  const resp = await fetch(`${API_BASE_URL}/api/tools/${toolId}`, {
    method: "PATCH",
    headers: defaultHeaders,
    body: JSON.stringify({ enabled })
  });
  return handleResponse(resp);
}

