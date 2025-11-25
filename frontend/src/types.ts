export type Role = "user" | "assistant" | "system";

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  createdAt: string;
  pending?: boolean;
  sources?: Array<{
    title: string;
    snippet: string;
  }>;
}

export interface AgentTool {
  id: string;
  label: string;
  enabled: boolean;
  description: string;
}

export interface ChatSession {
  id: string;
  title: string;
  lastMessageAt: string;
}

