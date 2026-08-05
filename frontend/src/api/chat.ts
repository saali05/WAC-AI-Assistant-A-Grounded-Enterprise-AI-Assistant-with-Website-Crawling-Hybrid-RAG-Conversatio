import api from "./axios";
import type { Provider } from "../types/chat";

export interface ChatRequest {
  conversation_id?: string;
  provider: Provider;
  message: string;
}

export interface ChatResponse {
  conversation_id: string;
  response: string;
}

export async function sendMessage(
  data: ChatRequest
): Promise<ChatResponse> {
  const response = await api.post<ChatResponse>(
    "/chat",
    data
  );

  return response.data;
}