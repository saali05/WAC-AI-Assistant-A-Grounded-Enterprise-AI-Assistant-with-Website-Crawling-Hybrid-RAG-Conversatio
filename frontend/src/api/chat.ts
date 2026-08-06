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

  try {

    const response =
      await api.post("/chat", data);

    return response.data;

  } catch (error: any) {

    if (error.response?.data?.error) {

      throw new Error(
        error.response.data.error.message
      );

    }

    throw new Error(
      "Unable to connect to the AI service."
    );
  }
}