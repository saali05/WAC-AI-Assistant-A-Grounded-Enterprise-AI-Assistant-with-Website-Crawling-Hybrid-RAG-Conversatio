import api from "./axios";

export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ConversationDetail {
  id: string;
  title: string;
  messages: ConversationMessage[];
}

export async function getConversations() {
  const response =
    await api.get<ConversationDetail[]>(
      "/conversations"
    );

  return response.data;
}

export async function getConversation(
  id: string
): Promise<ConversationDetail> {
  const response =
    await api.get(`/conversations/${id}`);

  return response.data;
}

export async function renameConversation(
  id: string,
  title: string
) {
  return api.patch(
    `/conversations/${id}`,
    {
      title,
    }
  );
}

export async function deleteConversation(
  id: string
) {
  return api.delete(
    `/conversations/${id}`
  );
}