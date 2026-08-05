export type Provider = "gemini" | "groq";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

export interface Conversation {
  id: string;
  title: string;
}

export interface ChatInputProps {
  loading: boolean;
  onSend: (
    message: string,
    provider: Provider
  ) => void;
}