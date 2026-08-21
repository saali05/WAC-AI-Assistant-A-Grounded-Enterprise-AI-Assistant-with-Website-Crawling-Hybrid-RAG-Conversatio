export type Provider = "gemini" | "groq";

export interface SourceItem {
  title: string;
  url: string;
  heading?: string;
  score?: number;
}

export interface Message {
  id?: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceItem[];
  rag_used?: boolean;
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