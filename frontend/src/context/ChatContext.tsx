import {
  createContext,
  useContext,
  useState,
  type ReactNode,
} from "react";

import type { Provider } from "../types/chat";

import { sendMessage } from "../api/chat";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface ChatContextType {
  messages: Message[];

  loading: boolean;

  conversationId: string | null;

  send: (
    message: string,
    provider: Provider
  ) => Promise<void>;

  newChat: () => void;
}

const ChatContext =
  createContext<ChatContextType | null>(null);

export function ChatProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [messages, setMessages] =
    useState<Message[]>([]);

  const [loading, setLoading] =
    useState(false);

  const [conversationId, setConversationId] =
    useState<string | null>(null);

  /**
   * Start a completely new chat session.
   *
   * The previous session is intentionally
   * not loaded again.
   */
  const newChat = () => {
    setConversationId(null);
    setMessages([]);
  };

  /**
   * Send a message inside the current session.
   */
  const send = async (
    text: string,
    provider: Provider
  ) => {
    if (!text.trim() || loading) {
      return;
    }

    const userMessage: Message = {
      role: "user",
      content: text.trim(),
    };

    // Immediately show user message.
    setMessages((prev) => [
      ...prev,
      userMessage,
    ]);

    setLoading(true);

    try {
      const response = await sendMessage({
        conversation_id:
          conversationId ?? undefined,

        provider,

        message: text.trim(),
      });

      /*
       * If this is the first message of the session,
       * backend creates the conversation.
       */
      if (!conversationId) {
        setConversationId(
          response.conversation_id
        );
      }

      // Add assistant response to THIS session only.
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: response.response,
        },
      ]);
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Something went wrong while generating the response.";

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: errorMessage,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ChatContext.Provider
      value={{
        messages,
        loading,
        conversationId,
        send,
        newChat,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChat() {
  const context = useContext(ChatContext);

  if (!context) {
    throw new Error(
      "useChat must be used inside ChatProvider"
    );
  }

  return context;
}