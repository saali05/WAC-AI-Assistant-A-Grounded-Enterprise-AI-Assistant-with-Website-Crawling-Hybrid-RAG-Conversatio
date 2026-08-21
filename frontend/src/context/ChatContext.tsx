import {
  createContext,
  useContext,
  useState,
  type ReactNode,
} from "react";

import type { Provider, SourceItem } from "../types/chat";

import { sendMessage } from "../api/chat";

import {
  getConversation,
  getConversations,
} from "../api/conversation";


// ============================================================
// TYPES
// ============================================================

export interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: SourceItem[];
}

export interface Conversation {
  id: string;
  title: string;
}

interface ChatContextType {
  messages: Message[];

  conversations: Conversation[];

  loading: boolean;

  conversationId: string | null;

  send: (
    message: string,
    provider: Provider
  ) => Promise<void>;

  loadConversation: (
    id: string
  ) => Promise<void>;

  reloadConversations: () => Promise<void>;

  newChat: () => void;

  addVoiceUserMessage: (
    message: string
  ) => void;

  addVoiceAssistantMessage: (
    message: string
  ) => void;

  setVoiceConversationId: (
    id: string
  ) => void;
}


// ============================================================
// CONTEXT
// ============================================================

const ChatContext =
  createContext<ChatContextType | null>(
    null
  );


// ============================================================
// PROVIDER
// ============================================================

export function ChatProvider({
  children,
}: {
  children: ReactNode;
}) {
  // ----------------------------------------------------------
  // STATE
  // ----------------------------------------------------------

  const [
    messages,
    setMessages,
  ] = useState<Message[]>([]);

  const [
    conversations,
    setConversations,
  ] = useState<Conversation[]>([]);

  const [
    loading,
    setLoading,
  ] = useState(false);

  const [
    conversationId,
    setConversationId,
  ] = useState<string | null>(
    null
  );


  // ==========================================================
  // RELOAD CONVERSATIONS
  // ==========================================================

  const reloadConversations =
    async () => {
      try {
        const data =
          await getConversations();

        setConversations(data);
      } catch (error) {
        console.error(
          "Failed to load conversations:",
          error
        );
      }
    };


  // ==========================================================
  // NEW CHAT
  // ==========================================================

  const newChat = () => {
    setConversationId(null);

    setMessages([]);
  };


  // ==========================================================
  // LOAD EXISTING CONVERSATION
  // ==========================================================

  const loadConversation =
    async (
      id: string
    ) => {
      try {
        const data =
          await getConversation(id);

        setConversationId(id);

        setMessages(
          Array.isArray(data.messages)
            ? data.messages
            : []
        );
      } catch (error) {
        console.error(
          "Failed to load conversation:",
          error
        );
      }
    };


  // ==========================================================
  // NORMAL TEXT CHAT
  // ==========================================================

  const send =
    async (
      text: string,
      provider: Provider
    ) => {
      const message =
        text.trim();

      if (!message) {
        return;
      }


      // ------------------------------------------------------
      // Immediately display user message
      // ------------------------------------------------------

      setMessages(
        previous => [
          ...previous,

          {
            role: "user",
            content: message,
          },
        ]
      );


      setLoading(true);


      try {
        const response =
          await sendMessage({
            conversation_id:
              conversationId ??
              undefined,

            provider,

            message,
          });


        // ----------------------------------------------------
        // New conversation created
        // ----------------------------------------------------

        if (!conversationId) {
          setConversationId(
            response.conversation_id
          );

          await reloadConversations();
        }


        // ----------------------------------------------------
        // Add assistant response
        // ----------------------------------------------------

        setMessages(
          previous => [
            ...previous,

            {
              role: "assistant",
              content:
                response.response,
              sources:
                response.sources,
            },
          ]
        );

      } catch (error) {
        console.error(
          "Chat request failed:",
          error
        );


        setMessages(
          previous => [
            ...previous,

            {
              role: "assistant",

              content:
                "I'm sorry, I couldn't process your request. Please try again.",
            },
          ]
        );

      } finally {
        setLoading(false);
      }
    };


  // ==========================================================
  // VOICE USER MESSAGE
  // ==========================================================

  const addVoiceUserMessage =
    (
      message: string
    ) => {
      const text =
        message.trim();

      if (!text) {
        return;
      }


      setMessages(
        previous => [
          ...previous,

          {
            role: "user",
            content: text,
          },
        ]
      );
    };


  // ==========================================================
  // VOICE ASSISTANT MESSAGE
  // ==========================================================

  const addVoiceAssistantMessage =
    (
      message: string
    ) => {
      const text =
        message.trim();

      if (!text) {
        return;
      }


      setMessages(
        previous => [
          ...previous,

          {
            role: "assistant",
            content: text,
          },
        ]
      );
    };


  // ==========================================================
  // VOICE CONVERSATION ID
  // ==========================================================

  const setVoiceConversationId =
    (
      id: string
    ) => {
      setConversationId(id);
    };


  // ==========================================================
  // PROVIDER
  // ==========================================================

  return (
    <ChatContext.Provider
      value={{
        messages,

        conversations,

        loading,

        conversationId,

        send,

        loadConversation,

        reloadConversations,

        newChat,

        addVoiceUserMessage,

        addVoiceAssistantMessage,

        setVoiceConversationId,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
}


// ============================================================
// HOOK
// ============================================================

export function useChat() {
  const context =
    useContext(
      ChatContext
    );

  if (!context) {
    throw new Error(
      "useChat must be used inside ChatProvider"
    );
  }

  return context;
}