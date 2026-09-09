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
  isStreaming?: boolean;
  provider?: string;
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

  updateStreamingAssistantMessage: (
    deltaText: string,
    isDone: boolean
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

  const addVoiceUserMessage = (message: string) => {
    const text = message.trim();
    if (!text) {
      return;
    }

    setMessages((previous) => {
      if (previous.length > 0) {
        const lastIndex = previous.length - 1;
        const last = previous[lastIndex];
        if (last.role === "user") {
          const updated = [...previous];
          updated[lastIndex] = {
            ...last,
            content: text,
          };
          return updated;
        }
      }

      return [
        ...previous,
        {
          role: "user",
          content: text,
        },
      ];
    });
  };


  // ==========================================================
  // VOICE ASSISTANT STREAMING & FINAL MESSAGE
  // ==========================================================

  const updateStreamingAssistantMessage = (
    deltaText: string,
    isDone: boolean
  ) => {
    setMessages((previous) => {
      if (previous.length === 0) {
        if (isDone || !deltaText) return previous;
        return [
          {
            role: "assistant",
            content: deltaText,
            provider: "gemini-live",
            isStreaming: !isDone,
          },
        ];
      }

      const lastIndex = previous.length - 1;
      const lastMessage = previous[lastIndex];

      if (lastMessage.role === "assistant") {
        const updated = [...previous];
        updated[lastIndex] = {
          ...lastMessage,
          content: isDone && !deltaText ? lastMessage.content : lastMessage.content + deltaText,
          isStreaming: !isDone,
        };
        return updated;
      }

      if (isDone || !deltaText) {
        return previous;
      }

      return [
        ...previous,
        {
          role: "assistant",
          content: deltaText,
          provider: "gemini-live",
          isStreaming: !isDone,
        },
      ];
    });
  };

  const addVoiceAssistantMessage = (message: string) => {
    const text = message.trim();
    if (!text) {
      return;
    }

    setMessages((previous) => {
      if (previous.length > 0) {
        const lastIndex = previous.length - 1;
        const last = previous[lastIndex];
        if (last.role === "assistant") {
          const updated = [...previous];
          updated[lastIndex] = {
            ...last,
            content: text,
            isStreaming: false,
          };
          return updated;
        }
      }

      return [
        ...previous,
        {
          role: "assistant",
          content: text,
          provider: "gemini-live",
          isStreaming: false,
        },
      ];
    });
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

        updateStreamingAssistantMessage,

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