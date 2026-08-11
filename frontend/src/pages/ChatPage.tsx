import { Plus } from "lucide-react";

import Welcome from "../components/chat/Welcome";
import ChatWindow from "../components/chat/ChatWindow";
import ChatInput from "../components/input/ChatInput";

import { useChat } from "../context/ChatContext";

export default function ChatPage() {
  const {
    messages,
    newChat,
  } = useChat();

  const hasMessages = messages.length > 0;

  return (
    <div className="flex h-full w-full flex-col bg-[#F8FAFC]">

      {/* Header */}
      <header className="flex h-16 shrink-0 items-center justify-between border-b border-gray-200 bg-white px-6">

        <div>
          <h1 className="text-lg font-semibold text-gray-900">
            WAC AI
          </h1>

          <p className="text-xs text-gray-400">
            Web and Craft AI Assistant
          </p>
        </div>

        <button
          onClick={newChat}
          className="
            flex
            items-center
            gap-2
            rounded-xl
            border
            border-gray-200
            bg-white
            px-4
            py-2
            text-sm
            font-medium
            text-gray-700
            transition
            hover:bg-gray-50
          "
        >
          <Plus size={17} />
          New Chat
        </button>

      </header>

      {/* Chat */}
      <div className="flex min-h-0 flex-1 justify-center">

        <div className="relative flex h-full w-full max-w-6xl flex-col">

          {!hasMessages ? (

            <div className="flex flex-1 items-center justify-center px-6">

              <div className="flex w-full flex-col items-center">

                <Welcome />

                <div className="mt-10 w-full max-w-2xl">

                  <ChatInput
                    variant="center"
                  />

                </div>

              </div>

            </div>

          ) : (

            <>
              <div className="min-h-0 flex-1 overflow-y-auto">

                <ChatWindow
                  messages={messages}
                />

              </div>

              <ChatInput
                variant="bottom"
              />
            </>

          )}

        </div>

      </div>

    </div>
  );
}