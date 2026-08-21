import { useState } from "react";
import { Plus, Cpu, Activity } from "lucide-react";

import Welcome from "../components/chat/Welcome";
import ChatWindow from "../components/chat/ChatWindow";
import ChatInput from "../components/input/ChatInput";
import SessionAnalytics from "../components/analytics/SessionAnalytics";

import { useChat } from "../context/ChatContext";

export default function ChatPage() {
  const { messages, conversationId, newChat } = useChat();
  const [initialPrompt, setInitialPrompt] = useState("");
  const [showAnalytics, setShowAnalytics] = useState(false);

  const hasMessages = messages.length > 0;

  const handleSelectPrompt = (promptText: string) => {
    setInitialPrompt(promptText);
  };

  return (
    <div className="flex h-full w-full flex-col bg-transparent relative">
      {/* Header Navigation Bar */}
      <header className="flex h-16 shrink-0 items-center justify-between border-b border-white/10 bg-[#0B0F19]/80 backdrop-blur-xl px-6 z-20">
        {/* Brand Logo & Title */}
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-red-600 via-rose-500 to-violet-600 p-[1px] shadow-lg shadow-red-500/20">
            <div className="flex h-full w-full items-center justify-center rounded-[11px] bg-[#0B0F19]">
              <span className="font-extrabold text-xs tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-red-500 to-rose-300">
                WAC
              </span>
            </div>
          </div>

          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-sm font-bold tracking-tight text-white">
                WEBCRAFTS <span className="text-red-500 font-extrabold">AI</span>
              </h1>
              <span className="rounded-full bg-red-500/10 border border-red-500/20 px-2 py-0.5 text-[10px] font-semibold text-red-400">
                ENTERPRISE
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              WebAndCrafts Digital Intelligence Platform
            </p>
          </div>
        </div>

        {/* Right Action Controls & Status */}
        <div className="flex items-center gap-3">
          {/* Analytics Dashboard Toggle Button */}
          <button
            onClick={() => setShowAnalytics((prev) => !prev)}
            className="
              flex
              items-center
              gap-2
              rounded-xl
              border
              border-red-500/30
              bg-red-500/10
              px-3.5
              py-1.5
              text-xs
              font-semibold
              text-red-300
              shadow-md
              transition-all
              duration-200
              hover:border-red-500/50
              hover:bg-red-500/20
              hover:text-white
              active:scale-95
            "
          >
            <Activity size={14} className="text-red-400" />
            <span>Session Analytics</span>
          </button>

          {/* Active Model Status Indicator */}
          <div className="hidden sm:flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-slate-300 backdrop-blur-md">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <Cpu size={13} className="text-slate-400" />
            <span className="text-[11px]">Gemini & Groq Active</span>
          </div>

          {/* New Chat Button */}
          <button
            onClick={() => {
              setInitialPrompt("");
              newChat();
            }}
            className="
              flex
              items-center
              gap-2
              rounded-xl
              border
              border-white/10
              bg-[#141C2E]
              px-3.5
              py-1.5
              text-xs
              font-semibold
              text-slate-200
              shadow-md
              transition-all
              duration-200
              hover:border-red-500/40
              hover:bg-[#1A253D]
              hover:text-white
              hover:shadow-red-500/10
              active:scale-95
            "
          >
            <Plus size={15} className="text-red-400" />
            <span>New Chat</span>
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex min-h-0 flex-1 justify-center">
        <div className="relative flex h-full w-full max-w-6xl flex-col">
          {!hasMessages ? (
            <div className="flex flex-1 items-center justify-center px-4 sm:px-6 py-6 overflow-y-auto">
              <div className="flex w-full flex-col items-center max-w-4xl">
                <Welcome onSelectPrompt={handleSelectPrompt} />

                <div className="mt-8 w-full max-w-2xl">
                  <ChatInput
                    variant="center"
                    initialMessage={initialPrompt}
                  />
                </div>
              </div>
            </div>
          ) : (
            <>
              <div className="min-h-0 flex-1 overflow-y-auto">
                <ChatWindow messages={messages} />
              </div>

              <ChatInput variant="bottom" />
            </>
          )}
        </div>
      </div>

      {/* Slide-out Session Analytics Drawer */}
      <SessionAnalytics
        conversationId={conversationId}
        isOpen={showAnalytics}
        onClose={() => setShowAnalytics(false)}
      />
    </div>
  );
}