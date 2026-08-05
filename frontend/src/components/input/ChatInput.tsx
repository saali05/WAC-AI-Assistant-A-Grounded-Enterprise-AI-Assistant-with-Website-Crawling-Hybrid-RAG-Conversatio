import { useState } from "react";

import { useChat } from "../../context/ChatContext";

import ModelSelector from "./ModelSelector";
import AttachButton from "./AttachButton";
import MessageInput from "./MessageInput";
import VoiceButton from "./VoiceButton";
import SendButton from "./SendButton";

import type { Provider } from "../../types/chat";

interface ChatInputProps {
  variant: "center" | "bottom";
}

export default function ChatInput({
  variant,
}: ChatInputProps) {
  const {
    send,
    loading,
  } = useChat();

  const [provider, setProvider] =
    useState<Provider>("gemini");

  const [message, setMessage] =
    useState("");

  const handleSend = async () => {
    const text = message.trim();

    if (!text) return;

    await send(text, provider);

    setMessage("");
  };

  return (
    <div
      className={
        variant === "center"
          ? "w-full transition-all duration-500"
          : "sticky bottom-0 left-0 right-0 border-t border-gray-200 bg-white/90 backdrop-blur-md p-6 transition-all duration-500"
      }
    >
      <div
        className={
          variant === "center"
            ? "mx-auto w-full max-w-2xl"
            : "mx-auto w-full max-w-5xl"
        }
      >
        <div
          className="
            flex
            items-end
            gap-3
            rounded-3xl
            border
            border-gray-200
            bg-white
            px-4
            py-3
            shadow-lg
          "
        >
          <ModelSelector
            provider={provider}
            onChange={setProvider}
          />

          <AttachButton />

          <div className="flex-1">
            <MessageInput
              value={message}
              onChange={setMessage}
              onSend={handleSend}
              disabled={loading}
            />
          </div>

          <VoiceButton />

          <SendButton
            disabled={!message.trim()}
            loading={loading}
            onClick={handleSend}
          />
        </div>

        {variant === "bottom" && (
          <p className="mt-3 text-center text-xs text-gray-400">
            AI responses may contain mistakes.
            Verify important information.
          </p>
        )}
      </div>
    </div>
  );
}