
import Welcome from "../components/chat/Welcome";
import ChatWindow from "../components/chat/ChatWindow";
import ChatInput from "../components/input/ChatInput";

import { useChat } from "../context/ChatContext";


export default function ChatPage() {
  const {
    messages,
  } = useChat();
  console.log(messages);

  const hasMessages =
    messages.length > 0;

  return (
    <div className="flex h-full w-full justify-center bg-[#F8FAFC]">

      <div className="relative flex h-full w-full max-w-6xl flex-col">

        {!hasMessages ? (
          <div className="flex flex-1 items-center justify-center">

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
            <div className="flex-1 overflow-y-auto px-6 py-8">

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
  );
}