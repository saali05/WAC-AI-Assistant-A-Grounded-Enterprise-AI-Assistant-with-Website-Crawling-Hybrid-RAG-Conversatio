import ConversationItem from "./ConversationItem";
import { useChat } from "../../context/ChatContext";

export default function ConversationList() {
  const {
    conversations,
    loadConversation,
  } = useChat();

  return (
    <div className="flex-1 overflow-y-auto">

      <div className="px-4 py-2">

        <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-400">
          Recent
        </h2>

      </div>

      <div className="space-y-1 px-2">

        {conversations.map((conversation) => (
          <ConversationItem
            key={conversation.id}
            id={conversation.id}
            title={conversation.title}
            onClick={loadConversation}
          />
        ))}

      </div>

    </div>
  );
}