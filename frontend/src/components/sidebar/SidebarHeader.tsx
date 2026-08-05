import { Menu, SquarePen } from "lucide-react";
import { useChat } from "../../context/ChatContext";

interface Props {
  collapsed: boolean;
  onToggle: () => void;
}

export default function SidebarHeader({
  collapsed,
  onToggle,
}: Props) {
  const { newChat } = useChat();

  return (
    <div className="border-b border-gray-200 p-4">

      <div className="flex items-center justify-between">

        <button
          onClick={onToggle}
          className="rounded-xl p-2 hover:bg-gray-100"
        >
          <Menu size={20} />
        </button>

        {!collapsed && (
          <button
            onClick={newChat}
            className="
              flex
              items-center
              gap-2
              rounded-xl
              bg-blue-600
              px-4
              py-2
              text-white
              hover:bg-blue-700
            "
          >
            <SquarePen size={18} />
            New Chat
          </button>
        )}

      </div>

    </div>
  );
}