import { MessageSquare } from "lucide-react";

interface Props {
  id: string;
  title: string;
  onClick: (id: string) => void;
}

export default function ConversationItem({
  id,
  title,
  onClick,
}: Props) {
  return (
    <button
      onClick={() => onClick(id)}
      className="
        flex
        w-full
        items-center
        gap-3
        rounded-xl
        px-4
        py-3
        text-left
        transition
        hover:bg-gray-100
      "
    >
      <MessageSquare size={18} />

      <span className="truncate text-sm">
        {title}
      </span>
    </button>
  );
}