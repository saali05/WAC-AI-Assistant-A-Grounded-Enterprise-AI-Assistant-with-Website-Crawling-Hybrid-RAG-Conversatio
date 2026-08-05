import { Search } from "lucide-react";

export default function SearchChats() {
  return (
    <div className="p-4">

      <div
        className="
        flex
        items-center
        rounded-xl
        border
        border-gray-200
        px-3
        py-2
      "
      >
        <Search
          size={18}
          className="text-gray-400"
        />

        <input
          placeholder="Search chats..."
          className="
          ml-3
          flex-1
          bg-transparent
          outline-none
          text-sm
        "
        />

      </div>

    </div>
  );
}