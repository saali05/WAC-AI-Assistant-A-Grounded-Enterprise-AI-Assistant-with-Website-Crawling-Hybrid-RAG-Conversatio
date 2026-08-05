import SidebarHeader from "./SidebarHeader";
import SearchChats from "./SearchChats";
import ConversationList from "./ConversationList";

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export default function Sidebar({
  collapsed,
  onToggle,
}: SidebarProps) {
  return (
    <aside
      className={`
        bg-white
        border-r
        border-gray-200
        transition-all
        duration-300
        ${collapsed ? "w-[72px]" : "w-[280px]"}
      `}
    >
      <div className="flex h-full flex-col">
        <SidebarHeader
          collapsed={collapsed}
          onToggle={onToggle}
        />

        {!collapsed && (
          <>
            <SearchChats />
            <ConversationList />
          </>
        )}
      </div>
    </aside>
  );
}