import { type ReactNode, useState } from "react";

import Sidebar from "../components/sidebar/sidebar";

interface Props {
  children: ReactNode;
}

export default function MainLayout({
  children,
}: Props) {
  const [collapsed, setCollapsed] =
    useState(false);

  return (
    <div className="flex h-screen bg-[#F8FAFC]">

      <Sidebar
        collapsed={collapsed}
        onToggle={() =>
          setCollapsed(!collapsed)
        }
      />

      <main className="flex flex-1 justify-center overflow-hidden">
        <div className="flex h-full w-full max-w-7xl">
          {children}
        </div>
      </main>

    </div>
  );
}