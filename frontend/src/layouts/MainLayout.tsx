import { type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export default function MainLayout({ children }: Props) {
  return (
    <div className="relative h-screen w-full overflow-hidden bg-[#080C14] text-slate-100 selection:bg-red-500/30 selection:text-red-200">
      {/* Background radial glow & grid patterns */}
      <div className="pointer-events-none absolute inset-0 wac-bg-pattern z-0" />
      <div className="pointer-events-none absolute inset-0 wac-grid-overlay opacity-60 z-0" />
      
      {/* Main Container */}
      <main className="relative z-10 flex h-full w-full flex-col">
        {children}
      </main>
    </div>
  );
}