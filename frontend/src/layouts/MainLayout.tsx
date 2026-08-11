import { type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export default function MainLayout({
  children,
}: Props) {
  return (
    <div className="h-screen w-full overflow-hidden bg-[#F8FAFC]">
      <main className="h-full w-full">
        {children}
      </main>
    </div>
  );
}