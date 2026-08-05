import { Paperclip } from "lucide-react";

interface AttachButtonProps {
  onClick?: () => void;
}

export default function AttachButton({
  onClick,
}: AttachButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="rounded-xl border border-gray-200 bg-white p-3 transition hover:bg-gray-100"
      title="Attach File"
    >
      <Paperclip size={18} />
    </button>
  );
}