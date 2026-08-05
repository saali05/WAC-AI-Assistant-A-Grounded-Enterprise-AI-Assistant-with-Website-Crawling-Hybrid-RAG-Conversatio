import { Mic } from "lucide-react";

interface VoiceButtonProps {
  onClick?: () => void;
}

export default function VoiceButton({
  onClick,
}: VoiceButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="rounded-xl border border-gray-200 bg-white p-3 transition hover:bg-gray-100"
      title="Voice (Coming Soon)"
    >
      <Mic size={18} />
    </button>
  );
}