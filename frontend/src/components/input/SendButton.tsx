import { ArrowUp, LoaderCircle } from "lucide-react";

interface SendButtonProps {
  disabled: boolean;
  loading: boolean;
  onClick: () => void;
}

export default function SendButton({
  disabled,
  loading,
  onClick,
}: SendButtonProps) {
  return (
    <button
      type="button"
      disabled={disabled || loading}
      onClick={onClick}
      className="flex h-11 w-11 items-center justify-center rounded-xl bg-blue-600 text-white transition hover:bg-blue-500 hover:scale-105 transition-all disabled:cursor-not-allowed disabled:bg-gray-300"
    >
      {loading ? (
        <LoaderCircle
          size={18}
          className="animate-spin"
        />
      ) : (
        <ArrowUp size={18} />
      )}
    </button>
  );
}