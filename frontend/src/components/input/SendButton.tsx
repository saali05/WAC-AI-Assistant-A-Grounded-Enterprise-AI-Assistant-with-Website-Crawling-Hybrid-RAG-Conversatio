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
      className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-r from-red-600 via-rose-500 to-violet-600 text-white shadow-lg shadow-red-600/30 transition-all duration-200 hover:scale-105 hover:shadow-red-500/40 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:scale-100 disabled:shadow-none"
    >
      {loading ? (
        <LoaderCircle
          size={18}
          className="animate-spin"
        />
      ) : (
        <ArrowUp size={18} className="stroke-[2.5]" />
      )}
    </button>
  );
}