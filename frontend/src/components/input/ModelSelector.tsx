import { Sparkles, Zap, ChevronDown } from "lucide-react";
import type { Provider } from "../../types/chat";

interface ModelSelectorProps {
  provider: Provider;
  onChange: (provider: Provider) => void;
}

export default function ModelSelector({
  provider,
  onChange,
}: ModelSelectorProps) {
  return (
    <div className="relative inline-block">
      <div className="flex items-center gap-1.5 rounded-xl border border-white/10 bg-[#121929] px-3 py-2 text-xs font-semibold text-slate-200 transition-all hover:border-red-500/40 hover:bg-[#182236]">
        {provider === "gemini" ? (
          <Sparkles size={10} className="text-amber-400" />
        ) : (
          <Zap size={10} className="text-violet-400" />
        )}

        <select
          value={provider}
          onChange={(e) => onChange(e.target.value as Provider)}
          className="appearance-none bg-transparent pr-4 font-semibold text-slate-200 outline-none cursor-pointer"
        >
          <option value="gemini" className="bg-[#0F1626] text-white">
            Gemini 2.5
          </option>
          <option value="groq" className="bg-[#0F1626] text-white">
            Groq Llama 3
          </option>
        </select>
        
        <ChevronDown size={14} className="pointer-events-none absolute right-2 text-slate-400" />
      </div>
    </div>
  );
}