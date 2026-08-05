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
    <select
      value={provider}
      onChange={(e) => onChange(e.target.value as Provider)}
      className="rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm font-medium outline-none transition focus:border-blue-500"
    >
      <option value="gemini">Gemini</option>
      <option value="groq">Groq</option>
    </select>
  );
}