import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { User, Sparkles, Copy, Check, Terminal } from "lucide-react";
import { useChat } from "../../context/ChatContext";
import type { Message } from "../../types/chat";

interface ChatWindowProps {
  messages: Message[];
}

export default function ChatWindow({ messages }: ChatWindowProps) {
  const { loading } = useChat();
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const handleCopy = (content: string, index: number) => {
    navigator.clipboard.writeText(content);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 px-4 sm:px-8 py-8">
      {messages.map((message, index) => {
        const isUser = message.role === "user";

        return (
          <div
            key={index}
            className={`flex gap-3 sm:gap-4 ${isUser ? "flex-row-reverse" : "flex-row"} items-start`}
          >
            {/* Avatar Badge */}
            <div
              className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl shadow-lg ${
                isUser
                  ? "bg-gradient-to-br from-red-600 to-rose-700 text-white shadow-red-600/20"
                  : "bg-gradient-to-br from-[#1A2338] to-[#0E1524] border border-white/10 text-red-400"
              }`}
            >
              {isUser ? (
                <User size={18} />
              ) : (
                <span className="font-extrabold text-[10px] tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-red-500 to-rose-300">
                  WAC
                </span>
              )}
            </div>

            {/* Message Bubble Container */}
            <div
              className={`group relative max-w-[85%] sm:max-w-[78%] rounded-2xl px-5 py-4 shadow-xl transition-all duration-200 ${
                isUser
                  ? "bg-gradient-to-r from-red-950/40 via-[#182033] to-[#121927] border border-red-500/25 text-slate-100 rounded-tr-xs"
                  : "wac-glass text-slate-200 rounded-tl-xs"
              }`}
            >
              {/* Message Header info & Copy Button */}
              <div className="flex items-center justify-between mb-2 pb-1 border-b border-white/5 text-[11px] font-medium text-slate-400">
                <span className="flex items-center gap-1.5">
                  {!isUser && <Sparkles size={12} className="text-red-400" />}
                  {isUser ? "You" : "WAC AI Assistant"}
                </span>

                <button
                  onClick={() => handleCopy(message.content, index)}
                  className="flex items-center gap-1 opacity-60 hover:opacity-100 transition-opacity text-slate-400 hover:text-white px-1.5 py-0.5 rounded bg-white/5"
                  title="Copy content"
                >
                  {copiedIndex === index ? (
                    <>
                      <Check size={12} className="text-emerald-400" />
                      <span className="text-[10px] text-emerald-400">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy size={12} />
                      <span className="text-[10px]">Copy</span>
                    </>
                  )}
                </button>
              </div>

              {/* Markdown Content */}
              <div className="prose prose-invert max-w-none text-sm leading-relaxed tracking-normal text-slate-200 prose-p:my-1.5 prose-headings:text-white prose-headings:font-bold prose-code:text-red-300 prose-code:bg-red-950/40 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-pre:bg-[#070A11] prose-pre:border prose-pre:border-white/10 prose-pre:rounded-xl prose-a:text-red-400 underline-offset-4">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.content}
                </ReactMarkdown>
              </div>

              {/* RAG Sources Section */}
              {message.sources && message.sources.length > 0 && (
                <div className="mt-4 pt-3 border-t border-white/10">
                  <div className="text-[11px] font-semibold text-slate-400 mb-2 tracking-wide uppercase flex items-center gap-1.5">
                    <span className="h-1.5 w-1.5 rounded-full bg-red-400"></span>
                    Sources
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {message.sources.map((source, sIdx) => (
                      <a
                        key={sIdx}
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="group flex flex-col p-2 rounded-xl bg-white/5 border border-white/10 hover:border-red-500/40 hover:bg-white/10 transition-all text-xs max-w-xs"
                      >
                        <span className="font-semibold text-slate-200 group-hover:text-red-300 truncate">
                          {source.title}
                        </span>
                        {source.heading && (
                          <span className="text-[10px] text-slate-400 truncate">
                            {source.heading}
                          </span>
                        )}
                        <span className="text-[10px] text-red-400/80 truncate mt-0.5 font-mono">
                          {source.url.replace(/^https?:\/\//, "")}
                        </span>
                      </a>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        );
      })}

      {/* Typing Indicator when AI response is loading */}
      {loading && (
        <div className="flex gap-3 sm:gap-4 flex-row items-start animate-fade-in">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-[#1A2338] to-[#0E1524] border border-red-500/30 text-red-400 shadow-lg">
            <span className="font-extrabold text-[10px] tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-red-500 to-rose-300">
              WAC
            </span>
          </div>

          <div className="wac-glass rounded-2xl rounded-tl-xs px-5 py-4 border border-red-500/20 flex items-center gap-3">
            <Terminal size={16} className="text-red-400 animate-spin" />
            <span className="text-xs font-medium text-slate-300">
              WAC AI is crafting your response...
            </span>
            <div className="flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-red-500 animate-bounce [animation-delay:-0.3s]"></span>
              <span className="h-1.5 w-1.5 rounded-full bg-rose-400 animate-bounce [animation-delay:-0.15s]"></span>
              <span className="h-1.5 w-1.5 rounded-full bg-violet-400 animate-bounce"></span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}