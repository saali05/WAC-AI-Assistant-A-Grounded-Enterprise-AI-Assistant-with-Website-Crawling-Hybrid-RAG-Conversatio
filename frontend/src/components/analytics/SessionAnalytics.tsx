import { useEffect, useState } from "react";
import {
  X,
  Activity,
  Zap,
  Clock,
  DollarSign,
  Layers,
  Mic,
  AlertCircle,
  RefreshCw,
  Info,
  ShieldAlert,
  Database,
} from "lucide-react";
import { getAnalytics } from "../../api/analytics";
import type { SessionAnalyticsData } from "../../types/analytics";


interface Props {
  conversationId: string | null;
  isOpen: boolean;
  onClose: () => void;
}

export default function SessionAnalytics({
  conversationId,
  isOpen,
  onClose,
}: Props) {
  const [data, setData] = useState<SessionAnalyticsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAnalytics = async () => {
    if (!conversationId) {
      setData(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await getAnalytics(conversationId);
      setData(res);
    } catch (err: any) {
      setError(err.message || "Failed to load session analytics.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen && conversationId) {
      fetchAnalytics();
    }
  }, [isOpen, conversationId]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col bg-[#0B0F19]/95 text-slate-100 shadow-2xl backdrop-blur-2xl border-l border-white/10 transition-transform duration-300">
      {/* Header */}
      <div className="flex h-16 items-center justify-between border-b border-white/10 px-6 bg-[#080C14]/80">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-red-500/20 text-red-400 border border-red-500/30">
            <Activity size={18} />
          </div>
          <div>
            <h2 className="text-sm font-bold tracking-tight text-white flex items-center gap-2">
              Session Analytics & Usage
              <span className="rounded-full bg-red-500/10 border border-red-500/20 px-2 py-0.5 text-[10px] font-semibold text-red-400">
                Current Session
              </span>
            </h2>
            <p className="text-[11px] text-slate-400">
              Scoped strictly to Conversation {conversationId ? `#${conversationId.slice(-6)}` : "(New)"}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={fetchAnalytics}
            disabled={loading || !conversationId}
            title="Refresh Analytics"
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-slate-400 hover:bg-white/10 hover:text-white transition"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          </button>

          <button
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-slate-400 hover:bg-white/10 hover:text-white transition"
          >
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Main Scrollable Content */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {!conversationId ? (
          <div className="flex flex-col items-center justify-center py-16 text-center text-slate-400">
            <Info size={36} className="mb-3 text-slate-500" />
            <h3 className="text-sm font-semibold text-slate-200">No Active Conversation</h3>
            <p className="mt-1 text-xs max-w-xs">
              Start a new chat session to observe real-time message counters, context usage, and performance metrics.
            </p>
          </div>
        ) : loading && !data ? (
          <div className="flex flex-col items-center justify-center py-16 text-slate-400">
            <RefreshCw size={28} className="animate-spin text-red-500 mb-3" />
            <span className="text-xs">Gathering session analytics...</span>
          </div>
        ) : error ? (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-xs text-red-300 flex items-start gap-3">
            <AlertCircle size={18} className="shrink-0 text-red-400 mt-0.5" />
            <div>
              <p className="font-semibold">Unable to Load Analytics</p>
              <p className="mt-1 text-red-300/80">{error}</p>
            </div>
          </div>
        ) : data ? (
          <>
            {/* SECTION 1: SESSION OVERVIEW */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="rounded-xl border border-white/10 bg-[#141C2E]/60 p-3">
                <span className="text-[10px] font-medium text-slate-400 uppercase tracking-wider">Messages</span>
                <p className="text-lg font-extrabold text-white mt-1">{data.session.message_count}</p>
              </div>

              <div className="rounded-xl border border-white/10 bg-[#141C2E]/60 p-3">
                <span className="text-[10px] font-medium text-slate-400 uppercase tracking-wider">AI Requests</span>
                <p className="text-lg font-extrabold text-rose-400 mt-1">{data.session.ai_request_count}</p>
              </div>

              <div className="rounded-xl border border-white/10 bg-[#141C2E]/60 p-3">
                <span className="text-[10px] font-medium text-slate-400 uppercase tracking-wider">Session Started</span>
                <p className="text-xs font-semibold text-slate-300 mt-1 truncate">
                  {data.session.started_at
                    ? new Date(data.session.started_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                    : "Not started"}
                </p>
              </div>

              <div className="rounded-xl border border-white/10 bg-[#141C2E]/60 p-3">
                <span className="text-[10px] font-medium text-slate-400 uppercase tracking-wider">Last Activity</span>
                <p className="text-xs font-semibold text-slate-300 mt-1 truncate">
                  {data.session.last_activity_at
                    ? new Date(data.session.last_activity_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                    : "None"}
                </p>
              </div>
            </div>

            {/* SECTION 2: DUAL PROGRESS BARS (Context Capacity vs Session Usage) */}
            <div className="rounded-xl border border-white/10 bg-[#141C2E]/80 p-4 space-y-4">
              <div className="flex items-center justify-between border-b border-white/5 pb-2">
                <h3 className="text-xs font-bold text-white flex items-center gap-2">
                  <Zap size={14} className="text-amber-400" />
                  Model Context Capacity (Current Request Prompt)
                </h3>
                <span className="text-[10px] text-slate-400">Model: {data.context.model}</span>
              </div>

              {/* Progress Bar 1: Model Context Capacity (Current Request Prompt) */}
              <div>
                <div className="flex justify-between text-xs font-medium mb-1">
                  <span className="text-slate-300">Current Context:</span>
                  <span className="text-amber-400 font-semibold">
                    {(data.context.current_prompt_tokens || 0).toLocaleString()} / {data.context.limit.toLocaleString()} tokens
                  </span>
                </div>
                <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-amber-500 to-rose-500 transition-all duration-300"
                    style={{
                      width: `${Math.min(
                        100,
                        (((data.context.current_prompt_tokens || 0) / data.context.limit) * 100)
                      ).toFixed(1)}%`,
                    }}
                  />
                </div>
                <p className="text-[10px] text-slate-400 mt-1">
                  Current Prompt Context Capacity Remaining:{" "}
                  <strong className="text-amber-300">
                    {data.context.remaining !== null ? data.context.remaining.toLocaleString() : "Unknown"}
                  </strong>{" "}
                  tokens remaining for this prompt context window.
                </p>
              </div>

              {/* Progress Bar 2: Session Total Tokens Consumed */}
              <div>
                <div className="flex justify-between text-xs font-medium mb-1">
                  <span className="text-slate-300">Session Total Tokens Consumed:</span>
                  <span className="text-rose-400 font-semibold">
                    {data.tokens.total.toLocaleString()} tokens
                  </span>
                </div>
                <div className="flex gap-2 text-[11px] text-slate-400 bg-black/20 p-2 rounded-lg border border-white/5">
                  <div>Input: <strong className="text-slate-200">{data.tokens.input.toLocaleString()}</strong></div>
                  <span>•</span>
                  <div>Output: <strong className="text-slate-200">{data.tokens.output.toLocaleString()}</strong></div>
                  <span>•</span>
                  <div>Total: <strong className="text-slate-200">{data.tokens.total.toLocaleString()}</strong></div>
                </div>
              </div>
            </div>

            {/* SECTION 3: PROVIDER API QUOTA */}
            <div className="rounded-xl border border-white/10 bg-[#141C2E]/80 p-4">
              <h3 className="text-xs font-bold text-white flex items-center gap-2 mb-3">
                <ShieldAlert size={14} className="text-cyan-400" />
                Provider API Quota / Rate Limits
              </h3>

              {data.quota.available ? (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-xs text-emerald-400 font-semibold bg-emerald-500/10 p-2 rounded-lg border border-emerald-500/20">
                    <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                    Header Rate-Limit Exposed ({data.quota.usage_source})
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="p-2.5 rounded-lg bg-black/20 border border-white/5">
                      <span className="text-slate-400 block text-[10px]">Remaining Requests</span>
                      <span className="text-white font-bold text-sm">
                        {data.quota.remaining_requests ?? "N/A"} / {data.quota.limit_requests ?? "N/A"}
                      </span>
                    </div>

                    <div className="p-2.5 rounded-lg bg-black/20 border border-white/5">
                      <span className="text-slate-400 block text-[10px]">Remaining Tokens ({data.quota.quota_scope})</span>
                      <span className="text-white font-bold text-sm">
                        {data.quota.remaining_tokens?.toLocaleString() ?? "N/A"} / {data.quota.limit_tokens?.toLocaleString() ?? "N/A"}
                      </span>
                    </div>
                  </div>

                  {data.quota.reset_time && (
                    <p className="text-[10px] text-slate-400">
                      Rate limit resets in: <strong className="text-slate-200">{data.quota.reset_time}</strong>
                    </p>
                  )}
                </div>
              ) : (
                <div className="rounded-lg border border-white/5 bg-black/20 p-3 text-xs text-slate-400 space-y-1">
                  <div className="flex items-center gap-2 font-semibold text-slate-300">
                    <Info size={14} className="text-slate-400" />
                    Not Available from Provider API
                  </div>
                  <p className="text-[11px] text-slate-400">{data.quota.reason}</p>
                  <p className="text-[10px] text-slate-500 italic mt-1">
                    Note: Free API usage is subject to rate limits, but real-time remaining quota headers are not returned in completion responses by this provider.
                  </p>
                </div>
              )}
            </div>

            {/* SECTION 4: ESTIMATED COST */}
            <div className="rounded-xl border border-white/10 bg-[#141C2E]/80 p-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold text-white flex items-center gap-2">
                  <DollarSign size={14} className="text-emerald-400" />
                  Estimated API Cost
                </h3>
                <span className="rounded-full bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 text-[10px] font-semibold text-emerald-400 uppercase">
                  {data.cost.pricing_tier} Tier
                </span>
              </div>

              <div className="mt-3 flex items-baseline gap-2">
                <span className="text-2xl font-extrabold text-white">
                  ${data.cost.estimated.toFixed(4)}
                </span>
                <span className="text-xs text-slate-400">{data.cost.currency}</span>
              </div>

              {data.cost.pricing_tier === "free" && data.cost.estimated === 0 && (
                <p className="mt-2 text-[11px] text-slate-400">
                  <strong className="text-emerald-400">$0.00 estimated API cost</strong> (Configured under official Free Tier). API usage may still be subject to provider quotas.
                </p>
              )}
            </div>

            {/* SECTION 5: PERFORMANCE */}
            <div className="rounded-xl border border-white/10 bg-[#141C2E]/80 p-4">
              <h3 className="text-xs font-bold text-white flex items-center gap-2 mb-3">
                <Clock size={14} className="text-cyan-400" />
                Real Observed Performance
              </h3>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-lg bg-black/20 border border-white/5">
                  <span className="text-slate-400 block text-[10px]">Average Response Latency</span>
                  <span className="text-white font-bold text-sm">
                    {data.performance.average_latency_ms ? `${data.performance.average_latency_ms} ms` : "N/A"}
                  </span>
                </div>

                <div className="p-3 rounded-lg bg-black/20 border border-white/5">
                  <span className="text-slate-400 block text-[10px]">Observed Output Rate</span>
                  <span className="text-white font-bold text-sm">
                    {data.performance.average_output_tokens_per_second
                      ? `${data.performance.average_output_tokens_per_second} tok/s`
                      : "N/A"}
                  </span>
                </div>
              </div>
            </div>

            {/* SECTION 6: PROVIDER & MODEL BREAKDOWN */}
            <div className="rounded-xl border border-white/10 bg-[#141C2E]/80 p-4 space-y-4">
              <h3 className="text-xs font-bold text-white flex items-center gap-2">
                <Layers size={14} className="text-purple-400" />
                Provider & Model Breakdown
              </h3>

              {/* Providers */}
              <div>
                <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-2">Providers</span>
                <div className="space-y-2">
                  {Object.keys(data.providers).length === 0 ? (
                    <span className="text-xs text-slate-500">No provider usage recorded yet.</span>
                  ) : (
                    Object.entries(data.providers).map(([pName, pStat]) => (
                      <div key={pName} className="flex items-center justify-between text-xs p-2 rounded-lg bg-black/20 border border-white/5">
                        <span className="font-bold text-white capitalize">{pName}</span>
                        <div className="text-slate-400 text-right">
                          <span className="text-slate-200 font-semibold">{pStat.request_count} reqs</span> • {pStat.total_tokens.toLocaleString()} toks
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Models */}
              <div>
                <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-2">Models</span>
                <div className="space-y-2">
                  {Object.keys(data.models).length === 0 ? (
                    <span className="text-xs text-slate-500">No model usage recorded yet.</span>
                  ) : (
                    Object.entries(data.models).map(([mName, mStat]) => (
                      <div key={mName} className="flex items-center justify-between text-xs p-2 rounded-lg bg-black/20 border border-white/5">
                        <span className="font-mono text-slate-300 text-[11px]">{mName}</span>
                        <div className="text-slate-400 text-right">
                          <span className="text-slate-200 font-semibold">{mStat.request_count} reqs</span> • {mStat.total_tokens.toLocaleString()} toks
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>

            {/* SECTION 7: VOICE (GEMINI LIVE) */}
            <div className="rounded-xl border border-white/10 bg-[#141C2E]/80 p-4">
              <h3 className="text-xs font-bold text-white flex items-center gap-2 mb-3">
                <Mic size={14} className="text-rose-400" />
                Gemini Live Voice Usage
              </h3>

              {data.voice.available ? (
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
                  <div className="p-2.5 rounded-lg bg-black/20 border border-white/5">
                    <span className="text-slate-400 block text-[10px]">Voice Sessions</span>
                    <span className="text-white font-bold">{data.voice.session_count}</span>
                  </div>

                  <div className="p-2.5 rounded-lg bg-black/20 border border-white/5">
                    <span className="text-slate-400 block text-[10px]">Audio In (sec)</span>
                    <span className="text-white font-bold">{data.voice.audio_input_seconds.toFixed(1)}s</span>
                  </div>

                  <div className="p-2.5 rounded-lg bg-black/20 border border-white/5">
                    <span className="text-slate-400 block text-[10px]">Audio Out (sec)</span>
                    <span className="text-white font-bold">{data.voice.audio_output_seconds.toFixed(1)}s</span>
                  </div>
                </div>
              ) : (
                <div className="rounded-lg border border-white/5 bg-black/20 p-3 text-xs text-slate-400 space-y-1">
                  <div className="flex items-center gap-2 font-semibold text-slate-300">
                    <Info size={14} className="text-slate-400" />
                    Usage Data Unavailable
                  </div>
                  <p className="text-[11px] text-slate-400">
                    {data.voice.session_count > 0
                      ? "Live usage metadata is not returned by the persistent WebSocket API. Token estimates are not fabricated."
                      : "No voice sessions initiated in this conversation."}
                  </p>
                </div>
              )}
            </div>


            {/* SECTION 8: REQUEST HISTORY */}
            <div className="rounded-xl border border-white/10 bg-[#141C2E]/80 p-4">
              <h3 className="text-xs font-bold text-white flex items-center gap-2 mb-3">
                <Database size={14} className="text-blue-400" />
                Request History ({data.request_history.length})
              </h3>

              {data.request_history.length === 0 ? (
                <p className="text-xs text-slate-500 py-2">No AI requests logged in this session.</p>
              ) : (
                <div className="max-h-60 overflow-y-auto space-y-2 pr-1">
                  {data.request_history.map((item) => (
                    <div
                      key={item.id}
                      className="p-2.5 rounded-lg bg-black/30 border border-white/5 text-xs space-y-1"
                    >
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="font-semibold text-slate-200 capitalize">{item.provider} ({item.model})</span>
                        <span className="text-slate-500">
                          {new Date(item.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                        </span>
                      </div>

                      <div className="flex justify-between text-[10px] text-slate-400">
                        <span>Tokens: In <strong className="text-slate-300">{item.input_tokens ?? 0}</strong> / Out <strong className="text-slate-300">{item.output_tokens ?? 0}</strong></span>
                        <span>Latency: <strong className="text-slate-300">{item.latency_ms ? `${item.latency_ms}ms` : "N/A"}</strong></span>
                        <span>Cost: <strong className="text-slate-300">${(item.estimated_cost || 0).toFixed(4)}</strong></span>
                      </div>

                      {item.usage_source && (
                        <div className="text-[9px] text-slate-500 pt-0.5 flex justify-between">
                          <span>Source: {item.usage_source}</span>
                          <span>Scope: {item.quota_scope || "unknown"}</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}
