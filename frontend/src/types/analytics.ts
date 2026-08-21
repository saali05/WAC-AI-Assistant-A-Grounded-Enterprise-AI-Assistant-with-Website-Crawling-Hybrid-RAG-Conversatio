export type UsageSource =
  | "provider_metadata"
  | "provider_headers"
  | "calculated"
  | "unavailable";

export type QuotaScope =
  | "request"
  | "minute"
  | "day"
  | "session"
  | "account"
  | "project"
  | "unknown";

export interface SessionOverview {
  message_count: number;
  ai_request_count: number;
  started_at: string | null;
  last_activity_at: string | null;
}

export interface TokenMetrics {
  input: number;
  output: number;
  total: number;
}

export interface ContextMetrics {
  model: string;
  limit: number;
  remaining: number | null;
  current_prompt_tokens: number | null;
}

export interface QuotaMetrics {
  available: boolean;
  remaining_requests: number | null;
  remaining_tokens: number | null;
  limit_requests: number | null;
  limit_tokens: number | null;
  reset_time: string | null;
  usage_source: UsageSource;
  quota_scope: QuotaScope;
  reason?: string | null;
}

export interface CostMetrics {
  estimated: number;
  currency: string;
  pricing_tier: string;
}

export interface PerformanceMetrics {
  average_latency_ms: number | null;
  average_output_tokens_per_second: number | null;
  average_time_to_first_token_ms: number | null;
}

export interface BreakdownItem {
  request_count: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  estimated_cost: number;
}

export interface VoiceMetrics {
  available?: boolean;
  reason?: string | null;
  session_count: number;
  audio_input_seconds: number;
  audio_output_seconds: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  estimated_cost: number;
}


export interface RequestHistoryItem {
  id: string;
  time: string;
  provider: string;
  model: string;
  request_type: string;
  input_tokens: number | null;
  output_tokens: number | null;
  total_tokens: number | null;
  latency_ms: number | null;
  tokens_per_second: number | null;
  estimated_cost: number | null;
  usage_source?: UsageSource;
  quota_scope?: QuotaScope;
}

export interface SessionAnalyticsData {
  conversation_id: string;
  session: SessionOverview;
  tokens: TokenMetrics;
  context: ContextMetrics;
  quota: QuotaMetrics;
  cost: CostMetrics;
  performance: PerformanceMetrics;
  providers: Record<string, BreakdownItem>;
  models: Record<string, BreakdownItem>;
  voice: VoiceMetrics;
  request_history: RequestHistoryItem[];
}
