const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface HeadRef {
  layer: number;
  head: number;
}

export interface TokenMeta {
  index: number;
  token_id: number;
  raw: string;
  clean: string;
  display: string;
  is_space_prefixed: boolean;
  is_newline: boolean;
  is_special: boolean;
  byte_length: number;
}

export interface HeadData {
  layer: number;
  head: number;
  label: string;
  matrix: number[][];
  matrix_normalized: number[][];
  seq_len: number;
  entropies: number[];
  mean_entropy: number;
  max_possible_entropy: number;
  focus_style: string;
}

export interface LayerSummary {
  layer: number;
  avg_entropy: number;
  min_entropy: number;
  max_entropy: number;
  num_focused_heads: number;
  num_diffuse_heads: number;
}

export interface TopToken {
  token: string;
  token_id: number;
  probability: number;
  rank: number;
  percentage: string;
  bar_width: number;
  is_top: boolean;
}

export interface AnalyzeResponse {
  tokens: string[];
  num_tokens: number;
  attention: {
    heads: HeadData[];
    total_heads: number;
    layer_summaries: LayerSummary[];
    most_focused_head: string;
    most_diffuse_head: string;
  };
  top_output_tokens: TopToken[];
  generated_text: string;
  model: string;
  elapsed_seconds: number;
  token_metadata: TokenMeta[];
}

export interface ProbDiff {
  token: string;
  baseline_prob: number;
  ablated_prob: number;
  delta: number;
  direction: string;
  color: string;
  baseline_pct: string;
  ablated_pct: string;
  delta_pct: string;
  abs_delta: number;
}

export interface RunSummary {
  generated_text: string;
  top_tokens: { token: string; token_id: number; probability: number }[];
}

export interface AblateResponse {
  prompt: string;
  ablated_heads: HeadRef[];
  baseline: RunSummary;
  ablated: RunSummary;
  probability_diff: ProbDiff[];
  output_changed: boolean;
  elapsed_seconds: number;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {}
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const analyze = (prompt: string, maxNewTokens = 10) =>
  post<AnalyzeResponse>("/analyze", { prompt, max_new_tokens: maxNewTokens });

export const ablate = (prompt: string, heads: HeadRef[]) =>
  post<AblateResponse>("/ablate", { prompt, heads_to_ablate: heads });