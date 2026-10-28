"use client";

import { useState } from "react";
import {
  analyze,
  ablate,
  AnalyzeResponse,
  AblateResponse,
  HeadRef,
} from "@/lib/api";
import TokenInput from "@/components/TokenInput";
import HeadExplorer from "@/components/HeadExplorer";
import CircuitGraph from "@/components/CircuitGraph";
import AblationPanel from "@/components/AblationPanel";
import OutputDiff from "@/components/OutputDiff";

const DEFAULT_PROMPT = "The Eiffel Tower is located in";

function parseLabel(label: string): HeadRef {
  const m = /L(\d+)H(\d+)/.exec(label);
  return m ? { layer: Number(m[1]), head: Number(m[2]) } : { layer: 0, head: 0 };
}

export default function Home() {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [data, setData] = useState<AnalyzeResponse | null>(null);
  const [result, setResult] = useState<AblateResponse | null>(null);
  const [selected, setSelected] = useState<HeadRef>({ layer: 0, head: 0 });
  const [ablated, setAblated] = useState<HeadRef[]>([]);
  const [loading, setLoading] = useState(false);
  const [ablating, setAblating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const same = (a: HeadRef, b: HeadRef) => a.layer === b.layer && a.head === b.head;

  async function runAnalyze(p: string, n: number) {
    setLoading(true);
    setError(null);
    try {
      const d = await analyze(p, n);
      setPrompt(p);
      setData(d);
      setResult(null);
      setSelected(parseLabel(d.attention.most_focused_head));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  async function runAblation() {
    if (!ablated.length) return;
    setAblating(true);
    setError(null);
    try {
      setResult(await ablate(prompt, ablated));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setAblating(false);
    }
  }

  function toggleAblate(h: HeadRef) {
    setResult(null);
    setAblated((cur) =>
      cur.some((x) => same(x, h)) ? cur.filter((x) => !same(x, h)) : [...cur, h]
    );
  }

  return (
    <main className="mx-auto max-w-6xl space-y-6 p-6">
      <header>
        <h1 className="text-3xl font-bold">NeuroScope</h1>
        <p className="text-slate-400">
          Inspect GPT-2 attention heads and ablate them to see what changes.
        </p>
      </header>

      <TokenInput initialPrompt={DEFAULT_PROMPT} loading={loading} onSubmit={runAnalyze} />

      {error && (
        <div className="rounded-lg border border-red-800 bg-red-950 p-3 text-sm text-red-200">
          {error}
        </div>
      )}

      {data && (
        <>
          <section className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <h2 className="mb-3 font-semibold">Tokens</h2>
            <div className="flex flex-wrap gap-2">
              {data.token_metadata.map((t) => (
                <span
                  key={t.index}
                  className="rounded bg-slate-800 px-2 py-1 font-mono text-sm"
                  title={`id ${t.token_id}`}
                >
                  {t.display}
                </span>
              ))}
            </div>
            <p className="mt-3 text-sm text-slate-400">
              Generated: <span className="font-mono text-slate-200">{data.generated_text}</span>
            </p>
          </section>

          <section className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <h2 className="mb-3 font-semibold">Top next-token predictions</h2>
            <div className="space-y-1">
              {data.top_output_tokens.map((t) => (
                <div key={t.token_id} className="flex items-center gap-3 text-sm">
                  <span className="w-24 truncate font-mono">{t.token}</span>
                  <div className="h-3 flex-1 rounded bg-slate-800">
                    <div
                      className="h-3 rounded bg-sky-500"
                      style={{ width: `${Math.max(2, t.bar_width)}%` }}
                    />
                  </div>
                  <span className="w-14 text-right text-slate-400">{t.percentage}</span>
                </div>
              ))}
            </div>
          </section>

          <HeadExplorer
            data={data}
            selected={selected}
            onSelect={setSelected}
            ablated={ablated}
            onToggleAblate={toggleAblate}
          />

          <CircuitGraph data={data} selected={selected} ablated={ablated} onSelect={setSelected} />

          <AblationPanel
            heads={ablated}
            loading={ablating}
            onRemove={toggleAblate}
            onClear={() => {
              setAblated([]);
              setResult(null);
            }}
            onRun={runAblation}
          />

          {result && <OutputDiff result={result} />}
        </>
      )}
    </main>
  );
}