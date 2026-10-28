"use client";

import { AblateResponse } from "@/lib/api";

export default function OutputDiff({ result }: { result: AblateResponse }) {
  const maxAbs = Math.max(0.0001, ...result.probability_diff.map((d) => d.abs_delta));

  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900 p-4">
      <div className="mb-3 flex items-center gap-3">
        <h2 className="font-semibold">Result</h2>
        <span
          className={`rounded px-2 py-0.5 text-xs ${
            result.output_changed ? "bg-red-900 text-red-200" : "bg-slate-800 text-slate-300"
          }`}
        >
          {result.output_changed ? "Generated text changed" : "Generated text unchanged"}
        </span>
        <span className="ml-auto text-xs text-slate-500">{result.elapsed_seconds}s</span>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <div className="rounded-lg bg-slate-950 p-3">
          <div className="mb-1 text-xs uppercase tracking-wide text-slate-500">Baseline</div>
          <p className="font-mono text-sm">{result.baseline.generated_text}</p>
        </div>
        <div className="rounded-lg bg-slate-950 p-3">
          <div className="mb-1 text-xs uppercase tracking-wide text-slate-500">Ablated</div>
          <p className="font-mono text-sm">{result.ablated.generated_text}</p>
        </div>
      </div>

      <h3 className="mb-2 mt-5 text-sm font-semibold">Next-token probability shift</h3>
      <div className="space-y-1">
        {result.probability_diff.map((d) => (
          <div key={d.token} className="flex items-center gap-3 text-sm">
            <span className="w-24 truncate font-mono">{d.token}</span>
            <div className="h-3 flex-1 rounded bg-slate-800">
              <div
                className={`h-3 rounded ${d.delta < 0 ? "bg-red-500" : "bg-emerald-500"}`}
                style={{ width: `${Math.max(1, (d.abs_delta / maxAbs) * 100)}%` }}
              />
            </div>
            <span className="w-16 text-right text-xs text-slate-400">{d.baseline_pct}</span>
            <span className="w-16 text-right text-xs text-slate-400">{d.ablated_pct}</span>
            <span
              className={`w-16 text-right text-xs ${d.delta < 0 ? "text-red-400" : "text-emerald-400"}`}
            >
              {d.delta_pct}
            </span>
          </div>
        ))}
      </div>
      <p className="mt-2 text-xs text-slate-500">Columns: baseline, ablated, change.</p>
    </section>
  );
}