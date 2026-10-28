"use client";

import { HeadRef } from "@/lib/api";

type Props = {
  heads: HeadRef[];
  loading: boolean;
  onRemove: (h: HeadRef) => void;
  onClear: () => void;
  onRun: () => void;
};

export default function AblationPanel({ heads, loading, onRemove, onClear, onRun }: Props) {
  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900 p-4">
      <h2 className="mb-1 font-semibold">Ablation experiment</h2>
      <p className="mb-3 text-sm text-slate-400">
        Zero out the selected heads and compare the model output with the baseline.
      </p>

      {heads.length === 0 ? (
        <p className="text-sm text-slate-500">
          No heads selected. Pick a head above and press "Add to ablation".
        </p>
      ) : (
        <div className="flex flex-wrap gap-2">
          {heads.map((h) => (
            <button
              key={`${h.layer}-${h.head}`}
              onClick={() => onRemove(h)}
              className="rounded-full bg-red-900 px-3 py-1 font-mono text-xs hover:bg-red-800"
              title="Remove"
            >
              L{h.layer}H{h.head} x
            </button>
          ))}
        </div>
      )}

      <div className="mt-4 flex gap-3">
        <button
          onClick={onRun}
          disabled={loading || heads.length === 0}
          className="rounded-lg bg-red-600 px-4 py-2 text-sm font-medium hover:bg-red-500 disabled:opacity-50"
        >
          {loading ? "Running..." : "Run ablation"}
        </button>
        <button
          onClick={onClear}
          disabled={heads.length === 0}
          className="rounded-lg bg-slate-800 px-4 py-2 text-sm hover:bg-slate-700 disabled:opacity-50"
        >
          Clear
        </button>
      </div>
    </section>
  );
}