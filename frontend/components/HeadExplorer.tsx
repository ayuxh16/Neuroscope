"use client";

import { AnalyzeResponse, HeadData, HeadRef } from "@/lib/api";

type Props = {
  data: AnalyzeResponse;
  selected: HeadRef;
  onSelect: (h: HeadRef) => void;
  ablated: HeadRef[];
  onToggleAblate: (h: HeadRef) => void;
};

const LAYERS = Array.from({ length: 12 }, (_, i) => i);
const HEADS = Array.from({ length: 12 }, (_, i) => i);

export function focusScore(h: HeadData): number {
  if (!h.max_possible_entropy) return 0;
  return Math.max(0, Math.min(1, 1 - h.mean_entropy / h.max_possible_entropy));
}

export default function HeadExplorer({ data, selected, onSelect, ablated, onToggleAblate }: Props) {
  const byKey = new Map<string, HeadData>(
    data.attention.heads.map((h) => [`${h.layer}-${h.head}`, h])
  );
  const current = byKey.get(`${selected.layer}-${selected.head}`);
  const labels = data.token_metadata.map((t) => t.display);
  const isAblated = (l: number, h: number) => ablated.some((a) => a.layer === l && a.head === h);

  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900 p-4">
      <h2 className="mb-1 font-semibold">Layer explorer (12 layers x 12 heads)</h2>
      <p className="mb-4 text-sm text-slate-400">
        Brighter cells have more focused attention (lower entropy). Click a head to inspect it.
      </p>

      <div className="flex flex-col gap-6 lg:flex-row">
        <div className="overflow-x-auto">
          <div className="inline-block">
            <div className="mb-1 ml-8 flex gap-1 text-[10px] text-slate-500">
              {HEADS.map((h) => (
                <span key={h} className="w-6 text-center">H{h}</span>
              ))}
            </div>
            {LAYERS.map((l) => (
              <div key={l} className="mb-1 flex items-center gap-1">
                <span className="w-7 text-[10px] text-slate-500">L{l}</span>
                {HEADS.map((h) => {
                  const hd = byKey.get(`${l}-${h}`);
                  const f = hd ? focusScore(hd) : 0;
                  const isSel = selected.layer === l && selected.head === h;
                  return (
                    <button
                      key={h}
                      onClick={() => onSelect({ layer: l, head: h })}
                      title={`L${l}H${h} - focus ${(f * 100).toFixed(0)}%`}
                      className={`h-6 w-6 rounded-sm ${
                        isAblated(l, h) ? "ring-2 ring-red-500" : ""
                      } ${isSel ? "outline outline-2 outline-white" : ""}`}
                      style={{ backgroundColor: `rgba(56,189,248,${0.12 + f * 0.88})` }}
                    />
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        <div className="min-w-0 flex-1">
          {current ? (
            <>
              <div className="mb-3 flex flex-wrap items-center gap-3">
                <h3 className="font-mono text-lg">{current.label}</h3>
                <span className="rounded bg-slate-800 px-2 py-0.5 text-xs">
                  {current.focus_style}
                </span>
                <span className="text-xs text-slate-400">
                  mean entropy {current.mean_entropy.toFixed(2)} / {current.max_possible_entropy.toFixed(2)}
                </span>
                <button
                  onClick={() => onToggleAblate({ layer: current.layer, head: current.head })}
                  className={`ml-auto rounded-lg px-3 py-1.5 text-sm font-medium ${
                    isAblated(current.layer, current.head)
                      ? "bg-slate-700 hover:bg-slate-600"
                      : "bg-red-600 hover:bg-red-500"
                  }`}
                >
                  {isAblated(current.layer, current.head) ? "Remove from ablation" : "Add to ablation"}
                </button>
              </div>

              <div className="overflow-x-auto">
                <table className="border-separate border-spacing-0.5 text-xs">
                  <thead>
                    <tr>
                      <th />
                      {labels.map((t, j) => (
                        <th key={j} className="max-w-[56px] truncate px-1 font-mono text-slate-400">
                          {t}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {current.matrix_normalized.map((row, i) => (
                      <tr key={i}>
                        <th className="max-w-[72px] truncate pr-2 text-right font-mono text-slate-400">
                          {labels[i]}
                        </th>
                        {row.map((v, j) => (
                          <td
                            key={j}
                            title={`${labels[i]} -> ${labels[j]}: ${(current.matrix[i]?.[j] ?? 0).toFixed(3)}`}
                            className="h-7 w-10 rounded-sm text-center"
                            style={{ backgroundColor: `rgba(251,146,60,${Math.min(1, v)})` }}
                          />
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-2 text-xs text-slate-500">
                Rows = token doing the attending, columns = token being attended to.
              </p>
            </>
          ) : (
            <p className="text-sm text-slate-400">Select a head.</p>
          )}
        </div>
      </div>
    </section>
  );
}