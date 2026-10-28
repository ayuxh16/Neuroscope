"use client";

import { AnalyzeResponse, HeadRef } from "@/lib/api";
import { focusScore } from "@/components/HeadExplorer";

type Props = {
  data: AnalyzeResponse;
  selected: HeadRef;
  ablated: HeadRef[];
  onSelect: (h: HeadRef) => void;
};

const COL = 56;
const ROW = 22;
const PAD = 30;

export default function CircuitGraph({ data, selected, ablated, onSelect }: Props) {
  const width = PAD * 2 + COL * 11;
  const height = PAD * 2 + ROW * 11;

  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900 p-4">
      <h2 className="mb-1 font-semibold">Focused-head map</h2>
      <p className="mb-3 text-sm text-slate-400">
        Each node is one head (x = layer, y = head). Larger, brighter nodes attend more sharply.
        Red rings mark ablated heads.
      </p>
      <div className="overflow-x-auto">
        <svg width={width} height={height} className="block">
          {Array.from({ length: 12 }, (_, l) => (
            <text key={l} x={PAD + l * COL} y={14} textAnchor="middle" fontSize={10} fill="#64748b">
              L{l}
            </text>
          ))}
          {data.attention.heads.map((h) => {
            const f = focusScore(h);
            const cx = PAD + h.layer * COL;
            const cy = PAD + h.head * ROW;
            const isSel = selected.layer === h.layer && selected.head === h.head;
            const isAbl = ablated.some((a) => a.layer === h.layer && a.head === h.head);
            return (
              <g key={h.label} onClick={() => onSelect({ layer: h.layer, head: h.head })} className="cursor-pointer">
                <circle
                  cx={cx}
                  cy={cy}
                  r={3 + f * 7}
                  fill={`rgba(56,189,248,${0.2 + f * 0.8})`}
                  stroke={isAbl ? "#ef4444" : isSel ? "#ffffff" : "none"}
                  strokeWidth={2}
                >
                  <title>{`${h.label} - focus ${(f * 100).toFixed(0)}%`}</title>
                </circle>
              </g>
            );
          })}
        </svg>
      </div>
    </section>
  );
}