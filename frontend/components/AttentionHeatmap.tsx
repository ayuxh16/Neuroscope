"use client";
import { useState } from "react";
import { TokenData } from "@/lib/api";

interface Props {
  attentionData: number[][][][];
  tokens: TokenData[];
  selectedToken: number;
  selectedHead: { layer: number; head: number } | null;
  onSelectHead: (h: { layer: number; head: number } | null) => void;
  ablatedHeads: Array<[number, number]>;
  onAblate: (layer: number, head: number) => void;
}

function getHeadEntropy(matrix: number[][]): number {
  const flat = matrix.flat().filter(v => v > 0);
  return -flat.reduce((s, p) => s + p * Math.log(p + 1e-9), 0);
}

function focusColor(entropy: number): string {
  // lower entropy = more focused = green; high = red
  if (entropy < 0.5) return "var(--green)";
  if (entropy < 1.5) return "var(--amber)";
  return "var(--red)";
}

function cellBg(value: number): string {
  // value 0-1 → dark purple to bright cyan
  const r = Math.round(139 * value * 0.4);
  const g = Math.round(211 * value);
  const b = Math.round(238 * value);
  return `rgb(${r}, ${g}, ${b})`;
}

export default function AttentionHeatmap({
  attentionData, tokens, selectedToken, selectedHead, onSelectHead, ablatedHeads, onAblate,
}: Props) {
  const [hoveredHead, setHoveredHead] = useState<{ layer: number; head: number } | null>(null);

  const displayHead = hoveredHead || selectedHead;
  const isAblated = (l: number, h: number) => ablatedHeads.some(([al, ah]) => al === l && ah === h);

  return (
    <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
      {/* Head grid */}
      <div style={{ flex: "0 0 auto" }}>
        <div style={{ fontSize: 11, color: "var(--text-muted)", letterSpacing: "0.1em", marginBottom: 10 }}>
          ALL HEADS (Layer × Head)
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(12, 28px)", gap: 3 }}>
          {/* Column labels */}
          {Array.from({ length: 12 }, (_, i) => (
            <div key={i} style={{ fontSize: 9, color: "var(--text-muted)", textAlign: "center", paddingBottom: 2 }}>{i}</div>
          ))}

          {attentionData.map((layer, li) =>
            layer.map((head, hi) => {
              const entropy = getHeadEntropy(head);
              const color = focusColor(entropy);
              const ablated = isAblated(li, hi);
              const isSelected = selectedHead?.layer === li && selectedHead?.head === hi;
              const isHovered = hoveredHead?.layer === li && hoveredHead?.head === hi;

              return (
                <div
                  key={`${li}-${hi}`}
                  style={{
                    width: 28, height: 28, borderRadius: 3, cursor: "pointer",
                    background: ablated ? "transparent" : color,
                    opacity: ablated ? 0.25 : isSelected || isHovered ? 1 : 0.65,
                    border: isSelected ? "2px solid var(--cyan)" : isHovered ? "2px solid var(--purple)" : ablated ? "1px dashed var(--text-muted)" : "1px solid transparent",
                    transition: "all 0.1s ease",
                    transform: isHovered ? "scale(1.15)" : "scale(1)",
                    position: "relative",
                  }}
                  onClick={() => onSelectHead(isSelected ? null : { layer: li, head: hi })}
                  onMouseEnter={() => setHoveredHead({ layer: li, head: hi })}
                  onMouseLeave={() => setHoveredHead(null)}
                  title={`Layer ${li}, Head ${hi} — entropy: ${entropy.toFixed(2)}`}
                />
              );
            })
          )}
        </div>

        {/* Legend */}
        <div style={{ display: "flex", gap: 12, marginTop: 12, fontSize: 10, color: "var(--text-muted)" }}>
          <span><span style={{ color: "var(--green)" }}>■</span> Sharp</span>
          <span><span style={{ color: "var(--amber)" }}>■</span> Moderate</span>
          <span><span style={{ color: "var(--red)" }}>■</span> Diffuse</span>
        </div>
      </div>

      {/* Per-head attention matrix */}
      <div style={{ flex: 1, minWidth: 240 }}>
        {displayHead ? (
          <>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
              <div style={{ fontSize: 11, color: "var(--text-muted)", letterSpacing: "0.1em" }}>
                LAYER {displayHead.layer} · HEAD {displayHead.head}
              </div>
              <button
                onClick={() => onAblate(displayHead.layer, displayHead.head)}
                style={{
                  fontSize: 11, padding: "3px 10px",
                  background: isAblated(displayHead.layer, displayHead.head) ? "var(--bg-hover)" : "var(--purple-dim)",
                  color: isAblated(displayHead.layer, displayHead.head) ? "var(--green)" : "var(--purple)",
                  border: "1px solid var(--border)", borderRadius: 4,
                  cursor: "pointer", fontFamily: "var(--font-mono)",
                }}
              >
                {isAblated(displayHead.layer, displayHead.head) ? "restore" : "ablate"}
              </button>
            </div>

            {/* Matrix */}
            {attentionData[displayHead.layer]?.[displayHead.head] && (() => {
              const matrix = attentionData[displayHead.layer][displayHead.head];
              const n = Math.min(matrix.length, tokens.length);
              return (
                <div style={{ overflowX: "auto" }}>
                  {/* Column token labels */}
                  <div style={{ display: "flex", gap: 2, marginBottom: 2, paddingLeft: 48 }}>
                    {tokens.slice(0, n).map((t, i) => (
                      <div key={i} style={{
                        width: 32, fontSize: 9, color: i === selectedToken ? "var(--purple)" : "var(--text-muted)",
                        textAlign: "center", overflow: "hidden", whiteSpace: "nowrap", transform: "rotate(-30deg)",
                      }}>
                        {t.display.slice(0, 4)}
                      </div>
                    ))}
                  </div>

                  {matrix.slice(0, n).map((row, ri) => (
                    <div key={ri} style={{ display: "flex", gap: 2, marginBottom: 2, alignItems: "center" }}>
                      <div style={{
                        width: 44, fontSize: 9, textAlign: "right", paddingRight: 6,
                        color: ri === selectedToken ? "var(--purple)" : "var(--text-muted)",
                        overflow: "hidden", whiteSpace: "nowrap",
                      }}>
                        {tokens[ri]?.display?.slice(0, 5) ?? ri}
                      </div>
                      {row.slice(0, n).map((val, ci) => (
                        <div
                          key={ci}
                          title={`${tokens[ri]?.display} → ${tokens[ci]?.display}: ${(val * 100).toFixed(1)}%`}
                          style={{
                            width: 32, height: 32, borderRadius: 2,
                            background: val > 0 ? cellBg(val) : "var(--bg-hover)",
                            opacity: val > 0 ? 0.3 + val * 0.7 : 0.2,
                            border: (ri === selectedToken || ci === selectedToken) ? "1px solid rgba(139,92,246,0.4)" : "none",
                          }}
                        />
                      ))}
                    </div>
                  ))}

                  {/* Top attention pairs */}
                  <div style={{ marginTop: 14 }}>
                    <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8 }}>TOP PAIRS</div>
                    {(() => {
                      const pairs: Array<{ from: number; to: number; val: number }> = [];
                      matrix.slice(0, n).forEach((row, ri) =>
                        row.slice(0, n).forEach((val, ci) => {
                          if (val > 0.05) pairs.push({ from: ri, to: ci, val });
                        })
                      );
                      return pairs.sort((a, b) => b.val - a.val).slice(0, 5).map((p, i) => (
                        <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6, fontSize: 12 }}>
                          <span style={{ color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>
                            {tokens[p.from]?.display ?? p.from}
                          </span>
                          <span style={{ color: "var(--text-muted)" }}>→</span>
                          <span style={{ color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>
                            {tokens[p.to]?.display ?? p.to}
                          </span>
                          <div style={{ flex: 1, height: 3, background: "var(--bg-hover)", borderRadius: 2 }}>
                            <div style={{ width: `${p.val * 100}%`, height: "100%", background: "var(--cyan)", borderRadius: 2 }} />
                          </div>
                          <span style={{ fontSize: 11, color: "var(--text-muted)", minWidth: 36, textAlign: "right" }}>
                            {(p.val * 100).toFixed(0)}%
                          </span>
                        </div>
                      ));
                    })()}
                  </div>
                </div>
              );
            })()}
          </>
        ) : (
          <div style={{
            height: 200, display: "flex", alignItems: "center", justifyContent: "center",
            border: "1px dashed var(--border)", borderRadius: 8,
            color: "var(--text-muted)", fontSize: 13, flexDirection: "column", gap: 8,
          }}>
            <div>Click any head to inspect</div>
            <div style={{ fontSize: 11 }}>hover to preview</div>
          </div>
        )}
      </div>
    </div>
  );
}