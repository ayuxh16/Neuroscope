"use client";

import { useState } from "react";

type Props = {
  initialPrompt: string;
  loading: boolean;
  onSubmit: (prompt: string, maxNewTokens: number) => void;
};

export default function TokenInput({ initialPrompt, loading, onSubmit }: Props) {
  const [text, setText] = useState(initialPrompt);
  const [n, setN] = useState(10);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
      <label className="mb-2 block text-sm text-slate-400">Prompt</label>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        maxLength={500}
        rows={2}
        className="w-full rounded-lg border border-slate-700 bg-slate-950 p-3 font-mono text-sm outline-none focus:border-sky-500"
      />
      <div className="mt-3 flex items-center gap-4">
        <label className="text-sm text-slate-400">New tokens: {n}</label>
        <input
          type="range"
          min={1}
          max={30}
          value={n}
          onChange={(e) => setN(Number(e.target.value))}
        />
        <button
          disabled={loading || !text.trim()}
          onClick={() => onSubmit(text.trim(), n)}
          className="ml-auto rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium hover:bg-sky-500 disabled:opacity-50"
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </div>
    </div>
  );
}