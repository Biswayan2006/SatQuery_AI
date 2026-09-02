"use client";

import { useState, KeyboardEvent } from "react";
import { motion } from "framer-motion";
import { Sparkles, Search, ChevronRight } from "lucide-react";

const EXAMPLE_QUERIES = [
  { text: "Describe the land cover types", category: "captioning" },
  { text: "What changed between these images?", category: "change" },
  { text: "How many buildings are visible?", category: "vqa" },
  { text: "Identify all water bodies", category: "grounding" },
  { text: "Use SAR and optical together to analyze the scene", category: "fusion" },
  { text: "What percentage of the area is forested?", category: "vqa" },
  { text: "Detect any signs of flooding", category: "vqa" },
  { text: "Locate the roads in this image", category: "grounding" },
];

const CATEGORY_COLORS: Record<string, string> = {
  captioning: "text-sky-400 border-sky-500/30 bg-sky-900/30",
  change: "text-amber-400 border-amber-500/30 bg-amber-900/30",
  vqa: "text-violet-400 border-violet-500/30 bg-violet-900/30",
  grounding: "text-emerald-400 border-emerald-500/30 bg-emerald-900/30",
  fusion: "text-rose-400 border-rose-500/30 bg-rose-900/30",
};

interface Props {
  query: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  loading: boolean;
  hasImages: boolean;
}

export default function QueryInput({
  query,
  onChange,
  onSubmit,
  loading,
  hasImages,
}: Props) {
  const [focused, setFocused] = useState(false);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      if (canSubmit) onSubmit();
    }
  };

  const canSubmit = hasImages && query.trim().length > 0 && !loading;

  return (
    <div className="flex flex-col gap-4">
      {/* Textarea */}
      <div
        className={`relative rounded-xl border transition-all duration-300 ${
          focused
            ? "border-satellite-500/50 shadow-satellite-sm"
            : "border-slate-600/50"
        } bg-slate-900/60`}
      >
        <textarea
          value={query}
          onChange={(e) => onChange(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything about your satellite image(s)…
e.g. What land cover types are visible? What changed between the two images?"
          rows={4}
          disabled={loading}
          className="w-full bg-transparent text-slate-200 placeholder-slate-500 p-4 resize-none rounded-xl outline-none text-sm leading-relaxed disabled:opacity-60"
        />

        {/* Character count + hint */}
        <div className="flex items-center justify-between px-4 pb-3 pt-1">
          <span className="text-xs text-slate-600 font-mono">⌘↵ to submit</span>
          <span
            className={`text-xs font-mono ${
              query.length > 1800 ? "text-amber-400" : "text-slate-600"
            }`}
          >
            {query.length}/2000
          </span>
        </div>
      </div>

      {/* Submit button */}
      <motion.button
        onClick={onSubmit}
        disabled={!canSubmit}
        whileTap={canSubmit ? { scale: 0.97 } : {}}
        className={`
          relative w-full py-3 px-6 rounded-xl font-semibold text-sm
          flex items-center justify-center gap-2
          transition-all duration-300 overflow-hidden
          ${canSubmit
            ? "bg-satellite-600 hover:bg-satellite-500 text-white shadow-satellite cursor-pointer"
            : "bg-slate-800 text-slate-500 cursor-not-allowed"
          }
        `}
      >
        {loading ? (
          <>
            <div className="flex gap-1">
              {[0, 1, 2].map((i) => (
                <span
                  key={i}
                  className="w-1.5 h-1.5 rounded-full bg-white/60 loading-dot"
                  style={{ animationDelay: `${i * 0.2}s` }}
                />
              ))}
            </div>
            <span>Analysing…</span>
          </>
        ) : (
          <>
            <Sparkles className="w-4 h-4" />
            <span>Run Analysis</span>
            <ChevronRight className="w-4 h-4 ml-auto" />
          </>
        )}

        {/* Glow pulse on hover */}
        {canSubmit && (
          <div className="absolute inset-0 bg-satellite-400/10 opacity-0 hover:opacity-100 transition-opacity rounded-xl" />
        )}
      </motion.button>

      {/* Warning if no images */}
      {!hasImages && (
        <p className="text-xs text-amber-400/80 text-center">
          Upload at least one image to start analysis
        </p>
      )}

      {/* Example queries */}
      <div>
        <div className="flex items-center gap-2 mb-2.5">
          <Search className="w-3.5 h-3.5 text-slate-500" />
          <span className="text-xs text-slate-500 font-medium uppercase tracking-wider">
            Example queries
          </span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {EXAMPLE_QUERIES.map((example) => (
            <button
              key={example.text}
              onClick={() => onChange(example.text)}
              disabled={loading}
              className={`
                px-2.5 py-1 rounded-lg text-xs border transition-all duration-200
                hover:brightness-125 disabled:opacity-50 disabled:cursor-not-allowed
                ${CATEGORY_COLORS[example.category] ?? "text-slate-400 border-slate-600/30 bg-slate-800/40"}
              `}
            >
              {example.text}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
