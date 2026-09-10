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
  captioning: "text-water border-water/30 bg-water-soft",
  change: "text-change border-change/30 bg-change-soft",
  vqa: "text-accent border-accent/30 bg-accent-soft",
  grounding: "text-veg border-veg/30 bg-veg-soft",
  fusion: "text-sar border-sar/30 bg-sar-soft",
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
            ? "border-accent-border shadow-sm"
            : "border-line"
        } bg-surface`}
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
          className="w-full bg-transparent text-ink placeholder:text-ink-faint p-4 resize-none rounded-xl outline-none text-sm leading-relaxed disabled:opacity-60"
        />

        {/* Character count + hint */}
        <div className="flex items-center justify-between px-4 pb-3 pt-1">
          <span className="text-xs text-ink-faint font-mono">⌘↵ to submit</span>
          <span
            className={`text-xs font-mono ${
              query.length > 1800 ? "text-warning" : "text-ink-faint"
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
          relative w-full py-3 px-6 rounded-lg font-semibold text-sm
          flex items-center justify-center gap-2
          transition-all duration-300 overflow-hidden no-tap
          ${canSubmit
            ? "bg-accent hover:bg-accent-hover text-accent-contrast shadow-sm cursor-pointer"
            : "bg-raised text-ink-faint cursor-not-allowed"
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
      </motion.button>

      {/* Warning if no images */}
      {!hasImages && (
        <p className="text-xs text-warning text-center">
          Upload at least one image to start analysis
        </p>
      )}

      {/* Example queries */}
      <div>
        <div className="flex items-center gap-2 mb-2.5">
          <Search className="w-3.5 h-3.5 text-ink-muted" />
          <span className="text-xs text-ink-muted font-medium">
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
                hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed no-tap
                ${CATEGORY_COLORS[example.category] ?? "text-ink-muted border-line bg-raised"}
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
