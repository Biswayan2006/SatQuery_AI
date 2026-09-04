"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { History, ChevronDown, ChevronUp, Trash2, Clock } from "lucide-react";
import type { AnalysisResponse } from "@/types";

const STORAGE_KEY = "satquery_history";
const MAX_ITEMS   = 10;

export interface HistoryEntry {
  id: string;
  timestamp: number;
  query: string;
  task: string;
  answer: string;
  confidence: number;
  session_id: string;
}

export function saveToHistory(result: AnalysisResponse, query: string) {
  try {
    const prev: HistoryEntry[] = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
    const entry: HistoryEntry = {
      id: crypto.randomUUID(),
      timestamp: Date.now(),
      query,
      task: result.task,
      answer: result.answer.slice(0, 200),
      confidence: result.confidence,
      session_id: result.session_id,
    };
    const next = [entry, ...prev].slice(0, MAX_ITEMS);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch { /* storage unavailable */ }
}

const TASK_COLOR: Record<string, string> = {
  SINGLE_VQA:         "text-violet-400 bg-violet-950/50 border-violet-700/30",
  CAPTIONING:         "text-sky-400 bg-sky-950/50 border-sky-700/30",
  GROUNDING:          "text-emerald-400 bg-emerald-950/50 border-emerald-700/30",
  CHANGE_VQA:         "text-amber-400 bg-amber-950/50 border-amber-700/30",
  CHANGE_DESCRIPTION: "text-orange-400 bg-orange-950/50 border-orange-700/30",
  SAR_OPTICAL_FUSION: "text-rose-400 bg-rose-950/50 border-rose-700/30",
};

interface Props {
  onSelect: (query: string) => void;
}

export default function SessionHistory({ onSelect }: Props) {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [open, setOpen]       = useState(false);

  useEffect(() => {
    try {
      setEntries(JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]"));
    } catch { /* */ }
  }, [open]);

  const clear = () => {
    localStorage.removeItem(STORAGE_KEY);
    setEntries([]);
  };

  const remove = (id: string) => {
    const next = entries.filter(e => e.id !== id);
    setEntries(next);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  };

  if (entries.length === 0 && !open) return null;

  return (
    <div className="glass overflow-hidden">
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-white/[0.02] transition-colors no-tap"
      >
        <div className="flex items-center gap-2.5">
          <History className="w-3.5 h-3.5 text-slate-500" />
          <span className="text-xs font-semibold text-slate-400">Recent Sessions</span>
          <span className="text-[10px] text-slate-600 bg-space-800/60 border border-white/[0.04] px-1.5 py-0.5 rounded-full">
            {entries.length}
          </span>
        </div>
        {open ? <ChevronUp className="w-3.5 h-3.5 text-slate-600" /> : <ChevronDown className="w-3.5 h-3.5 text-slate-600" />}
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <div className="border-t border-white/[0.04] max-h-72 overflow-y-auto">
              {entries.length === 0 ? (
                <p className="text-xs text-slate-600 text-center py-6">No history yet</p>
              ) : (
                <>
                  {entries.map((e) => (
                    <div key={e.id} className="group flex items-start gap-2 px-4 py-3 hover:bg-white/[0.02] border-b border-white/[0.03]">
                      <button
                        onClick={() => onSelect(e.query)}
                        className="flex-1 text-left min-w-0 no-tap"
                      >
                        <div className="flex items-center gap-1.5 mb-1">
                          <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded border ${TASK_COLOR[e.task] ?? "text-slate-400 bg-space-800 border-white/[0.04]"}`}>
                            {e.task.replace("_", " ")}
                          </span>
                          <span className="text-[9px] text-slate-700 font-mono flex items-center gap-0.5">
                            <Clock className="w-2.5 h-2.5" />
                            {formatTime(e.timestamp)}
                          </span>
                          <span className="text-[9px] text-slate-600 font-mono ml-auto">
                            {Math.round(e.confidence * 100)}%
                          </span>
                        </div>
                        <p className="text-xs text-slate-300 truncate font-medium">{e.query}</p>
                        <p className="text-[10px] text-slate-600 truncate mt-0.5 leading-relaxed">{e.answer}</p>
                      </button>
                      <button
                        onClick={() => remove(e.id)}
                        className="flex-shrink-0 w-5 h-5 rounded flex items-center justify-center text-slate-700 hover:text-red-400 hover:bg-red-950/40 transition-colors opacity-0 group-hover:opacity-100 no-tap"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  ))}
                  <div className="px-4 py-2">
                    <button
                      onClick={clear}
                      className="text-[10px] text-slate-600 hover:text-red-400 transition-colors no-tap"
                    >
                      Clear all history
                    </button>
                  </div>
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function formatTime(ts: number): string {
  const diff = Date.now() - ts;
  if (diff < 60_000)      return "just now";
  if (diff < 3_600_000)   return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000)  return `${Math.floor(diff / 3_600_000)}h ago`;
  return new Date(ts).toLocaleDateString();
}
