"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Terminal, ChevronDown, ChevronUp, Clock, Cpu, CheckCheck } from "lucide-react";
import type { ExecutionSummary } from "@/types";

interface Props {
  summary: ExecutionSummary;
  defaultOpen?: boolean;
}

export default function ExecutionTrace({ summary, defaultOpen = false }: Props) {
  const [open, setOpen] = useState(defaultOpen);

  const lines = buildTraceLines(summary);

  return (
    <div className="glass-card overflow-hidden border-slate-700/50">
      {/* Header */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-slate-800/40 transition-colors"
      >
        <div className="flex items-center gap-2.5">
          <Terminal className="w-4 h-4 text-emerald-400" />
          <span className="text-sm font-semibold text-slate-300">Execution Trace</span>
          <span className="text-xs text-slate-500 font-mono">
            {summary.processing_time_ms.toFixed(0)} ms
          </span>
        </div>
        {open ? (
          <ChevronUp className="w-4 h-4 text-slate-500" />
        ) : (
          <ChevronDown className="w-4 h-4 text-slate-500" />
        )}
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: "easeInOut" }}
          >
            {/* Terminal body */}
            <div className="bg-[#060d1a] border-t border-slate-800/60 p-4 font-mono text-xs">
              {/* Metrics row */}
              <div className="flex flex-wrap gap-4 mb-4 pb-3 border-b border-slate-800">
                <MetricChip
                  icon={<Cpu className="w-3 h-3" />}
                  label="Task"
                  value={summary.selected_task}
                  color="text-violet-400"
                />
                <MetricChip
                  icon={<CheckCheck className="w-3 h-3" />}
                  label="Confidence"
                  value={`${(summary.task_confidence * 100).toFixed(0)}%`}
                  color="text-emerald-400"
                />
                <MetricChip
                  icon={<Clock className="w-3 h-3" />}
                  label="Time"
                  value={`${summary.processing_time_ms.toFixed(1)} ms`}
                  color="text-satellite-400"
                />
              </div>

              {/* Log lines */}
              <div className="space-y-1.5">
                {lines.map((line, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.05, duration: 0.2 }}
                    className={`terminal-line ${line.type}`}
                  >
                    <span className="text-slate-600 mr-2 select-none">{String(i + 1).padStart(2, "0")}</span>
                    <span className="text-slate-600 mr-2">{getPrompt(line.type)}</span>
                    {line.text}
                  </motion.div>
                ))}
              </div>

              {/* Parameters */}
              {Object.keys(summary.parameters).length > 0 && (
                <div className="mt-4 pt-3 border-t border-slate-800">
                  <p className="text-slate-500 mb-2">─── Parameters</p>
                  {Object.entries(summary.parameters).map(([k, v]) => (
                    <div key={k} className="terminal-line info">
                      <span className="text-slate-600 mr-2 select-none">  </span>
                      <span className="text-slate-400">{k}</span>
                      <span className="text-slate-600"> = </span>
                      <span className="text-satellite-400">{String(v)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

type TraceLine = { text: string; type: "info" | "success" | "warn" };

function buildTraceLines(summary: ExecutionSummary): TraceLine[] {
  const lines: TraceLine[] = [];

  lines.push({ type: "info", text: `[CLASSIFIER] task=${summary.selected_task}  confidence=${(summary.task_confidence * 100).toFixed(0)}%` });

  summary.steps.forEach((step, i) => {
    lines.push({
      type: i < summary.steps.length - 1 ? "info" : "success",
      text: `[STEP ${i + 1}/${summary.steps.length}] ${step.replace(/_/g, " ")}`,
    });
  });

  summary.models_used.forEach((model) => {
    lines.push({ type: "success", text: `[MODEL] ${model} — inference complete` });
  });

  lines.push({
    type: "success",
    text: `[DONE] total=${summary.processing_time_ms.toFixed(1)} ms`,
  });

  return lines;
}

function getPrompt(type: TraceLine["type"]): string {
  switch (type) {
    case "success": return "✓";
    case "warn": return "!";
    default: return "›";
  }
}

function MetricChip({
  icon,
  label,
  value,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-slate-500">{icon}</span>
      <span className="text-slate-500">{label}:</span>
      <span className={`font-semibold ${color}`}>{value}</span>
    </div>
  );
}
