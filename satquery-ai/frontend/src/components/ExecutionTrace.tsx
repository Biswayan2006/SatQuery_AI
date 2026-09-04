"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Terminal, ChevronDown, ChevronUp, Clock, Cpu, CheckCheck,
  Layers, Hexagon,
} from "lucide-react";
import type { ExecutionSummary } from "@/types";

interface Props {
  summary: ExecutionSummary;
  defaultOpen?: boolean;
}

export default function ExecutionTrace({ summary, defaultOpen = false }: Props) {
  const [open, setOpen] = useState(defaultOpen);

  const lines = buildTraceLines(summary);

  return (
    <div
      className="overflow-hidden rounded-lg"
      style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
    >
      {/* Header */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3.5 py-2.5 transition-colors no-tap"
        style={{
          background: open ? "var(--accent-soft)" : "transparent",
          borderBottom: open ? "1px solid var(--border)" : "none",
        }}
        onMouseEnter={(e) => {
          if (!open) e.currentTarget.style.background = "var(--bg-raised)";
        }}
        onMouseLeave={(e) => {
          if (!open) e.currentTarget.style.background = "transparent";
        }}
      >
        <div className="flex items-center gap-2.5 min-w-0 flex-1">
          <Terminal className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--accent)" }} />
          <span className="text-xs font-semibold truncate" style={{ color: "var(--text-secondary)" }}>
            Audit trail — execution log
          </span>
          <span
            className="text-[10px] font-mono flex-shrink-0 px-1.5 py-0.5 rounded"
            style={{
              background: "var(--bg-inset)",
              color: "var(--text-muted)",
            }}
          >
            {(summary.processing_time_ms / 1000).toFixed(2)}s
          </span>
        </div>
        {open ? (
          <ChevronUp className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--text-muted)" }} />
        ) : (
          <ChevronDown className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--text-muted)" }} />
        )}
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            {/* Terminal body */}
            <div
              className="p-3.5 font-mono"
              style={{
                background: "var(--bg-inset)",
                fontSize: "11px",
                lineHeight: 1.6,
              }}
            >
              {/* Metrics row */}
              <div
                className="flex flex-wrap gap-3 mb-3 pb-3"
                style={{ borderBottom: "1px dashed var(--border-strong)" }}
              >
                <MetricChip
                  icon={<Hexagon className="w-3 h-3" style={{ color: "var(--accent)" }} />}
                  label="TASK"
                  value={summary.selected_task.replace(/_/g, " ")}
                  color="var(--accent)"
                />
                <MetricChip
                  icon={<CheckCheck className="w-3 h-3" style={{ color: "var(--veg)" }} />}
                  label="CONF"
                  value={`${(summary.task_confidence * 100).toFixed(0)}%`}
                  color="var(--veg)"
                />
                <MetricChip
                  icon={<Clock className="w-3 h-3" style={{ color: "var(--water)" }} />}
                  label="TOTAL"
                  value={`${summary.processing_time_ms.toFixed(0)} ms`}
                  color="var(--water)"
                />
                <MetricChip
                  icon={<Cpu className="w-3 h-3" style={{ color: "var(--sar)" }} />}
                  label="MODELS"
                  value={String(summary.models_used.length)}
                  color="var(--sar)"
                />
                <MetricChip
                  icon={<Layers className="w-3 h-3" style={{ color: "var(--bare)" }} />}
                  label="STEPS"
                  value={String(summary.steps.length)}
                  color="var(--bare)"
                />
              </div>

              {/* Log lines */}
              <div className="space-y-1">
                {lines.map((line, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.04, duration: 0.18 }}
                    className="flex items-start gap-2"
                  >
                    <span style={{ color: "var(--text-faint)", userSelect: "none", flexShrink: 0 }}>
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <span style={{ color: line.color, flexShrink: 0 }}>
                      {getPrompt(line.type)}
                    </span>
                    <span
                      className="whitespace-pre-wrap break-all"
                      style={{ color: line.textColor }}
                    >
                      {line.text}
                    </span>
                  </motion.div>
                ))}
              </div>

              {/* Parameters */}
              {Object.keys(summary.parameters).length > 0 && (
                <div
                  className="mt-3 pt-3"
                  style={{ borderTop: "1px dashed var(--border-strong)" }}
                >
                  <p className="mb-1.5" style={{ color: "var(--text-faint)" }}>
                    ─── PARAMETERS
                  </p>
                  <div className="space-y-0.5 pl-3">
                    {Object.entries(summary.parameters).map(([k, v]) => (
                      <div key={k} className="flex items-start gap-2">
                        <span style={{ color: "var(--text-muted)" }}>{k}</span>
                        <span style={{ color: "var(--text-faint)" }}>=</span>
                        <span style={{ color: "var(--water)", wordBreak: "break-all" }}>
                          {String(v)}
                        </span>
                      </div>
                    ))}
                  </div>
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

type TraceType = "info" | "success" | "warn";
type TraceLine = { text: string; type: TraceType; color: string; textColor: string };

function buildTraceLines(summary: ExecutionSummary): TraceLine[] {
  const lines: TraceLine[] = [];

  lines.push({
    type: "info",
    text: `[CLASSIFIER] task=${summary.selected_task}  confidence=${(summary.task_confidence * 100).toFixed(0)}%`,
    color: "var(--water)",
    textColor: "var(--text-muted)",
  });

  summary.steps.forEach((step, i) => {
    const isLast = i === summary.steps.length - 1;
    lines.push({
      type: isLast ? "success" : "info",
      text: `[STEP ${i + 1}/${summary.steps.length}] ${step.replace(/_/g, " ")}`,
      color: isLast ? "var(--veg)" : "var(--water)",
      textColor: isLast ? "var(--text-secondary)" : "var(--text-muted)",
    });
  });

  summary.models_used.forEach((model) => {
    lines.push({
      type: "success",
      text: `[MODEL] ${model} — inference complete`,
      color: "var(--veg)",
      textColor: "var(--text-secondary)",
    });
  });

  lines.push({
    type: "success",
    text: `[DONE] total=${summary.processing_time_ms.toFixed(1)} ms · steps=${summary.steps.length} · models=${summary.models_used.length}`,
    color: "var(--veg)",
    textColor: "var(--text-primary)",
  });

  return lines;
}

function getPrompt(type: TraceType): string {
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
    <div
      className="flex items-center gap-1.5 px-2 py-1 rounded-md flex-shrink-0"
      style={{
        background: "var(--bg-raised)",
        border: "1px solid var(--border)",
      }}
    >
      {icon}
      <span style={{ color: "var(--text-faint)", fontWeight: 700, letterSpacing: "0.05em" }}>
        {label}
      </span>
      <span style={{ color, fontWeight: 700 }}>{value}</span>
    </div>
  );
}
