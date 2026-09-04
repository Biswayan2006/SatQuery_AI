"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, ChevronDown, ChevronUp, RefreshCw, CheckCircle2, Clock, AlertCircle } from "lucide-react";
import axios from "axios";
import type { ModelsListResponse, ModelInfo } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const mix = (v: string, pct: number) => `color-mix(in srgb, ${v} ${pct}%, transparent)`;

// Model task → data-palette hue
const TASK_COLOR: Record<string, string> = {
  "Visual Question Answering": "var(--accent)",
  "Image Captioning":          "var(--water)",
  "Text-Guided Grounding":     "var(--veg)",
  "Change Detection":          "var(--change)",
  "SAR-Optical Fusion":        "var(--bare)",
};

export default function ModelStatus() {
  const [data,    setData]    = useState<ModelsListResponse | null>(null);
  const [open,    setOpen]    = useState(false);
  const [loading, setLoading] = useState(false);

  const fetch = async () => {
    setLoading(true);
    try {
      const res = await axios.get<ModelsListResponse>(`${API_BASE}/api/models`, { timeout: 5000 });
      setData(res.data);
    } catch { /* silent */ } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetch();
    const id = setInterval(fetch, 20_000);
    return () => clearInterval(id);
  }, []);

  if (!data) return null;

  const loaded = data.total_loaded;
  const total  = data.models.length;
  const pct    = total > 0 ? Math.round((loaded / total) * 100) : 0;
  const allReady = loaded === total;
  const barColor = allReady ? "var(--veg)" : "var(--sar)";

  return (
    <div
      className="overflow-hidden rounded-lg"
      style={{ background: "var(--bg-inset)", border: "1px solid var(--border)" }}
    >
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center justify-between px-3 py-2 transition-colors no-tap"
        style={{ background: "transparent" }}
        onMouseEnter={(e) => (e.currentTarget.style.background = "var(--accent-soft)")}
        onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
      >
        <div className="flex items-center gap-2">
          <Cpu
            className="w-3.5 h-3.5 flex-shrink-0"
            style={{ color: barColor }}
          />
          <span className="text-xs font-semibold" style={{ color: "var(--text-secondary)" }}>Model Registry</span>
          {/* Progress bar */}
          <div
            className="w-20 h-1.5 rounded-full overflow-hidden"
            style={{ background: "var(--border-strong)" }}
          >
            <motion.div
              className="h-full rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${pct}%` }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              style={{ background: barColor }}
            />
          </div>
          <span
            className="text-[10px] font-mono"
            style={{ color: barColor }}
          >
            {loaded}/{total}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={(e) => { e.stopPropagation(); fetch(); }}
            className="p-1 rounded transition-colors no-tap"
            style={{ color: "var(--text-muted)" }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text-secondary)")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-muted)")}
          >
            <RefreshCw className={`w-3 h-3 ${loading ? "animate-spin" : ""}`} />
          </button>
          {open
            ? <ChevronUp className="w-3.5 h-3.5" style={{ color: "var(--text-muted)" }} />
            : <ChevronDown className="w-3.5 h-3.5" style={{ color: "var(--text-muted)" }} />
          }
        </div>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div
              style={{ borderTop: "1px solid var(--border)" }}
            >
              {data.models.map((m: ModelInfo) => (
                <div
                  key={m.name}
                  className="flex items-center justify-between px-3 py-2"
                  style={{
                    borderBottom: "1px solid var(--border)",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "var(--accent-soft)")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    {m.loaded
                      ? <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--veg)" }} />
                      : <Clock        className="w-3.5 h-3.5 flex-shrink-0 animate-pulse" style={{ color: mix("var(--sar)", 60) }} />
                    }
                    <div className="min-w-0">
                      <p
                        className="text-xs font-medium truncate"
                        style={{ color: TASK_COLOR[m.task] ?? "var(--text-muted)" }}
                      >
                        {m.name}
                      </p>
                      <p
                        className="text-[10px] font-mono truncate"
                        style={{ color: "var(--text-faint)", maxWidth: "160px" }}
                      >
                        {m.model_id}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <span
                      className="text-[10px] px-1.5 py-0.5 rounded font-medium"
                      style={{
                        background: m.loaded ? mix("var(--veg)", 12) : mix("var(--sar)", 12),
                        border: `1px solid ${m.loaded ? mix("var(--veg)", 28) : mix("var(--sar)", 28)}`,
                        color: m.loaded ? "var(--veg)" : "var(--sar)",
                      }}
                    >
                      {m.loaded ? "ready" : "loading"}
                    </span>
                    {m.device && (
                      <span className="text-[10px] font-mono" style={{ color: "var(--text-faint)" }}>
                        {m.device}
                      </span>
                    )}
                  </div>
                </div>
              ))}
              {/* Device info */}
              <div className="px-3 py-1.5 flex items-center gap-2">
                <AlertCircle className="w-3 h-3" style={{ color: "var(--text-faint)" }} />
                <span className="text-[10px]" style={{ color: "var(--text-faint)" }}>
                  Running on <span className="font-mono" style={{ color: "var(--text-muted)" }}>{data.device.toUpperCase()}</span>
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
