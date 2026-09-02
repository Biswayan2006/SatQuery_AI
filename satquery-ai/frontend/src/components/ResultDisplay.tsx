"use client";

import { motion } from "framer-motion";
import { CheckCircle, Zap, Map, Eye, Grid3X3, AlertCircle } from "lucide-react";
import type { AnalysisResponse } from "@/types";

const TASK_LABELS: Record<string, { label: string; color: string }> = {
  SINGLE_VQA: { label: "Visual Q&A", color: "text-violet-400 bg-violet-900/40 border-violet-500/30" },
  CAPTIONING: { label: "Scene Caption", color: "text-sky-400 bg-sky-900/40 border-sky-500/30" },
  GROUNDING: { label: "Object Grounding", color: "text-emerald-400 bg-emerald-900/40 border-emerald-500/30" },
  CHANGE_VQA: { label: "Change Q&A", color: "text-amber-400 bg-amber-900/40 border-amber-500/30" },
  CHANGE_DESCRIPTION: { label: "Change Detection", color: "text-orange-400 bg-orange-900/40 border-orange-500/30" },
  SAR_OPTICAL_FUSION: { label: "SAR-Optical Fusion", color: "text-rose-400 bg-rose-900/40 border-rose-500/30" },
};

interface Props {
  result: AnalysisResponse;
}

export default function ResultDisplay({ result }: Props) {
  const taskMeta = TASK_LABELS[result.task] ?? { label: result.task, color: "text-slate-400 bg-slate-800/40 border-slate-600/30" };

  const confPct = Math.round(result.confidence * 100);
  const confColor =
    result.confidence >= 0.7 ? "#22c55e" : result.confidence >= 0.4 ? "#eab308" : "#ef4444";

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className="flex flex-col gap-5"
    >
      {/* Task + Confidence header */}
      <div className="glass-card p-4 flex flex-col gap-3">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <span
            className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${taskMeta.color}`}
          >
            <Zap className="w-3 h-3" />
            {taskMeta.label}
          </span>
          <div className="flex items-center gap-2">
            <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-xs text-slate-400">
              Confidence:{" "}
              <span className="font-semibold" style={{ color: confColor }}>
                {confPct}%
              </span>
            </span>
          </div>
        </div>

        {/* Confidence bar */}
        <div className="confidence-bar">
          <div
            className="confidence-fill"
            style={{ width: `${confPct}%`, background: confColor }}
          />
        </div>
      </div>

      {/* Answer */}
      <div className="glass-card p-4">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-1.5 h-5 bg-satellite-500 rounded-full" />
          <h3 className="text-sm font-semibold text-slate-300">Answer</h3>
        </div>
        <p className="text-slate-200 text-sm leading-relaxed whitespace-pre-wrap">
          {result.answer}
        </p>

        {result.change_percentage !== null && result.change_percentage !== undefined && (
          <div className="mt-3 flex items-center gap-2 p-2 rounded-lg bg-amber-900/20 border border-amber-700/30">
            <AlertCircle className="w-4 h-4 text-amber-400 flex-shrink-0" />
            <span className="text-sm text-amber-300">
              Changed area: <strong>{result.change_percentage.toFixed(1)}%</strong> of scene
            </span>
          </div>
        )}
      </div>

      {/* Visual Evidence */}
      {result.visual_evidence && (
        <ImagePanel
          title="Visual Evidence"
          b64={result.visual_evidence}
          icon={<Eye className="w-4 h-4 text-sky-400" />}
        />
      )}

      {/* Change Map */}
      {result.change_map && (
        <ImagePanel
          title="Change Map"
          b64={result.change_map}
          icon={<Map className="w-4 h-4 text-amber-400" />}
          caption="Red = changed  ·  Green = unchanged"
        />
      )}

      {/* Fusion Map */}
      {result.fusion_map && (
        <ImagePanel
          title="SAR-Optical Fusion"
          b64={result.fusion_map}
          icon={<Grid3X3 className="w-4 h-4 text-rose-400" />}
          caption="Optical  ·  SAR  ·  Fusion feature map"
        />
      )}

      {/* Grounding Boxes */}
      {result.grounding_boxes && result.grounding_boxes.length > 0 && (
        <div className="glass-card p-4">
          <div className="flex items-center gap-2 mb-3">
            <Grid3X3 className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-semibold text-slate-300">
              Detected Objects ({result.grounding_boxes.length})
            </h3>
          </div>
          <div className="flex flex-col gap-1.5 max-h-48 overflow-y-auto">
            {result.grounding_boxes.map((box, i) => (
              <div
                key={i}
                className="flex items-center justify-between text-xs p-2 rounded-lg bg-slate-800/60 border border-slate-700/40"
              >
                <div className="flex items-center gap-2">
                  <span className="w-5 h-5 rounded-full bg-emerald-900/60 border border-emerald-500/40 flex items-center justify-center text-emerald-400 font-bold text-[10px]">
                    {i + 1}
                  </span>
                  <span className="text-slate-300 font-medium">{box.label}</span>
                </div>
                <div className="flex items-center gap-3 text-slate-500 font-mono">
                  <span className="text-emerald-400 font-semibold">
                    {(box.score * 100).toFixed(0)}%
                  </span>
                  <span>
                    [{box.x1.toFixed(2)}, {box.y1.toFixed(2)}, {box.x2.toFixed(2)},{" "}
                    {box.y2.toFixed(2)}]
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
}

// ── Image Panel ───────────────────────────────────────────────────────────────

function ImagePanel({
  title,
  b64,
  icon,
  caption,
}: {
  title: string;
  b64: string;
  icon: React.ReactNode;
  caption?: string;
}) {
  return (
    <div className="glass-card p-4">
      <div className="flex items-center gap-2 mb-3">
        {icon}
        <h3 className="text-sm font-semibold text-slate-300">{title}</h3>
      </div>
      <div className="rounded-lg overflow-hidden border border-slate-700/40 bg-slate-900/40">
        <img
          src={`data:image/png;base64,${b64}`}
          alt={title}
          className="w-full object-contain max-h-64"
        />
      </div>
      {caption && (
        <p className="text-xs text-slate-500 text-center mt-2 font-mono">{caption}</p>
      )}
    </div>
  );
}
