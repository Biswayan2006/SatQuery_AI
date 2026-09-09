"use client";

import { motion } from "framer-motion";
import {
  Zap, Map, Eye, Grid3X3, AlertCircle, Sparkles,
  Target, Activity, BarChart3, ShieldCheck,
} from "lucide-react";
import type { AnalysisResponse } from "@/types";

// Task: data-palette hue. Six visually distinct observation hues.
const mix = (v: string, pct: number) => `color-mix(in srgb, ${v} ${pct}%, transparent)`;
const TASK_LABELS: Record<string, { label: string; color: string; bg: string; border: string }> = {
  SINGLE_VQA:         { label: "Visual Q&A",         color: "var(--accent)", bg: mix("var(--accent)", 13), border: mix("var(--accent)", 32) },
  CAPTIONING:         { label: "Scene Caption",      color: "var(--water)",  bg: mix("var(--water)", 13),  border: mix("var(--water)", 32) },
  GROUNDING:          { label: "Object Grounding",   color: "var(--veg)",    bg: mix("var(--veg)", 13),    border: mix("var(--veg)", 32) },
  CHANGE_VQA:         { label: "Change Q&A",         color: "var(--change)", bg: mix("var(--change)", 13), border: mix("var(--change)", 32) },
  CHANGE_DESCRIPTION: { label: "Change Detection",   color: "var(--change)", bg: mix("var(--change)", 13), border: mix("var(--change)", 32) },
  SAR_OPTICAL_FUSION: { label: "SAR-Optical Fusion", color: "var(--sar)",    bg: mix("var(--sar)", 13),    border: mix("var(--sar)", 32) },
};

interface Props {
  result: AnalysisResponse;
}

export default function ResultDisplay({ result }: Props) {
  const taskMeta = TASK_LABELS[result.task] ?? {
    label: result.task,
    color: "var(--text-muted)",
    bg: mix("var(--text-muted)", 13),
    border: mix("var(--text-muted)", 32),
  };

  const confPct = Math.round(result.confidence * 100);
  const confColor =
    result.confidence >= 0.7 ? "var(--veg)" : result.confidence >= 0.4 ? "var(--warning)" : "var(--danger)";

  const hasGrounding  = result.grounding_boxes && result.grounding_boxes.length > 0;
  const hasChange     = result.change_percentage != null;
  const hasVizEvidence = !!result.visual_evidence;
  const hasChangeMap  = !!result.change_map;
  const hasFusionMap  = !!result.fusion_map;
  const hasAnyVisual  = hasVizEvidence || hasChangeMap || hasFusionMap;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="flex flex-col gap-3"
    >
      {/* ── Evidence header strip ──────────────────────────────────── */}
      <SectionHeader
        tag="Evidence summary"
        pill={{
          label: taskMeta.label,
          color: taskMeta.color,
          bg: taskMeta.bg,
          border: taskMeta.border,
          icon: <Zap className="w-3 h-3" />,
        }}
      />

      <EvidenceRow>
        {/* Confidence metric */}
        <EvidenceMetric
          icon={<BarChart3 className="w-3.5 h-3.5" />}
          label="Confidence"
          value={`${confPct}%`}
          valueColor={confColor}
          detail={
            <div className="h-1 w-full rounded-full" style={{ background: "var(--border-strong)" }}>
              <div
                className="h-full rounded-full"
                style={{ width: `${confPct}%`, background: confColor }}
              />
            </div>
          }
        />

        {/* Detected Changes */}
        {hasChange && (
          <EvidenceMetric
            icon={<Activity className="w-3.5 h-3.5" />}
            label="Detected changes"
            value={`${result.change_percentage!.toFixed(1)}%`}
            valueColor={"var(--change)"}
            detail={
              <div
                className="text-[10px] px-1.5 py-0.5 rounded"
                style={{
                  background: mix("var(--change)", 12),
                  border: `1px solid ${mix("var(--change)", 25)}`,
                  color: "var(--change)",
                }}
              >
                {result.change_percentage! < 5 ? "Minimal" : result.change_percentage! < 20 ? "Moderate" : "Significant"}
              </div>
            }
          />
        )}

        {/* Key Objects */}
        {hasGrounding && (
          <EvidenceMetric
            icon={<Target className="w-3.5 h-3.5" />}
            label="Key objects"
            value={`${result.grounding_boxes!.length}`}
            valueColor="var(--veg)"
            detail={
              <div className="flex flex-wrap gap-1">
                {result.grounding_boxes!.slice(0, 3).map((b, i) => (
                  <span
                    key={i}
                    className="text-[10px] px-1.5 py-0.5 rounded"
                    style={{
                      background: mix("var(--veg)", 12),
                      border: `1px solid ${mix("var(--veg)", 25)}`,
                      color: "var(--veg)",
                    }}
                  >
                    {b.label}
                  </span>
                ))}
                {result.grounding_boxes!.length > 3 && (
                  <span className="text-[10px]" style={{ color: "var(--text-faint)" }}>
                    +{result.grounding_boxes!.length - 3}
                  </span>
                )}
              </div>
            }
          />
        )}
      </EvidenceRow>

      {/* ── Evidence-Grounded Response ─────────────────────────────── */}
      <EvidenceSection
        icon={<Sparkles className="w-3.5 h-3.5" style={{ color: "var(--accent)" }} />}
        label="Evidence-grounded response"
      >
        <p
          className="text-[12.5px] leading-relaxed whitespace-pre-wrap"
          style={{ color: "var(--text-secondary)" }}
        >
          {result.answer}
        </p>

        {hasChange && (
          <div
            className="mt-2.5 flex items-center gap-2 p-2 rounded-lg"
            style={{
              background: mix("var(--change)", 8),
              border: `1px solid ${mix("var(--change)", 22)}`,
            }}
          >
            <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--change)" }} />
            <span className="text-[11px]" style={{ color: "var(--change)" }}>
              Changed area occupies <strong>{result.change_percentage!.toFixed(1)}%</strong> of the observed scene
            </span>
          </div>
        )}
      </EvidenceSection>

      {/* ── Key Objects (extended list) ────────────────────────────── */}
      {hasGrounding && result.grounding_boxes!.length > 0 && (
        <EvidenceSection
          icon={<Grid3X3 className="w-3.5 h-3.5" style={{ color: "var(--veg)" }} />}
          label="Key objects detected"
          tag={`${result.grounding_boxes!.length} targets`}
          tagColor="var(--veg)"
        >
          <div className="flex flex-col gap-1 max-h-36 overflow-y-auto pr-1">
            {result.grounding_boxes!.map((box, i) => (
              <div
                key={i}
                className="flex items-center justify-between text-[11px] p-1.5 rounded-md"
                style={{
                  background: "var(--bg-raised)",
                  border: "1px solid var(--border)",
                }}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span
                    className="w-4 h-4 rounded-full flex items-center justify-center font-bold flex-shrink-0"
                    style={{
                      background: mix("var(--veg)", 15),
                      border: `1px solid ${mix("var(--veg)", 30)}`,
                      color: "var(--veg)",
                      fontSize: "9px",
                    }}
                  >
                    {i + 1}
                  </span>
                  <span
                    className="font-medium truncate"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {box.label}
                  </span>
                </div>
                <div className="flex items-center gap-2 font-mono flex-shrink-0" style={{ color: "var(--text-faint)", fontSize: "10px" }}>
                  <span style={{ color: "var(--veg)", fontWeight: 600 }}>
                    {(box.score * 100).toFixed(0)}%
                  </span>
                  <span className="hidden sm:inline">
                    [{box.x1.toFixed(1)}, {box.y1.toFixed(1)} to {box.x2.toFixed(1)}, {box.y2.toFixed(1)}]
                  </span>
                </div>
              </div>
            ))}
          </div>
        </EvidenceSection>
      )}

      {/* ── Visual Panels ──────────────────────────────────────────── */}
      {hasAnyVisual && (
        <>
          <SectionHeader tag="Visual evidence" />

          {hasChangeMap && (
            <MiniVisual
              title="Change Map"
              icon={<Map className="w-3.5 h-3.5" style={{ color: "var(--change)" }} />}
              b64={result.change_map!}
              caption="Red = changed, green = unchanged"
            />
          )}

          {hasVizEvidence && (
            <MiniVisual
              title="Visual Evidence"
              icon={<Eye className="w-3.5 h-3.5" style={{ color: "var(--water)" }} />}
              b64={result.visual_evidence!}
            />
          )}

          {hasFusionMap && (
            <MiniVisual
              title="SAR-Optical Fusion"
              icon={<Grid3X3 className="w-3.5 h-3.5" style={{ color: "var(--bare)" }} />}
              b64={result.fusion_map!}
              caption="Optical, SAR & fusion feature map"
            />
          )}
        </>
      )}

      {/* ── Provenance footer ──────────────────────────────────────── */}
      <div
        className="flex items-center justify-between px-3 py-2 rounded-lg"
        style={{
          background: "var(--accent-soft)",
          border: "1px solid var(--accent-border)",
        }}
      >
        <div className="flex items-center gap-1.5 min-w-0">
          <ShieldCheck className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--accent)" }} />
          <span
            className="text-[11px] font-medium truncate"
            style={{ color: "var(--accent)" }}
          >
            Evidence verified against source imagery
          </span>
        </div>
        <span className="text-[10px] font-mono flex-shrink-0" style={{ color: "var(--text-faint)" }}>
          session {result.session_id.slice(0, 8)}
        </span>
      </div>
    </motion.div>
  );
}

// ── Shared layout primitives ───────────────────────────────────────────────

function SectionHeader({
  tag,
  pill,
}: {
  tag: string;
  pill?: { label: string; color: string; bg: string; border: string; icon: React.ReactNode };
}) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        <div className="h-0.5 w-3 rounded-full" style={{ background: "var(--accent)" }} />
        <span
          className="text-[11px] font-semibold"
          style={{ color: "var(--text-muted)" }}
        >
          {tag}
        </span>
      </div>
      {pill && (
        <span
          className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold"
          style={{ background: pill.bg, border: `1px solid ${pill.border}`, color: pill.color }}
        >
          {pill.icon}
          {pill.label}
        </span>
      )}
    </div>
  );
}

function EvidenceRow({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
      {children}
    </div>
  );
}

function EvidenceMetric({
  icon,
  label,
  value,
  valueColor,
  detail,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  valueColor: string;
  detail?: React.ReactNode;
}) {
  return (
    <div
      className="flex flex-col gap-1.5 p-2.5 rounded-lg"
      style={{
        background: "var(--bg-inset)",
        border: "1px solid var(--border)",
      }}
    >
      <div className="flex items-center gap-1.5">
        <span style={{ color: "var(--accent)" }}>{icon}</span>
        <span
          className="text-[10px] font-semibold"
          style={{ color: "var(--text-muted)" }}
        >
          {label}
        </span>
      </div>
      <div className="flex items-end justify-between gap-2">
        <span
          className="text-lg font-bold leading-none data-mono"
          style={{ color: valueColor }}
        >
          {value}
        </span>
      </div>
      {detail}
    </div>
  );
}

function EvidenceSection({
  icon,
  label,
  tag,
  tagColor,
  children,
}: {
  icon: React.ReactNode;
  label: string;
  tag?: string;
  tagColor?: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{
        background: "var(--bg-inset)",
        border: "1px solid var(--border)",
      }}
    >
      <div
        className="flex items-center justify-between px-3 py-2"
        style={{
          borderBottom: "1px solid var(--border)",
          background: "var(--bg-raised)",
        }}
      >
        <div className="flex items-center gap-1.5 min-w-0">
          {icon}
          <span
            className="text-[11px] font-semibold truncate"
            style={{ color: "var(--text-primary)" }}
          >
            {label}
          </span>
        </div>
        {tag && (
          <span
            className="text-[10px] font-mono flex-shrink-0 px-1.5 py-0.5 rounded"
            style={{
              color: tagColor ?? "var(--text-muted)",
              background: "var(--bg-inset)",
            }}
          >
            {tag}
          </span>
        )}
      </div>
      <div className="p-3">{children}</div>
    </div>
  );
}

function MiniVisual({
  title,
  icon,
  b64,
  caption,
}: {
  title: string;
  icon: React.ReactNode;
  b64: string;
  caption?: string;
}) {
  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{
        background: "var(--bg-inset)",
        border: "1px solid var(--border)",
      }}
    >
      <div
        className="flex items-center gap-1.5 px-3 py-2"
        style={{
          borderBottom: "1px solid var(--border)",
          background: "var(--bg-raised)",
        }}
      >
        {icon}
        <span className="text-[11px] font-semibold" style={{ color: "var(--text-primary)" }}>
          {title}
        </span>
      </div>
      <div
        className="rounded-md overflow-hidden m-2.5"
        style={{ border: "1px solid var(--border)", background: "var(--bg-base)" }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`data:image/png;base64,${b64}`}
          alt={title}
          className="w-full object-contain max-h-52"
        />
      </div>
      {caption && (
        <p
          className="text-[10px] text-center font-mono pb-2.5"
          style={{ color: "var(--text-muted)" }}
        >
          {caption}
        </p>
      )}
    </div>
  );
}
