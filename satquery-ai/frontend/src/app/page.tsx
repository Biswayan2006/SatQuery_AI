"use client";

import { useState, useCallback, useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Satellite, Send, AlertCircle, CheckCircle2,
  Eye, Map, Radio, Circle, Loader2,
  ChevronUp, Layers, ImageIcon, ScanLine,
  Search, Brain, Sparkles, Boxes, FileDown,
} from "lucide-react";
import dynamic from "next/dynamic";
import toast from "react-hot-toast";

// ── Existing components: all unchanged ──────────────────────────────────
import ImageUpload    from "@/components/ImageUpload";
import ResultDisplay  from "@/components/ResultDisplay";
import ExecutionTrace from "@/components/ExecutionTrace";
import ChangeMap      from "@/components/ChangeMap";
import CompareSlider  from "@/components/CompareSlider";
import ModelStatus    from "@/components/ModelStatus";
import ProjectTabs    from "@/components/layout/ProjectTabs";
import { useProject } from "@/context/ProjectContext";

// GlobeScene uses canvas: only render client-side
const GlobeScene = dynamic(() => import("@/components/GlobeScene"), { ssr: false });

// ── Existing hooks / types ────────────────────────────────────────────────
import { useImageUpload, useAnalysis, checkHealth, useReportDownload } from "@/hooks/useAnalysis";
import type { UploadedImage, HealthResponse, AnalysisResponse } from "@/types";

// ─────────────────────────────────────────────────────────────────────────────
// DashCard: shared card shell
// ─────────────────────────────────────────────────────────────────────────────
function DashCard({
  title,
  titleRight,
  children,
  className = "",
  noPad = false,
}: {
  title: string;
  titleRight?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  noPad?: boolean;
}) {
  return (
    <div
      className={`sq-card flex flex-col overflow-hidden ${className}`}
    >
      <div
        className="flex items-center justify-between px-4 py-3 flex-shrink-0"
        style={{ borderBottom: "1px solid var(--border)" }}
      >
        <span className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>{title}</span>
        {titleRight}
      </div>
      <div className={`flex-1 overflow-y-auto ${noPad ? "" : ""}`}>{children}</div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// CommandBar: top query bar with focus-triggered example query suggestions
// ─────────────────────────────────────────────────────────────────────────────
const EXAMPLE_QUERIES = [
  "Describe the land cover types",
  "What changed between these images?",
  "How many buildings are visible?",
  "Identify all water bodies",
  "Use SAR and optical together to analyze the scene",
  "What percentage of the area is forested?",
  "Detect any signs of flooding",
  "Locate the roads in this image",
] as const;

function CommandBar({
  query, onQueryChange, onSubmit, loading, hasImages,
}: {
  query: string;
  onQueryChange: (v: string) => void;
  onSubmit: () => void;
  loading: boolean;
  hasImages: boolean;
}) {
  const canSubmit = hasImages && query.trim().length > 0 && !loading;
  const [focused, setFocused] = useState(false);

  // Suggestions surface the moment the input gains focus while empty, and
  // disappear once the user starts typing or focus leaves the field.
  const showSuggestions = focused && query.trim().length === 0 && !loading;

  const pickSuggestion = (q: string) => {
    onQueryChange(q);
    setFocused(false);
  };

  return (
    <div className="relative">
      <div
        className="flex items-center gap-3 px-4 py-3 sq-card"
      >
        <div
          className="flex-shrink-0 w-9 h-9 rounded-lg flex items-center justify-center"
          style={{ background: "var(--accent-soft)", border: "1px solid var(--accent-border)" }}
        >
          <Satellite className="w-4 h-4" style={{ color: "var(--accent)" }} />
        </div>

        <input
          type="text"
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && canSubmit) onSubmit();
            if (e.key === "Escape") setFocused(false);
          }}
          placeholder="Describe the major features and detect changes in the region…"
          disabled={loading}
          className="flex-1 bg-transparent text-sm text-ink-soft placeholder:text-ink-faint outline-none disabled:opacity-60"
        />

        <button
          onClick={onSubmit}
          disabled={!canSubmit}
          className="flex-shrink-0 flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all"
          style={{
            background: canSubmit ? "var(--accent)" : "var(--accent-soft)",
            border: `1px solid ${canSubmit ? "var(--accent)" : "transparent"}`,
            color: canSubmit ? "var(--accent-contrast)" : "var(--text-muted)",
            cursor: canSubmit ? "pointer" : "not-allowed",
            opacity: canSubmit ? 1 : 0.6,
          }}
        >
          {loading
            ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
            : <Send    className="w-3.5 h-3.5" />
          }
          {loading ? "Analysing…" : "Send Query"}
        </button>
      </div>

      {/* ── Example query suggestions (focus-triggered dropdown) ──────────── */}
      <AnimatePresence>
        {showSuggestions && (
          <motion.div
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.15 }}
            className="absolute left-0 right-0 top-full mt-2 z-40 rounded-xl p-3"
            style={{
              background: "var(--bg-elevated)",
              border: "1px solid var(--border)",
              boxShadow: "var(--shadow-lg)",
            }}
          >
            <div className="flex items-center gap-2 mb-2.5 px-1">
              <Search className="w-3.5 h-3.5" style={{ color: "var(--text-muted)" }} />
              <span className="text-xs font-medium" style={{ color: "var(--text-muted)" }}>
                Example queries
              </span>
            </div>
            <div className="flex flex-wrap gap-2">
              {EXAMPLE_QUERIES.map((q) => (
                <button
                  key={q}
                  type="button"
                  // preventDefault on mousedown keeps the input from blurring
                  // before the click fires, so the query fills reliably
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => pickSuggestion(q)}
                  className="text-xs px-3 py-1.5 rounded-lg border border-line
                             bg-raised text-ink-soft transition-colors no-tap
                             hover:bg-accent-soft hover:border-accent hover:text-ink"
                >
                  {q}
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ImageSlot: compact preview tile for an uploaded image.
// `tint` color-codes the slot by modality (optical=water, SAR=amber, output=accent).
// ─────────────────────────────────────────────────────────────────────────────
function ImageSlot({
  label, src, sublabel, icon, loading = false, error = null, tint,
}: {
  label: string;
  src: string | null;
  sublabel?: string;
  icon?: React.ReactNode;
  loading?: boolean;
  error?: string | null;
  tint?: string;
}) {
  const hasError = !!error && !loading;
  const labelColor = hasError ? "var(--danger)" : (tint ?? "var(--text-muted)");

  return (
    <div className="flex flex-col gap-1.5">
      <div
        className="relative rounded-lg overflow-hidden flex items-center justify-center"
        style={{
          aspectRatio: "1/1",
          background: "var(--bg-inset)",
          border: hasError
            ? "1px solid color-mix(in srgb, var(--danger) 50%, transparent)"
            : "1px solid var(--border)",
        }}
      >
        {loading ? (
          <div className="flex flex-col items-center gap-1.5">
            <Loader2 className="w-5 h-5 animate-spin" style={{ color: "var(--accent)" }} />
            <p className="text-[9px] font-medium" style={{ color: "var(--accent)" }}>Processing…</p>
          </div>
        ) : hasError ? (
          <div className="flex flex-col items-center gap-1 px-2 text-center">
            <AlertCircle className="w-5 h-5" style={{ color: "var(--danger)" }} />
            <p
              className="text-[9px] font-medium leading-tight"
              style={{ color: "var(--danger)" }}
              title={error}
            >
              {error.length > 24 ? error.slice(0, 22) + "…" : error}
            </p>
          </div>
        ) : src ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={src}
            alt={label}
            className="w-full h-full object-cover"
            onError={(e) => {
              (e.currentTarget as HTMLImageElement).style.display = "none";
            }}
          />
        ) : (
          <div className="flex flex-col items-center gap-1.5 opacity-50">
            {icon ?? <ImageIcon className="w-5 h-5" style={{ color: "var(--text-faint)" }} />}
            <p className="text-[9px] leading-tight" style={{ color: "var(--text-faint)" }}>Empty</p>
          </div>
        )}
      </div>
      <div className="text-center">
        <p className="text-[10px] leading-tight font-medium" style={{ color: labelColor }}>
          {label}
        </p>
        {sublabel && (
          <p className="text-[9px] leading-tight mt-0.5 font-mono" style={{ color: "var(--text-faint)" }}>
            {sublabel}
          </p>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ProcessingStatus: animated agent status line
// ─────────────────────────────────────────────────────────────────────────────
function ProcessingStatus({
  loading, result, execTimeMs,
}: {
  loading: boolean;
  result: AnalysisResponse | null;
  execTimeMs?: number | null;
}) {
  return (
    <div className="flex items-center justify-between gap-2 w-full">
      <div className="flex items-center gap-2 min-w-0 flex-1">
        {loading ? (
          <>
            <span className="relative flex-shrink-0">
              <span
                className="w-2 h-2 rounded-full block animate-pulse"
                style={{ background: "var(--accent)" }}
              />
              <span
                className="absolute inset-0 w-2 h-2 rounded-full block"
                style={{
                  background: "var(--accent)",
                  opacity: 0.4,
                  animation: "ping 1.2s cubic-bezier(0,0,0.2,1) infinite",
                }}
              />
            </span>
            <span className="text-[11px] font-medium truncate" style={{ color: "var(--accent)" }}>
              Agent processing query…
            </span>
            <span className="flex gap-0.5 flex-shrink-0 ml-auto">
              {[0, 1, 2].map((i) => (
                <span
                  key={i}
                  className="w-1 h-1 rounded-full loading-dot"
                  style={{ background: "var(--accent)", animationDelay: `${i * 0.18}s` }}
                />
              ))}
            </span>
          </>
        ) : result ? (
          <>
            <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--veg)" }} />
            <span className="text-[11px] truncate" style={{ color: "var(--text-muted)" }}>
              Analysis complete:{" "}
              <span className="font-medium" style={{ color: "var(--text-secondary)" }}>
                {result.task.replace(/_/g, " ")}
              </span>
            </span>
            {execTimeMs != null && (
              <span className="text-[10px] font-mono ml-auto flex-shrink-0" style={{ color: "var(--text-faint)" }}>
                {(execTimeMs / 1000).toFixed(1)}s
              </span>
            )}
          </>
        ) : (
          <>
            <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: "var(--border-strong)" }} />
            <span className="text-[11px] truncate" style={{ color: "var(--text-muted)" }}>
              Upload imagery and submit a query to begin
            </span>
          </>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// RealtimeCard: state-aware visualization panel
// ─────────────────────────────────────────────────────────────────────────────
function RealtimeCard({
  images, result, loading,
}: {
  images: UploadedImage[];
  result: AnalysisResponse | null;
  loading: boolean;
}) {
  const hasImages   = images.length > 0;
  const img0        = images[0];
  const img1        = images[1];
  const showCompare = !loading && result?.change_map != null && images.length >= 2;
  const execTimeMs  = result?.execution_summary?.processing_time_ms ?? null;

  // Build rich metadata from upload response
  const buildMeta = (img: UploadedImage | undefined) => {
    if (!img) return undefined;
    if (img.uploading) return "Uploading…";
    const r = img.uploadResponse;
    if (!r) return undefined;
    const size = r.file_size_kb != null
      ? r.file_size_kb >= 1024
        ? `${(r.file_size_kb / 1024).toFixed(1)} MB`
        : `${r.file_size_kb.toFixed(0)} KB`
      : null;
    const parts: string[] = [];
    parts.push(`${r.modality}, ${r.bands}b`);
    if (size) parts.push(size);
    if (r.is_geotiff) parts.push("GeoTIFF");
    return parts.join("  ");
  };

  const img0Meta = buildMeta(img0);
  const img1Meta = buildMeta(img1);

  // Output slot
  const outputB64 = result?.change_map ?? result?.visual_evidence ?? result?.fusion_map ?? null;
  const outputLabel = result?.change_map
    ? "Change Map"
    : result?.visual_evidence
    ? "Visual Evidence"
    : result?.fusion_map
    ? "SAR Fusion"
    : "Output";

  const outputMeta = result && outputB64
    ? result.change_percentage != null
      ? `${result.change_percentage.toFixed(1)}% changed`
      : `${(result.confidence * 100).toFixed(0)}% confidence`
    : undefined;

  const LiveBadge = () => (
    <div className="flex items-center gap-1.5">
      <span className="relative">
        <span className="w-2 h-2 rounded-full block animate-pulse" style={{ background: "var(--veg)" }} />
      </span>
      <span className="text-[11px] font-semibold" style={{ color: "var(--veg)" }}>
        Live
      </span>
      <ChevronUp className="w-3.5 h-3.5" style={{ color: "var(--text-muted)" }} />
    </div>
  );

  return (
    <DashCard title="Real-time Analysis" titleRight={<LiveBadge />} noPad>
      <div className="flex flex-col h-full">

        {/* ── Main visualization area ─────────────────────────────────── */}
        <div className={`flex-1 relative ${loading ? "scan-container" : ""}`}>

          {/* NO IMAGES: Globe + empty state */}
          {!hasImages && !loading && (
            <div className="relative h-full min-h-[220px]">
              <div className="absolute inset-0">
                <GlobeScene className="w-full h-full opacity-40" />
              </div>
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-center px-6">
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center"
                  style={{ background: "var(--accent-soft)", border: "1px solid var(--accent-border)" }}
                >
                  <ScanLine className="w-5 h-5" style={{ color: "var(--accent)" }} />
                </div>
                <div>
                  <p className="text-sm font-semibold" style={{ color: "var(--text-secondary)" }}>
                    No imagery available
                  </p>
                  <p className="text-xs mt-1 leading-relaxed" style={{ color: "var(--text-muted)" }}>
                    Upload optical or SAR imagery to begin analysis
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* COMPARE MODE: change_map result + 2 images */}
          {showCompare && (
            <div className="p-4 space-y-3">
              <div>
                <p className="text-[11px] font-medium mb-2" style={{ color: "var(--text-muted)" }}>
                  Before / after comparison
                </p>
                <CompareSlider
                  beforeSrc={img0!.previewUrl}
                  afterSrc={img1!.previewUrl}
                  beforeLabel={img0?.uploadResponse?.modality?.toUpperCase() ?? "T1"}
                  afterLabel={img1?.uploadResponse?.modality?.toUpperCase() ?? "T2"}
                  className="h-[160px]"
                />
              </div>
              <ChangeMap
                changeMapB64={result!.change_map!}
                changePercentage={result!.change_percentage}
              />
            </div>
          )}

          {/* STANDARD MODE: image slots grid */}
          {(hasImages || loading) && !showCompare && (
            <div className="p-4 space-y-3">
              <div className="grid grid-cols-3 gap-2.5">
                <ImageSlot
                  label="Optical Image"
                  sublabel={img0Meta}
                  src={img0?.previewUrl ?? null}
                  loading={img0?.uploading ?? false}
                  error={img0?.error ?? null}
                  tint="var(--water)"
                  icon={<Eye className="w-5 h-5" style={{ color: "var(--water)" }} />}
                />
                <ImageSlot
                  label="SAR Image"
                  sublabel={img1Meta}
                  src={img1?.previewUrl ?? null}
                  loading={img1?.uploading ?? false}
                  error={img1?.error ?? null}
                  tint="var(--sar)"
                  icon={<Radio className="w-5 h-5" style={{ color: "var(--sar)" }} />}
                />
                <ImageSlot
                  label={outputLabel}
                  sublabel={outputMeta}
                  src={outputB64 ? `data:image/png;base64,${outputB64}` : null}
                  loading={loading && !outputB64}
                  tint="var(--accent)"
                  icon={<Map className="w-5 h-5" style={{ color: "var(--accent)" }} />}
                />
              </div>

              {/* Visual evidence full-width when present and not change map */}
              {result?.visual_evidence && !result.change_map && (
                <div>
                  <p className="text-[11px] font-medium mb-2" style={{ color: "var(--text-muted)" }}>
                    Visual evidence
                  </p>
                  <div
                    className="rounded-lg overflow-hidden"
                    style={{ border: "1px solid var(--border)", background: "var(--bg-inset)" }}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`data:image/png;base64,${result.visual_evidence}`}
                      alt="Visual evidence"
                      className="w-full object-contain max-h-40"
                    />
                  </div>
                </div>
              )}

              {/* Fusion map full-width when present */}
              {result?.fusion_map && !result.change_map && (
                <div>
                  <p className="text-[11px] font-medium mb-2" style={{ color: "var(--text-muted)" }}>
                    SAR-optical fusion
                  </p>
                  <div
                    className="rounded-lg overflow-hidden"
                    style={{ border: "1px solid var(--border)", background: "var(--bg-inset)" }}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`data:image/png;base64,${result.fusion_map}`}
                      alt="Fusion map"
                      className="w-full object-contain max-h-40"
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Status bar ───────────────────────────────────────────────── */}
        <div
          className="px-4 py-2.5 flex-shrink-0 space-y-2"
          style={{ borderTop: "1px solid var(--border)" }}
        >
          <ProcessingStatus loading={loading} result={result} execTimeMs={execTimeMs} />
          <ModelStatus />
        </div>
      </div>
    </DashCard>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// AnalyzeDataCard: wraps ImageUpload unchanged
// ─────────────────────────────────────────────────────────────────────────────
function AnalyzeDataCard({
  images, onAddImage, onRemoveImage,
}: {
  images: UploadedImage[];
  onAddImage: (file: File) => void;
  onRemoveImage: (index: number) => void;
}) {
  return (
    <DashCard title="Analyze Data">
      <div className="p-4 space-y-3">
        <p className="text-[11px]" style={{ color: "var(--text-muted)" }}>
          Drag or select single/paired GeoTIFF/TIFF images
        </p>
        <ImageUpload
          images={images}
          onAddImage={onAddImage}
          onRemoveImage={onRemoveImage}
          maxImages={2}
        />
      </div>
    </DashCard>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// InsightsCard: ResultDisplay + ExecutionTrace + ReportDownload
// ─────────────────────────────────────────────────────────────────────────────
function InsightsCard({
  result, loading, error,
}: {
  result: AnalysisResponse | null;
  loading: boolean;
  error: string | null;
}) {
  const [auditOpen, setAuditOpen] = useState(false);
  const { downloadReport, downloading } = useReportDownloadHookWrapper();

  return (
    <DashCard
      title="Insights & Evidence"
      titleRight={
        result ? (
          <div className="flex items-center gap-2">
            {/* Download Report action */}
            <button
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-colors no-tap"
              style={{
                background: downloading ? "var(--bg-raised)" : "var(--accent-soft)",
                border: `1px solid ${downloading ? "var(--border)" : "var(--accent-border)"}`,
                color: downloading ? "var(--text-muted)" : "var(--accent)",
                cursor: downloading ? "not-allowed" : "pointer",
                opacity: downloading ? 0.75 : 1,
              }}
              disabled={downloading}
              onClick={() => downloadReport(result.session_id)}
            >
              {downloading
                ? <Loader2 className="w-3 h-3 animate-spin" />
                : <FileDown className="w-3 h-3" />
              }
              Download Report
            </button>
            {/* Audit Trail toggle */}
            <button
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-colors no-tap"
              style={{
                background: auditOpen ? "var(--accent-soft)" : "var(--bg-raised)",
                border: `1px solid ${auditOpen ? "var(--accent-border)" : "var(--border)"}`,
                color: auditOpen ? "var(--accent)" : "var(--text-muted)",
              }}
              onClick={() => setAuditOpen((v) => !v)}
            >
              <Layers className="w-3 h-3" />
              Audit Trail
              {auditOpen && (
                <span
                  className="text-[9px] font-semibold px-1 py-0.5 rounded"
                  style={{ background: "var(--accent-soft)", color: "var(--accent)" }}
                >
                  Open
                </span>
              )}
            </button>
          </div>
        ) : undefined
      }
    >
      <div className="p-3.5 space-y-3">
        {/* Loading */}
        {loading && (
          <div className="flex flex-col items-center justify-center gap-3 py-12">
            <div className="relative">
              <div
                className="w-10 h-10 rounded-xl flex items-center justify-center"
                style={{ background: "var(--accent-soft)", border: "1px solid var(--accent-border)" }}
              >
                <Loader2 className="w-5 h-5 animate-spin" style={{ color: "var(--accent)" }} />
              </div>
              <div
                className="absolute -inset-2 rounded-2xl pointer-events-none"
                style={{
                  border: "1px solid var(--accent-border)",
                  opacity: 0.6,
                  animation: "ping 1.6s cubic-bezier(0,0,0.2,1) infinite",
                }}
              />
            </div>
            <div className="text-center">
              <p className="text-xs font-semibold" style={{ color: "var(--text-secondary)" }}>
                Running satellite analysis
              </p>
              <p className="text-[11px] mt-0.5" style={{ color: "var(--text-muted)" }}>
                Classifying task, selecting specialist models, executing pipeline
              </p>
            </div>
          </div>
        )}

        {/* Error */}
        {error && !loading && (
          <div
            className="flex items-start gap-3 p-3 rounded-lg"
            style={{ background: "var(--change-soft)", border: "1px solid color-mix(in srgb, var(--danger) 22%, transparent)" }}
          >
            <div
              className="w-7 h-7 rounded-md flex items-center justify-center flex-shrink-0 mt-0.5"
              style={{ background: "var(--change-soft)", border: "1px solid color-mix(in srgb, var(--danger) 28%, transparent)" }}
            >
              <AlertCircle className="w-3.5 h-3.5" style={{ color: "var(--danger)" }} />
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-xs font-semibold" style={{ color: "var(--danger)" }}>
                Analysis failed
              </p>
              <p className="text-[11px] mt-0.5 leading-relaxed" style={{ color: "var(--text-muted)" }}>
                {error}
              </p>
            </div>
          </div>
        )}

        {/* Results: evidence sections */}
        {result && !loading && (
          <>
            <ResultDisplay result={result} />

            {/* Audit trail: ExecutionTrace */}
            <AnimatePresence>
              {auditOpen && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.22 }}
                  className="overflow-hidden"
                >
                  <ExecutionTrace summary={result.execution_summary} defaultOpen={true} />
                </motion.div>
              )}
            </AnimatePresence>

            {/* Report download row: inline report info */}
            <div
              className="flex items-center justify-between flex-wrap gap-3 p-2.5 rounded-lg"
              style={{ background: "var(--accent-soft)", border: "1px solid var(--accent-border)" }}
            >
              <div className="flex items-center gap-2 min-w-0 flex-1">
                <div
                  className="w-7 h-7 rounded-md flex items-center justify-center flex-shrink-0"
                  style={{ background: "var(--accent-soft)", border: "1px solid var(--accent-border)" }}
                >
                  <Satellite className="w-3.5 h-3.5" style={{ color: "var(--accent)" }} />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-[11px] font-semibold truncate" style={{ color: "var(--text-secondary)" }}>
                    Analysis Report
                  </p>
                  <p
                    className="text-[10px] font-mono truncate"
                    style={{ color: "var(--text-muted)" }}
                    title={result.session_id}
                  >
                    session {result.session_id.slice(0, 12)}… | {result.task.replace(/_/g, " ")} | {(result.execution_summary.processing_time_ms / 1000).toFixed(1)}s
                  </p>
                </div>
              </div>
              <button
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-semibold transition-colors no-tap"
                style={{
                  background: downloading ? "var(--bg-raised)" : "var(--accent)",
                  border: `1px solid ${downloading ? "var(--border)" : "var(--accent)"}`,
                  color: downloading ? "var(--text-muted)" : "var(--accent-contrast)",
                  cursor: downloading ? "not-allowed" : "pointer",
                  opacity: downloading ? 0.75 : 1,
                }}
                disabled={downloading}
                onClick={() => downloadReport(result.session_id)}
              >
                {downloading
                  ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  : <FileDown className="w-3.5 h-3.5" />
                }
                {downloading ? "Generating PDF…" : "Download PDF"}
              </button>
            </div>
          </>
        )}

        {/* Empty state */}
        {!result && !loading && !error && (
          <div className="flex flex-col items-center justify-center gap-2.5 py-12 text-center px-4">
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center"
              style={{ background: "var(--bg-raised)", border: "1px solid var(--border)" }}
            >
              <Satellite className="w-5 h-5" style={{ color: "var(--text-faint)" }} />
            </div>
            <div>
              <p className="text-xs font-semibold" style={{ color: "var(--text-secondary)" }}>
                No evidence yet
              </p>
              <p className="text-[11px] mt-0.5 leading-relaxed" style={{ color: "var(--text-muted)" }}>
                Upload imagery and submit a query to inspect satellite analysis results
              </p>
            </div>
            <div
              className="flex items-center gap-2 px-2.5 py-1 rounded-md mt-1"
              style={{ background: "var(--bg-raised)", border: "1px solid var(--border)" }}
            >
              <Search className="w-3 h-3" style={{ color: "var(--text-faint)" }} />
              <span className="text-[10px] font-mono" style={{ color: "var(--text-muted)" }}>
                Supports VQA, Captioning, Grounding, Change Detection, and SAR Fusion
              </span>
            </div>
          </div>
        )}
      </div>
    </DashCard>
  );
}

// Lightweight hook wrapper to reuse the existing real report download logic
function useReportDownloadHookWrapper() {
  const { downloadReport, downloading } = useReportDownload();
  return { downloadReport, downloading };
}

// ─────────────────────────────────────────────────────────────────────────────
// WorkflowSteps: maps real ExecutionSummary to a compact timeline.
// This IS a genuine sequence (classify, select, execute, integrate), so the
// stepped-timeline treatment is warranted. Each stage owns a data-palette hue.
// ─────────────────────────────────────────────────────────────────────────────
type StepStatus = "pending" | "active" | "completed" | "failed";

interface StepNode {
  id: string;
  label: string;
  detail?: string;
  status: StepStatus;
  pills?: { text: string; color: string }[];
  icon: React.ReactNode;
}

function WorkflowSteps({
  result, loading, error,
}: {
  result: AnalysisResponse | null;
  loading: boolean;
  error?: string | null;
}) {
  const hasError = !!error && !loading;

  const steps: StepNode[] = buildWorkflowNodes(result, loading, hasError, error ?? null);

  return (
    <div className="px-3.5 py-3.5">
      <div className="relative">
        {/* Vertical connector behind nodes */}
        <div className="absolute left-[10px] top-[10px] bottom-[10px]" style={{ width: "2px" }}>
          {steps.map((_, i) => {
            const isLast = i === steps.length - 1;
            const thisDone = steps[i].status === "completed" || steps[i].status === "failed";
            const nextActive = !isLast && (steps[i + 1].status === "active" || steps[i + 1].status === "completed");
            const filled = thisDone && nextActive;
            return (
              <div
                key={i}
                className="w-full"
                style={{
                  height: isLast ? "0px" : `calc(100% / ${steps.length})`,
                  background: filled ? "var(--accent)" : "var(--border-strong)",
                  opacity: filled ? 1 : 0.6,
                  transition: "background 0.4s ease, opacity 0.4s ease",
                }}
              />
            );
          })}
        </div>

        {/* Nodes */}
        <div className="space-y-3.5 relative">
          {steps.map((step) => (
            <TimelineNode key={step.id} step={step} />
          ))}
        </div>
      </div>
    </div>
  );
}

function TimelineNode({ step }: { step: StepNode }) {
  const { status } = step;

  const {
    nodeBg, nodeBorder, nodeRing,
    iconColor, textLabelColor, textDetailColor,
  } = statusStyles[status];

  return (
    <div className="flex items-start gap-3 relative z-10">
      {/* Node dot */}
      <div
        className="mt-0.5 w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 transition-all duration-300"
        style={{
          background: nodeBg,
          border: `1.5px solid ${nodeBorder}`,
          ...(nodeRing ? { boxShadow: `0 0 0 3px ${nodeRing}` } : {}),
        }}
      >
        {status === "active" ? (
          <Loader2 className="w-3 h-3 animate-spin" style={{ color: iconColor }} />
        ) : status === "completed" ? (
          <CheckCircle2 className="w-3 h-3" style={{ color: iconColor }} />
        ) : status === "failed" ? (
          <AlertCircle className="w-3 h-3" style={{ color: iconColor }} />
        ) : (
          <Circle className="w-2.5 h-2.5" style={{ color: iconColor }} />
        )}
      </div>

      {/* Content */}
      <div className="min-w-0 flex-1 pt-0.5">
        <div className="flex items-center gap-2 flex-wrap">
          {step.icon}
          <p className="text-[12px] font-semibold leading-snug" style={{ color: textLabelColor }}>
            {step.label}
          </p>
          <StatusPill status={status} />
        </div>
        {step.detail && (
          <p className="text-[10.5px] mt-0.5 leading-snug" style={{ color: textDetailColor }}>
            {step.detail}
          </p>
        )}
        {step.pills && step.pills.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-1.5">
            {step.pills.map((p, i) => (
              <span
                key={i}
                className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                style={{
                  background: `color-mix(in srgb, ${p.color} 14%, transparent)`,
                  border: `1px solid color-mix(in srgb, ${p.color} 32%, transparent)`,
                  color: p.color,
                }}
              >
                {p.text}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function StatusPill({ status }: { status: StepStatus }) {
  const map: Record<StepStatus, { label: string; color: string; bg: string; border: string }> = {
    pending:   { label: "Pending", color: "var(--text-faint)", bg: "var(--bg-raised)",  border: "var(--border)" },
    active:    { label: "Running", color: "var(--accent)",     bg: "var(--accent-soft)", border: "var(--accent-border)" },
    completed: { label: "Done",    color: "var(--veg)",        bg: "var(--veg-soft)",    border: "color-mix(in srgb, var(--veg) 30%, transparent)" },
    failed:    { label: "Failed",  color: "var(--danger)",     bg: "var(--change-soft)", border: "color-mix(in srgb, var(--danger) 30%, transparent)" },
  };
  const s = map[status];
  return (
    <span
      className="text-[9px] font-semibold px-1.5 py-0.5 rounded flex-shrink-0"
      style={{ color: s.color, background: s.bg, border: `1px solid ${s.border}` }}
    >
      {s.label}
    </span>
  );
}

const statusStyles: Record<StepStatus, {
  nodeBg: string; nodeBorder: string; nodeRing?: string;
  iconColor: string; textLabelColor: string; textDetailColor: string;
}> = {
  pending: {
    nodeBg: "transparent",
    nodeBorder: "var(--border-strong)",
    iconColor: "var(--border-strong)",
    textLabelColor: "var(--text-muted)",
    textDetailColor: "var(--text-faint)",
  },
  active: {
    nodeBg: "var(--accent-soft)",
    nodeBorder: "var(--accent)",
    nodeRing: "var(--accent-soft)",
    iconColor: "var(--accent)",
    textLabelColor: "var(--text-primary)",
    textDetailColor: "var(--text-secondary)",
  },
  completed: {
    nodeBg: "var(--veg-soft)",
    nodeBorder: "var(--veg)",
    iconColor: "var(--veg)",
    textLabelColor: "var(--text-secondary)",
    textDetailColor: "var(--text-muted)",
  },
  failed: {
    nodeBg: "var(--change-soft)",
    nodeBorder: "var(--danger)",
    iconColor: "var(--danger)",
    textLabelColor: "var(--danger)",
    textDetailColor: "var(--danger)",
  },
};

function buildWorkflowNodes(
  result: AnalysisResponse | null,
  loading: boolean,
  hasError: boolean,
  error: string | null,
): StepNode[] {
  // Each pipeline stage owns a data-palette hue.
  const iconClassify  = <Search    className="w-3 h-3 flex-shrink-0" style={{ color: "var(--accent)" }} />;
  const iconModel     = <Brain     className="w-3 h-3 flex-shrink-0" style={{ color: "var(--water)" }} />;
  const iconExecute   = <Boxes     className="w-3 h-3 flex-shrink-0" style={{ color: "var(--sar)" }} />;
  const iconIntegrate = <Sparkles  className="w-3 h-3 flex-shrink-0" style={{ color: "var(--veg)" }} />;

  // ── IDLE: all pending ──────────────────────────────────────────
  if (!result && !loading && !hasError) {
    return [
      { id: "c", label: "Task Classification", status: "pending",   icon: iconClassify },
      { id: "m", label: "Model Selection",      status: "pending",   icon: iconModel },
      { id: "e", label: "Execution Steps",      status: "pending",   icon: iconExecute },
      { id: "i", label: "Output Integration",   status: "pending",   icon: iconIntegrate },
    ];
  }

  // ── LOADING: classify active, rest pending ────────────────────
  if (loading && !result) {
    return [
      { id: "c", label: "Task Classification", status: "active",  icon: iconClassify, detail: "Analyzing query features and image modalities…" },
      { id: "m", label: "Model Selection",      status: "pending", icon: iconModel },
      { id: "e", label: "Execution Steps",      status: "pending", icon: iconExecute },
      { id: "i", label: "Output Integration",   status: "pending", icon: iconIntegrate },
    ];
  }

  // ── LOADING with partial existing result: first 2 done, execute active ──
  if (loading && result) {
    const es = result.execution_summary;
    return [
      {
        id: "c", label: "Task Classification", status: "completed", icon: iconClassify,
        detail: `${es.selected_task.replace(/_/g, " ")} (${(es.task_confidence * 100).toFixed(0)}% confidence)`,
        pills: [{ text: es.selected_task, color: "var(--accent)" }],
      },
      {
        id: "m", label: "Model Selection", status: "completed", icon: iconModel,
        detail: `${es.models_used.length} specialist model${es.models_used.length === 1 ? "" : "s"} selected`,
        pills: es.models_used.map((m) => ({ text: m, color: "var(--water)" })),
      },
      {
        id: "e", label: "Execution Steps", status: "active", icon: iconExecute,
        detail: "Running specialist pipeline on imagery inputs…",
      },
      { id: "i", label: "Output Integration", status: "pending", icon: iconIntegrate },
    ];
  }

  // ── ERROR: classify done, model done or active, execute failed ──
  if (hasError) {
    const classifyDone = !!result?.execution_summary.selected_task;
    const modelsUsed = result?.execution_summary.models_used ?? [];
    return [
      classifyDone && result
        ? {
            id: "c" as const,
            label: "Task Classification",
            status: "completed" as StepStatus,
            icon: iconClassify,
            detail: `${result.execution_summary.selected_task.replace(/_/g, " ")} (${(result.execution_summary.task_confidence * 100).toFixed(0)}% confidence)`,
            pills: [{ text: result.execution_summary.selected_task, color: "var(--accent)" }],
          }
        : {
            id: "c" as const,
            label: "Task Classification",
            status: "failed" as StepStatus,
            icon: iconClassify,
            detail: error ?? "Unknown classification error",
          },
      modelsUsed.length > 0
        ? {
            id: "m" as const,
            label: "Model Selection",
            status: "completed" as StepStatus,
            icon: iconModel,
            detail: `${modelsUsed.length} specialist model${modelsUsed.length === 1 ? "" : "s"} selected`,
            pills: modelsUsed.map((m) => ({ text: m, color: "var(--water)" })),
          }
        : {
            id: "m" as const,
            label: "Model Selection",
            status: "failed" as StepStatus,
            icon: iconModel,
          },
      {
        id: "e",
        label: "Execution Steps",
        status: "failed",
        icon: iconExecute,
        detail: error ?? "Execution aborted: see console or server logs for details",
      },
      { id: "i", label: "Output Integration", status: "pending", icon: iconIntegrate },
    ];
  }

  // ── COMPLETED: all 4 done, real data ──────────────────────────
  const es = result!.execution_summary;
  return [
    {
      id: "c", label: "Task Classification", status: "completed", icon: iconClassify,
      detail: `Classifier selected task with ${(es.task_confidence * 100).toFixed(0)}% confidence`,
      pills: [{ text: es.selected_task.replace(/_/g, " "), color: "var(--accent)" }],
    },
    {
      id: "m", label: "Model Selection", status: "completed", icon: iconModel,
      detail: `Dispatched to ${es.models_used.length} specialist model${es.models_used.length === 1 ? "" : "s"}`,
      pills: es.models_used.map((m) => ({ text: m, color: "var(--water)" })),
    },
    {
      id: "e", label: "Execution Steps", status: "completed", icon: iconExecute,
      detail: es.steps.map((s) => s.replace(/_/g, " ")).join("  ›  "),
      pills: [
        { text: `${es.steps.length} steps`, color: "var(--sar)" },
        { text: `${es.processing_time_ms.toFixed(0)} ms`, color: "var(--accent)" },
      ],
    },
    {
      id: "i", label: "Output Integration", status: "completed", icon: iconIntegrate,
      detail: `Response validated, evidence-grounded, ${result!.session_id.slice(0, 10)}`,
      pills: [
        { text: `Conf ${(result!.confidence * 100).toFixed(0)}%`, color: result!.confidence >= 0.7 ? "var(--veg)" : result!.confidence >= 0.4 ? "var(--warning)" : "var(--danger)" },
      ],
    },
  ];
}

function AgenticWorkflowCard({
  result, loading, error,
}: {
  result: AnalysisResponse | null;
  loading: boolean;
  error?: string | null;
}) {
  return (
    <DashCard title="Agentic Workflow" titleRight={<ChevronUp className="w-3.5 h-3.5" style={{ color: "var(--text-muted)" }} />}>
      <WorkflowSteps result={result} loading={loading} error={error} />
    </DashCard>
  );
}

interface ProjectWorkspaceState {
  images: UploadedImage[];
  query: string;
  result: AnalysisResponse | null;
  imageRevision: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// HomePage: main page, all hooks and API logic preserved exactly
// ─────────────────────────────────────────────────────────────────────────────
export default function HomePage() {
  const { activeProject, projects, workspaces, setWorkspaces } = useProject();
  const [health, setHealth] = useState<HealthResponse | null>(null);

  const { uploadImage }                    = useImageUpload();
  const { loading, error, analyze, reset } = useAnalysis();

  const currentWorkspace = workspaces[activeProject.id] ?? {
    images: [],
    query: "",
    result: null,
    imageRevision: 0,
  };
  const images = currentWorkspace.images;
  const query  = currentWorkspace.query;
  const result = currentWorkspace.result;

  /* Health poll: unchanged */
  useEffect(() => {
    checkHealth().then(setHealth);
    const id = setInterval(() => checkHealth().then(setHealth), 30_000);
    return () => clearInterval(id);
  }, []);

  /* Clear / reset analysis hook state whenever activeProject changes */
  useEffect(() => {
    reset();
  }, [activeProject.id, reset]);

  const setQuery = useCallback((newQuery: string) => {
    setWorkspaces((prev) => ({
      ...prev,
      [activeProject.id]: {
        ...(prev[activeProject.id] ?? { images: [], query: "", result: null, imageRevision: 0 }),
        query: newQuery,
      },
    }));
  }, [activeProject.id, setWorkspaces]);

  /* Image upload: scoped to active project ID and clears prior analysis result */
  const handleAddImage = useCallback(async (file: File) => {
    const previewUrl = URL.createObjectURL(file);
    const targetProjId = activeProject.id;
    const newImage: UploadedImage = {
      file,
      previewUrl,
      uploadResponse: null,
      uploading: true,
      error: null,
    };

    setWorkspaces((prev) => {
      const ws = prev[targetProjId] ?? { images: [], query: "", result: null, imageRevision: 0 };
      return {
        ...prev,
        [targetProjId]: {
          ...ws,
          images: [...ws.images, newImage],
          result: null,
          imageRevision: (ws.imageRevision ?? 0) + 1,
        },
      };
    });
    reset();

    try {
      const resp = await uploadImage(file);
      setWorkspaces((prev) => {
        const ws = prev[targetProjId];
        if (!ws) return prev;
        return {
          ...prev,
          [targetProjId]: {
            ...ws,
            images: ws.images.map((img) =>
              img.previewUrl === previewUrl
                ? { ...img, uploading: false, uploadResponse: resp, error: null }
                : img
            ),
          },
        };
      });
      if (!resp.valid) toast.error(`Image validation: ${resp.message}`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Upload failed";
      setWorkspaces((prev) => {
        const ws = prev[targetProjId];
        if (!ws) return prev;
        return {
          ...prev,
          [targetProjId]: {
            ...ws,
            images: ws.images.map((img) =>
              img.previewUrl === previewUrl
                ? { ...img, uploading: false, error: msg }
                : img
            ),
          },
        };
      });
      toast.error(`Upload failed: ${msg}`);
    }
  }, [uploadImage, reset, activeProject.id, setWorkspaces]);

  const handleRemoveImage = useCallback((index: number) => {
    const targetProjId = activeProject.id;
    setWorkspaces((prev) => {
      const ws = prev[targetProjId];
      if (!ws) return prev;
      const removed = ws.images[index];
      if (removed?.previewUrl) URL.revokeObjectURL(removed.previewUrl);
      return {
        ...prev,
        [targetProjId]: {
          ...ws,
          images: ws.images.filter((_, i) => i !== index),
          result: null,
          imageRevision: (ws.imageRevision ?? 0) + 1,
        },
      };
    });
    reset();
  }, [reset, activeProject.id, setWorkspaces]);

  /* Analysis: scoped to active project ID and verified against captured imageRevision */
  const handleAnalyze = useCallback(async () => {
    const ready = images.filter(
      (img) => img.uploadResponse?.valid && img.uploadResponse.image_id
    );
    if (!ready.length) {
      toast.error("Upload at least one valid image first.");
      return;
    }
    const targetProjId = activeProject.id;
    const capturedRevision = currentWorkspace.imageRevision ?? 0;

    const res = await analyze(ready.map((img) => img.uploadResponse!.image_id), query);
    if (res) {
      setWorkspaces((prev) => {
        // Ensure pending analysis result writes verify the project ID still exists before updating workspace state
        if (!projects.some((p) => p.id === targetProjId)) {
          return prev;
        }
        const ws = prev[targetProjId];
        // Only store if the image revision still matches what was captured before analyze started
        if (!ws || (ws.imageRevision ?? 0) !== capturedRevision) {
          return prev;
        }
        return {
          ...prev,
          [targetProjId]: {
            ...ws,
            result: res,
          },
        };
      });
    }
  }, [images, query, analyze, activeProject.id, projects, currentWorkspace.imageRevision, setWorkspaces]);

  const hasReadyImages = images.some((img) => img.uploadResponse?.valid);

  return (
    <div className="flex flex-col gap-4">

      {/* Models loading banner */}
      {health && health.models_loaded === 0 && (
        <div
          className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs"
          style={{
            background: "var(--sar-soft)",
            border: "1px solid color-mix(in srgb, var(--warning) 24%, transparent)",
            color: "var(--warning)",
          }}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse flex-shrink-0" />
          AI models loading: 0/5 ready. Responses will be mocked until models finish loading.
        </div>
      )}

      {/* ── Active project workspace & project tabs ──────────────────── */}
      <div className="flex items-center justify-between flex-wrap gap-2 px-1">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold" style={{ color: "var(--text-muted)" }}>
            Active project:
          </span>
          <span className="text-xs font-bold" style={{ color: "var(--text-primary)" }}>
            {activeProject.name}
          </span>
          <span
            className="text-[10px] font-mono px-2 py-0.5 rounded-full"
            style={{
              background: "color-mix(in srgb, var(--accent) 12%, transparent)",
              color: "var(--accent)",
              border: "1px solid color-mix(in srgb, var(--accent) 25%, transparent)",
            }}
          >
            {activeProject.sensor} ({activeProject.modality})
          </span>
        </div>
        <ProjectTabs />
      </div>

      {/* ── 1. Command bar ────────────────────────────────────────────── */}
      <CommandBar
        query={query}
        onQueryChange={setQuery}
        onSubmit={handleAnalyze}
        loading={loading}
        hasImages={hasReadyImages}
      />

      {/* ── 2+3. Analyze Data | Real-time Analysis ────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <AnalyzeDataCard
          images={images}
          onAddImage={handleAddImage}
          onRemoveImage={handleRemoveImage}
        />
        <RealtimeCard images={images} result={result} loading={loading} />
      </div>

      {/* ── 4+5. Insights & Evidence | Agentic Workflow ───────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <InsightsCard result={result} loading={loading} error={error} />
        <AgenticWorkflowCard result={result} loading={loading} error={error} />
      </div>

    </div>
  );
}
