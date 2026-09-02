"use client";

import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Satellite, Activity, Server, Cpu, AlertCircle } from "lucide-react";
import toast from "react-hot-toast";

import ImageUpload from "@/components/ImageUpload";
import QueryInput from "@/components/QueryInput";
import ResultDisplay from "@/components/ResultDisplay";
import ExecutionTrace from "@/components/ExecutionTrace";
import ChangeMap from "@/components/ChangeMap";
import ReportDownload from "@/components/ReportDownload";

import { useImageUpload, useAnalysis, checkHealth } from "@/hooks/useAnalysis";
import type { UploadedImage, HealthResponse } from "@/types";

export default function HomePage() {
  const [images, setImages] = useState<UploadedImage[]>([]);
  const [query, setQuery] = useState("");
  const [health, setHealth] = useState<HealthResponse | null>(null);

  const { uploadImage } = useImageUpload();
  const { loading, result, error, analyze, reset } = useAnalysis();

  // ── Health check ──────────────────────────────────────────────────────────
  useEffect(() => {
    checkHealth().then((h) => setHealth(h));
  }, []);

  // ── Image management ──────────────────────────────────────────────────────
  const handleAddImage = useCallback(
    async (file: File) => {
      const previewUrl = URL.createObjectURL(file);
      const newImg: UploadedImage = {
        file,
        previewUrl,
        uploadResponse: null,
        uploading: true,
        error: null,
      };

      setImages((prev) => [...prev, newImg]);

      try {
        const resp = await uploadImage(file);
        setImages((prev) =>
          prev.map((img) =>
            img.previewUrl === previewUrl
              ? { ...img, uploading: false, uploadResponse: resp, error: null }
              : img
          )
        );
        if (!resp.valid) {
          toast.error(`Image validation: ${resp.message}`);
        }
      } catch (err: any) {
        setImages((prev) =>
          prev.map((img) =>
            img.previewUrl === previewUrl
              ? { ...img, uploading: false, error: err.message }
              : img
          )
        );
        toast.error(`Upload failed: ${err.message}`);
      }
    },
    [uploadImage]
  );

  const handleRemoveImage = useCallback((index: number) => {
    setImages((prev) => {
      const removed = prev[index];
      if (removed?.previewUrl) URL.revokeObjectURL(removed.previewUrl);
      return prev.filter((_, i) => i !== index);
    });
    reset();
  }, [reset]);

  // ── Analysis ──────────────────────────────────────────────────────────────
  const handleAnalyze = useCallback(async () => {
    const readyImages = images.filter((img) => img.uploadResponse?.valid && img.uploadResponse.image_id);
    if (readyImages.length === 0) {
      toast.error("Upload at least one valid image first.");
      return;
    }
    const imageIds = readyImages.map((img) => img.uploadResponse!.image_id);
    await analyze(imageIds, query);
  }, [images, query, analyze]);

  const hasReadyImages = images.some((img) => img.uploadResponse?.valid);
  const imageCount = images.length;

  return (
    <div className="min-h-screen bg-space-gradient grid-bg">
      {/* ── Header ────────────────────────────────────────────────────────── */}
      <header className="border-b border-slate-800/60 bg-space-900/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative w-7 h-7">
              <Satellite className="w-7 h-7 text-satellite-400" />
              <div className="absolute inset-0 blur-md bg-satellite-400/30 animate-pulse-slow" />
            </div>
            <div>
              <h1 className="text-sm font-bold text-white tracking-tight">SatQuery AI</h1>
              <p className="text-[10px] text-slate-500 leading-none">Remote Sensing Intelligence</p>
            </div>
          </div>

          {/* Status indicators */}
          <div className="flex items-center gap-4">
            {health && (
              <>
                <StatusPill
                  icon={<Server className="w-3 h-3" />}
                  label={health.status}
                  ok={health.status === "healthy"}
                />
                <StatusPill
                  icon={<Cpu className="w-3 h-3" />}
                  label={health.device.toUpperCase()}
                  ok
                />
                <StatusPill
                  icon={<Activity className="w-3 h-3" />}
                  label={`${health.models_loaded} models`}
                  ok={health.models_loaded > 0}
                />
              </>
            )}
          </div>
        </div>
      </header>

      {/* ── Hero tagline ──────────────────────────────────────────────────── */}
      <div className="relative bg-satellite-glow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 text-center">
          <motion.h2
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-2xl sm:text-3xl font-bold text-white mb-2"
          >
            Multimodal Satellite Image Analysis
          </motion.h2>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-slate-400 text-sm max-w-xl mx-auto"
          >
            Upload optical, SAR, or multispectral imagery and ask natural language questions.
            The agentic pipeline routes your query to the best specialist model automatically.
          </motion.p>

          {/* Capability chips */}
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="flex flex-wrap justify-center gap-2 mt-4"
          >
            {["VQA", "Captioning", "Grounding", "Change Detection", "SAR Fusion"].map((cap) => (
              <span
                key={cap}
                className="px-3 py-1 rounded-full text-xs font-medium bg-slate-800/60 text-slate-300 border border-slate-700/40"
              >
                {cap}
              </span>
            ))}
          </motion.div>
        </div>
      </div>

      {/* ── Main three-panel layout ───────────────────────────────────────── */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6 pb-16">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* ── Left panel: Upload ─────────────────────────────────────────── */}
          <div className="lg:col-span-1">
            <SectionHeader
              number="1"
              title="Upload Images"
              subtitle={
                imageCount === 0
                  ? "Upload 1 image for VQA / captioning / grounding"
                  : imageCount === 1
                  ? "Add a second image for change detection or SAR fusion"
                  : "Ready for pairwise analysis"
              }
            />
            <ImageUpload
              images={images}
              onAddImage={handleAddImage}
              onRemoveImage={handleRemoveImage}
              maxImages={2}
            />
          </div>

          {/* ── Center panel: Query ───────────────────────────────────────── */}
          <div className="lg:col-span-1">
            <SectionHeader
              number="2"
              title="Ask a Question"
              subtitle="Natural language query — the agent handles task routing automatically"
            />
            <QueryInput
              query={query}
              onChange={setQuery}
              onSubmit={handleAnalyze}
              loading={loading}
              hasImages={hasReadyImages}
            />
          </div>

          {/* ── Right panel: Results ──────────────────────────────────────── */}
          <div className="lg:col-span-1">
            <SectionHeader
              number="3"
              title="Results"
              subtitle={
                result
                  ? `Session: ${result.session_id}`
                  : "Awaiting analysis…"
              }
            />

            <AnimatePresence mode="wait">
              {loading && <LoadingPanel key="loading" />}

              {error && !loading && (
                <motion.div
                  key="error"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-card p-5 border-red-900/40"
                >
                  <div className="flex items-start gap-3">
                    <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-semibold text-red-400 mb-1">Analysis failed</p>
                      <p className="text-xs text-slate-400">{error}</p>
                    </div>
                  </div>
                </motion.div>
              )}

              {result && !loading && (
                <motion.div
                  key="result"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex flex-col gap-4"
                >
                  <ResultDisplay result={result} />

                  {/* Change map gets its own dedicated component */}
                  {result.change_map && !result.visual_evidence && (
                    <ChangeMap
                      changeMapB64={result.change_map}
                      changePercentage={result.change_percentage}
                    />
                  )}

                  {/* Execution trace */}
                  <ExecutionTrace summary={result.execution_summary} />

                  {/* Report download */}
                  <ReportDownload
                    sessionId={result.session_id}
                    taskType={result.task}
                    processingTime={result.execution_summary.processing_time_ms}
                  />
                </motion.div>
              )}

              {!loading && !result && !error && (
                <EmptyState key="empty" hasImages={hasReadyImages} />
              )}
            </AnimatePresence>
          </div>
        </div>
      </main>

      {/* ── Footer ───────────────────────────────────────────────────────── */}
      <footer className="border-t border-slate-800/40 py-4 text-center">
        <p className="text-xs text-slate-600">
          SatQuery AI — Agentic Remote Sensing Intelligence · v1.0.0
        </p>
      </footer>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function SectionHeader({
  number,
  title,
  subtitle,
}: {
  number: string;
  title: string;
  subtitle: string;
}) {
  return (
    <div className="flex items-start gap-3 mb-4">
      <div className="w-7 h-7 rounded-full bg-satellite-600/20 border border-satellite-500/30 flex items-center justify-center flex-shrink-0 mt-0.5">
        <span className="text-xs font-bold text-satellite-400">{number}</span>
      </div>
      <div>
        <h2 className="text-sm font-semibold text-slate-200">{title}</h2>
        <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>
      </div>
    </div>
  );
}

function StatusPill({
  icon,
  label,
  ok,
}: {
  icon: React.ReactNode;
  label: string;
  ok: boolean;
}) {
  return (
    <div
      className={`hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs border ${
        ok
          ? "text-emerald-400 bg-emerald-900/20 border-emerald-700/30"
          : "text-red-400 bg-red-900/20 border-red-700/30"
      }`}
    >
      {icon}
      <span>{label}</span>
    </div>
  );
}

function LoadingPanel() {
  const steps = [
    "Validating input images…",
    "Classifying task type…",
    "Routing to specialist model…",
    "Running inference…",
    "Generating report…",
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-5 scan-container"
    >
      <div className="flex items-center gap-3 mb-5">
        <div className="relative">
          <Satellite className="w-5 h-5 text-satellite-400 animate-spin-slow" />
        </div>
        <span className="text-sm font-semibold text-satellite-400 animate-pulse">
          Agentic analysis in progress…
        </span>
      </div>
      <div className="space-y-2.5">
        {steps.map((step, i) => (
          <motion.div
            key={step}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.3, duration: 0.3 }}
            className="flex items-center gap-3"
          >
            <div className="flex gap-0.5">
              {[0, 1, 2].map((j) => (
                <span
                  key={j}
                  className="w-1 h-1 rounded-full bg-satellite-400 loading-dot"
                  style={{ animationDelay: `${i * 0.3 + j * 0.15}s` }}
                />
              ))}
            </div>
            <span className="text-xs text-slate-400 font-mono">{step}</span>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

function EmptyState({ hasImages }: { hasImages: boolean }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="glass-card p-8 text-center border-dashed border-slate-700/50"
    >
      <Satellite className="w-10 h-10 text-slate-600 mx-auto mb-3" />
      <p className="text-sm text-slate-500 font-medium">
        {hasImages ? "Enter a query and click Run Analysis" : "Upload an image to get started"}
      </p>
      <p className="text-xs text-slate-600 mt-1">
        Results will appear here after analysis
      </p>
    </motion.div>
  );
}
