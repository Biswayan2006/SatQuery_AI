"use client";

import { motion } from "framer-motion";
import { FileDown, Loader2, FileText, Download } from "lucide-react";
import { useReportDownload } from "@/hooks/useAnalysis";

interface Props {
  sessionId: string;
  taskType?: string;
  processingTime?: number;
}

export default function ReportDownload({ sessionId, taskType, processingTime }: Props) {
  const { downloadReport, downloading } = useReportDownload();

  const ready = !downloading;

  return (
    <div
      className="flex items-center justify-between flex-wrap gap-3 p-3 rounded-lg"
      style={{
        background: "var(--bg-inset)",
        border: "1px solid var(--border)",
      }}
    >
      <div className="flex items-center gap-3 min-w-0 flex-1">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
          style={{
            background: "var(--accent-soft)",
            border: "1px solid var(--accent-border)",
          }}
        >
          <FileText className="w-4 h-4" style={{ color: "var(--accent)" }} />
        </div>
        <div className="min-w-0 flex-1">
          <p
            className="text-xs font-semibold truncate"
            style={{ color: "var(--text-primary)" }}
          >
            Analysis Report
          </p>
          <p
            className="text-[10px] font-mono truncate"
            style={{ color: "var(--text-muted)" }}
            title={sessionId}
          >
            {sessionId.slice(0, 16)}…
            {taskType && (
              <span className="ml-1.5" style={{ color: "var(--text-faint)" }}>
                [{taskType.replace(/_/g, " ")}]
              </span>
            )}
            {processingTime != null && (
              <span className="ml-1.5" style={{ color: "var(--text-faint)" }}>
                {(processingTime / 1000).toFixed(1)}s
              </span>
            )}
          </p>
        </div>
      </div>

      <motion.button
        onClick={() => downloadReport(sessionId)}
        disabled={!ready}
        whileTap={ready ? { scale: 0.96 } : {}}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-semibold transition-colors no-tap"
        style={{
          background: ready ? "var(--accent)" : "var(--bg-raised)",
          border: `1px solid ${ready ? "var(--accent)" : "var(--border)"}`,
          color: ready ? "var(--accent-contrast)" : "var(--text-muted)",
          cursor: ready ? "pointer" : "not-allowed",
          opacity: ready ? 1 : 0.6,
        }}
      >
        {downloading ? (
          <>
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            <span>Generating…</span>
          </>
        ) : (
          <>
            <Download className="w-3.5 h-3.5" />
            <span>Download PDF</span>
            <FileDown className="w-3.5 h-3.5 opacity-60" />
          </>
        )}
      </motion.button>
    </div>
  );
}
