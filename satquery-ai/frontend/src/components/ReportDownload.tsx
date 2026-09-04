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
  const btnBg   = ready ? "rgba(56,139,253,0.14)"  : "rgba(48,54,61,0.4)";
  const btnBrd  = ready ? "rgba(56,139,253,0.32)"  : "rgba(48,54,61,0.5)";
  const btnTxt  = ready ? "#58A6FF"                : "#484F58";

  return (
    <div
      className="flex items-center justify-between flex-wrap gap-3 p-3 rounded-lg"
      style={{
        background: "rgba(48,54,61,0.2)",
        border: "1px solid rgba(48,54,61,0.5)",
      }}
    >
      <div className="flex items-center gap-3 min-w-0 flex-1">
        <div
          className="w-8 h-8 rounded-md flex items-center justify-center flex-shrink-0"
          style={{
            background: "rgba(56,139,253,0.1)",
            border: "1px solid rgba(56,139,253,0.22)",
          }}
        >
          <FileText className="w-4 h-4" style={{ color: "#58A6FF" }} />
        </div>
        <div className="min-w-0 flex-1">
          <p
            className="text-xs font-semibold truncate"
            style={{ color: "#E6EDF3" }}
          >
            Analysis Report
          </p>
          <p
            className="text-[10px] font-mono truncate"
            style={{ color: "#6E7681" }}
            title={sessionId}
          >
            {sessionId.slice(0, 16)}…
            {taskType && (
              <span className="ml-1.5" style={{ color: "#484F58" }}>
                · {taskType.replace(/_/g, " ")}
              </span>
            )}
            {processingTime != null && (
              <span className="ml-1.5" style={{ color: "#484F58" }}>
                · {(processingTime / 1000).toFixed(1)}s
              </span>
            )}
          </p>
        </div>
      </div>

      <motion.button
        onClick={() => downloadReport(sessionId)}
        disabled={!ready}
        whileTap={ready ? { scale: 0.96 } : {}}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[11px] font-semibold transition-colors no-tap"
        style={{
          background: btnBg,
          border: `1px solid ${btnBrd}`,
          color: btnTxt,
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
