"use client";

import { motion } from "framer-motion";
import { FileDown, Loader2, FileText } from "lucide-react";
import { useReportDownload } from "@/hooks/useAnalysis";

interface Props {
  sessionId: string;
  taskType?: string;
  processingTime?: number;
}

export default function ReportDownload({ sessionId, taskType, processingTime }: Props) {
  const { downloadReport, downloading } = useReportDownload();

  return (
    <div className="glass-card p-4">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-slate-800 border border-slate-600/50 flex items-center justify-center">
            <FileText className="w-4 h-4 text-satellite-400" />
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-200">Analysis Report</p>
            <p className="text-xs text-slate-500 font-mono">
              {sessionId.slice(0, 20)}…
              {taskType && (
                <span className="ml-2 text-slate-600">· {taskType}</span>
              )}
              {processingTime && (
                <span className="ml-2 text-slate-600">· {processingTime.toFixed(0)} ms</span>
              )}
            </p>
          </div>
        </div>

        <motion.button
          onClick={() => downloadReport(sessionId)}
          disabled={downloading}
          whileTap={!downloading ? { scale: 0.95 } : {}}
          className={`
            flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium
            transition-all duration-200
            ${downloading
              ? "bg-slate-800 text-slate-500 cursor-not-allowed"
              : "bg-satellite-600/20 hover:bg-satellite-600/30 text-satellite-400 border border-satellite-500/30 hover:border-satellite-400/50 cursor-pointer"
            }
          `}
        >
          {downloading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Downloading…</span>
            </>
          ) : (
            <>
              <FileDown className="w-4 h-4" />
              <span>Download PDF</span>
            </>
          )}
        </motion.button>
      </div>
    </div>
  );
}
