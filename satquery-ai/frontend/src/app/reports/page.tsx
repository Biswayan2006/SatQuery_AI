"use client";
import { motion } from "framer-motion";
import { FileText, Download, Eye, Map, Calendar, TrendingUp } from "lucide-react";

const REPORTS = [
  {
    id: "1",
    title: "Urban_Watch_Change_2026-09-04",
    date: "Sep 4, 2026 14:32",
    query: "Detect changes in built-up area",
    confidence: 0.92,
    task: "Change Detection",
    hasMap: true,
  },
  {
    id: "2",
    title: "Forest_Monitor_Caption_2026-09-04",
    date: "Sep 4, 2026 13:10",
    query: "Describe land cover and vegetation",
    confidence: 0.87,
    task: "Scene Caption",
    hasMap: false,
  },
  {
    id: "3",
    title: "Coastal_SAR_Fusion_2026-09-03",
    date: "Sep 3, 2026 16:20",
    query: "Identify flood extent using SAR and optical",
    confidence: 0.84,
    task: "SAR Fusion",
    hasMap: true,
  },
];

const TASK_COLOR: Record<string, string> = {
  "Change Detection": "change",
  "Scene Caption":    "water",
  "SAR Fusion":       "bare",
};

export default function ReportsPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

  return (
    <div className="max-w-[1400px] mx-auto space-y-5 animate-fade-in">
      <div>
        <h1 className="text-lg font-bold text-ink">Generated Reports</h1>
        <p className="text-xs text-ink-muted mt-0.5">
          Download and view analysis reports with visual evidence
        </p>
      </div>

      <div className="space-y-4">
        {REPORTS.map((r, i) => (
          <motion.div
            key={r.id}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.08 }}
            className="card p-5 flex flex-col sm:flex-row gap-4"
          >
            {/* Thumbnail */}
            <div className="w-full sm:w-28 h-20 sm:h-28 rounded-xl bg-inset border border-line flex items-center justify-center flex-shrink-0">
              {r.hasMap ? (
                <Map className="w-8 h-8 text-change" />
              ) : (
                <FileText className="w-8 h-8 text-water" />
              )}
            </div>

            {/* Info */}
            <div className="flex-1 min-w-0">
              <div className="flex items-start justify-between gap-3 flex-wrap mb-2">
                <div>
                  <p className="text-sm font-bold text-ink">{r.title}</p>
                  <div className="flex items-center gap-3 mt-1 flex-wrap">
                    <div className="flex items-center gap-1 text-[10px] text-ink-muted">
                      <Calendar className="w-3 h-3" />{r.date}
                    </div>
                    <span
                      className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                        TASK_COLOR[r.task] ? "" : "text-ink-muted bg-raised border-line"
                      }`}
                      style={
                        TASK_COLOR[r.task]
                          ? {
                              color: `var(--${TASK_COLOR[r.task]})`,
                              backgroundColor: `color-mix(in srgb, var(--${TASK_COLOR[r.task]}) 13%, transparent)`,
                              borderColor: `color-mix(in srgb, var(--${TASK_COLOR[r.task]}) 32%, transparent)`,
                            }
                          : undefined
                      }
                    >
                      {r.task}
                    </span>
                    <div className="flex items-center gap-1 text-[10px]">
                      <TrendingUp className="w-3 h-3 text-veg" />
                      <span className="text-veg">
                        {(r.confidence * 100).toFixed(0)}% confidence
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              <p className="text-xs text-ink-muted leading-relaxed mb-4 line-clamp-2">
                &ldquo;{r.query}&rdquo;
              </p>

              <div className="flex gap-2 flex-wrap">
                <a
                  href={`${apiBase}/api/report/${r.id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-accent-soft hover:bg-[color-mix(in_srgb,var(--accent)_20%,transparent)] border border-accent-border text-accent text-xs font-semibold transition-all no-tap"
                >
                  <Eye className="w-3.5 h-3.5" />View Online
                </a>
                <a
                  href={`${apiBase}/api/report/${r.id}`}
                  download
                  className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-raised hover:bg-inset border border-line text-ink-muted hover:text-ink text-xs font-semibold transition-all no-tap"
                >
                  <Download className="w-3.5 h-3.5" />Download PDF
                </a>
                {r.hasMap && (
                  <button className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-raised hover:bg-inset border border-line text-ink-muted hover:text-ink text-xs font-semibold transition-all no-tap">
                    <Map className="w-3.5 h-3.5" />Download GeoTIFF
                  </button>
                )}
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
