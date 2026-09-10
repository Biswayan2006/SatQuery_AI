"use client";
import { useState } from "react";
import { motion } from "framer-motion";
import { CheckCircle, Download, ChevronRight, Search } from "lucide-react";

const MOCK_HISTORY = [
  {
    id: "1",
    ts: "2026-09-04 14:32",
    project: "Urban Watch",
    query: "Detect changes in built-up area",
    tasks: ["Change Det", "VQA"],
    confidence: 0.92,
    status: "success",
  },
  {
    id: "2",
    ts: "2026-09-04 13:10",
    project: "Forest Monitor",
    query: "Describe land cover and vegetation",
    tasks: ["Captioning"],
    confidence: 0.87,
    status: "success",
  },
  {
    id: "3",
    ts: "2026-09-04 11:55",
    project: "Urban Watch",
    query: "Locate all water bodies",
    tasks: ["Grounding"],
    confidence: 0.79,
    status: "success",
  },
  {
    id: "4",
    ts: "2026-09-03 16:20",
    project: "Coastal Watch",
    query: "Use SAR and optical to identify flood extent",
    tasks: ["SAR Fusion", "VQA"],
    confidence: 0.84,
    status: "success",
  },
  {
    id: "5",
    ts: "2026-09-03 09:15",
    project: "Urban Watch",
    query: "What percentage of area is urban?",
    tasks: ["VQA"],
    confidence: 0.71,
    status: "success",
  },
];

// Canonical task: spectral hue (matches dashboard task encoding)
const taskHue = (t: string) =>
  t.includes("Change")
    ? "var(--change)"
    : t.includes("VQA")
    ? "var(--accent)"
    : t.includes("Caption")
    ? "var(--water)"
    : t.includes("Ground")
    ? "var(--veg)"
    : t.includes("SAR") || t.includes("Fusion")
    ? "var(--sar)"
    : "var(--accent)";

export default function HistoryPage() {
  const [search, setSearch] = useState("");

  const filtered = MOCK_HISTORY.filter(
    h =>
      h.query.toLowerCase().includes(search.toLowerCase()) ||
      h.project.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="max-w-[1400px] mx-auto space-y-5 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-lg font-bold text-ink">Execution History</h1>
          <p className="text-xs text-ink-muted mt-0.5">Auditable trail of all agent executions</p>
        </div>
        <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-raised border border-line text-xs font-semibold text-ink-soft hover:text-ink hover:bg-inset transition-all no-tap">
          <Download className="w-3.5 h-3.5" />Export CSV
        </button>
      </div>

      {/* Filters */}
      <div className="card p-4 flex flex-wrap gap-3">
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-raised border border-line flex-1 min-w-[200px]">
          <Search className="w-3.5 h-3.5 text-ink-muted flex-shrink-0" />
          <input
            type="text"
            placeholder="Search queries or projects…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="bg-transparent text-xs text-ink-soft placeholder:text-ink-faint outline-none w-full"
          />
        </div>
        <select className="px-3 py-2 rounded-lg bg-raised border border-line text-xs text-ink-muted outline-none cursor-pointer">
          <option>All Modalities</option>
          <option>Optical</option>
          <option>SAR</option>
          <option>Multispectral</option>
        </select>
        <select className="px-3 py-2 rounded-lg bg-raised border border-line text-xs text-ink-muted outline-none cursor-pointer">
          <option>All Tasks</option>
          <option>VQA</option>
          <option>Change Detection</option>
          <option>Grounding</option>
          <option>SAR Fusion</option>
        </select>
      </div>

      {/* Table */}
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-line">
                {["Timestamp", "Project", "Query", "Tasks", "Confidence", "Status", ""].map(h => (
                  <th
                    key={h}
                    className="text-left px-4 py-3 text-[10px] font-semibold text-ink-muted whitespace-nowrap"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {filtered.map((row, i) => (
                <motion.tr
                  key={row.id}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="hover:bg-raised transition-colors cursor-pointer group"
                >
                  <td className="px-4 py-3.5 text-xs text-ink-muted font-mono whitespace-nowrap">
                    {row.ts}
                  </td>
                  <td className="px-4 py-3.5 text-xs font-semibold text-accent whitespace-nowrap">
                    {row.project}
                  </td>
                  <td className="px-4 py-3.5 text-xs text-ink-soft max-w-[260px]">
                    <p className="truncate">{row.query}</p>
                  </td>
                  <td className="px-4 py-3.5">
                    <div className="flex flex-wrap gap-1">
                      {row.tasks.map(t => {
                        const hue = taskHue(t);
                        return (
                          <span
                            key={t}
                            className="px-2 py-0.5 rounded-full text-[10px] font-bold"
                            style={{
                              color: hue,
                              background: `color-mix(in srgb, ${hue} 13%, transparent)`,
                              border: `1px solid color-mix(in srgb, ${hue} 32%, transparent)`,
                            }}
                          >
                            {t}
                          </span>
                        );
                      })}
                    </div>
                  </td>
                  <td className="px-4 py-3.5">
                    <span
                      className={`text-xs font-bold ${
                        row.confidence >= 0.85
                          ? "text-veg"
                          : row.confidence >= 0.7
                          ? "text-warning"
                          : "text-danger"
                      }`}
                    >
                      {(row.confidence * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="px-4 py-3.5">
                    <div className="flex items-center gap-1.5">
                      <CheckCircle className="w-3.5 h-3.5 text-veg" />
                      <span className="text-xs text-veg font-medium">Success</span>
                    </div>
                  </td>
                  <td className="px-4 py-3.5">
                    <ChevronRight className="w-4 h-4 text-ink-faint group-hover:text-ink-muted transition-colors" />
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
