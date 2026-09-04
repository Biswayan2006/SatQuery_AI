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
          <h1 className="text-lg font-bold text-white">Execution History</h1>
          <p className="text-xs text-gray-500 mt-0.5">Auditable trail of all agent executions</p>
        </div>
        <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-elevated border border-white/[0.06] text-xs font-semibold text-gray-300 hover:text-white hover:bg-elevated/80 transition-all no-tap">
          <Download className="w-3.5 h-3.5" />Export CSV
        </button>
      </div>

      {/* Filters */}
      <div className="card p-4 flex flex-wrap gap-3">
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-elevated border border-white/[0.06] flex-1 min-w-[200px]">
          <Search className="w-3.5 h-3.5 text-gray-500 flex-shrink-0" />
          <input
            type="text"
            placeholder="Search queries or projects…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="bg-transparent text-xs text-gray-300 placeholder-gray-600 outline-none w-full"
          />
        </div>
        <select className="px-3 py-2 rounded-lg bg-elevated border border-white/[0.06] text-xs text-gray-400 outline-none cursor-pointer">
          <option>All Modalities</option>
          <option>Optical</option>
          <option>SAR</option>
          <option>Multispectral</option>
        </select>
        <select className="px-3 py-2 rounded-lg bg-elevated border border-white/[0.06] text-xs text-gray-400 outline-none cursor-pointer">
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
              <tr className="border-b border-white/[0.06]">
                {["Timestamp", "Project", "Query", "Tasks", "Confidence", "Status", ""].map(h => (
                  <th
                    key={h}
                    className="text-left px-4 py-3 text-[10px] font-bold text-gray-500 uppercase tracking-wider whitespace-nowrap"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04]">
              {filtered.map((row, i) => (
                <motion.tr
                  key={row.id}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="hover:bg-white/[0.02] transition-colors cursor-pointer group"
                >
                  <td className="px-4 py-3.5 text-xs text-gray-400 font-mono whitespace-nowrap">
                    {row.ts}
                  </td>
                  <td className="px-4 py-3.5 text-xs font-semibold text-primary whitespace-nowrap">
                    {row.project}
                  </td>
                  <td className="px-4 py-3.5 text-xs text-gray-300 max-w-[260px]">
                    <p className="truncate">{row.query}</p>
                  </td>
                  <td className="px-4 py-3.5">
                    <div className="flex flex-wrap gap-1">
                      {row.tasks.map(t => (
                        <span
                          key={t}
                          className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-primary/15 text-primary border border-primary/20"
                        >
                          {t}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="px-4 py-3.5">
                    <span
                      className={`text-xs font-bold ${
                        row.confidence >= 0.85
                          ? "text-green-400"
                          : row.confidence >= 0.7
                          ? "text-yellow-400"
                          : "text-red-400"
                      }`}
                    >
                      {(row.confidence * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="px-4 py-3.5">
                    <div className="flex items-center gap-1.5">
                      <CheckCircle className="w-3.5 h-3.5 text-green-400" />
                      <span className="text-xs text-green-400 font-medium">Success</span>
                    </div>
                  </td>
                  <td className="px-4 py-3.5">
                    <ChevronRight className="w-4 h-4 text-gray-700 group-hover:text-gray-400 transition-colors" />
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
