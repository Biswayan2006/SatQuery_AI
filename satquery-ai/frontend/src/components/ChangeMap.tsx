"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Map, AlertTriangle, TrendingUp } from "lucide-react";

interface Props {
  changeMapB64: string;
  changePercentage?: number | null;
  className?: string;
}

export default function ChangeMap({ changeMapB64, changePercentage, className = "" }: Props) {
  const [hovered, setHovered] = useState(false);

  const level =
    changePercentage == null
      ? null
      : changePercentage < 2
      ? { label: "Minimal Change", color: "text-emerald-400", bg: "bg-emerald-900/30 border-emerald-700/40" }
      : changePercentage < 10
      ? { label: "Minor Change", color: "text-sky-400", bg: "bg-sky-900/30 border-sky-700/40" }
      : changePercentage < 30
      ? { label: "Moderate Change", color: "text-amber-400", bg: "bg-amber-900/30 border-amber-700/40" }
      : changePercentage < 60
      ? { label: "Significant Change", color: "text-orange-400", bg: "bg-orange-900/30 border-orange-700/40" }
      : { label: "Extensive Change", color: "text-red-400", bg: "bg-red-900/30 border-red-700/40" };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={`glass-card overflow-hidden ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700/40">
        <div className="flex items-center gap-2">
          <Map className="w-4 h-4 text-amber-400" />
          <h3 className="text-sm font-semibold text-slate-300">Change Detection Map</h3>
        </div>
        {changePercentage != null && level && (
          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs border ${level.bg}`}>
            <TrendingUp className={`w-3 h-3 ${level.color}`} />
            <span className={level.color}>{level.label}</span>
          </div>
        )}
      </div>

      {/* Map image */}
      <div
        className="relative bg-slate-900/60 cursor-zoom-in"
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        <img
          src={`data:image/png;base64,${changeMapB64}`}
          alt="Change detection map"
          className="w-full object-contain max-h-72"
        />

        {/* Hover overlay with legend */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: hovered ? 1 : 0 }}
          className="absolute inset-0 bg-black/60 flex items-end p-3 pointer-events-none"
        >
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-sm bg-red-500" />
              <span className="text-slate-300">Changed</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-sm bg-emerald-500" />
              <span className="text-slate-300">Unchanged</span>
            </div>
            {changePercentage != null && (
              <div className="ml-auto flex items-center gap-1">
                <AlertTriangle className="w-3 h-3 text-amber-400" />
                <span className="text-amber-300 font-semibold">{changePercentage.toFixed(1)}% changed</span>
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* Stats bar */}
      {changePercentage != null && (
        <div className="px-4 py-3 border-t border-slate-700/40 bg-slate-900/40">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs text-slate-500">Changed area</span>
            <span className={`text-xs font-semibold ${level?.color ?? "text-slate-300"}`}>
              {changePercentage.toFixed(1)}%
            </span>
          </div>
          <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(changePercentage, 100)}%` }}
              transition={{ duration: 0.8, ease: "easeOut", delay: 0.2 }}
              className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-red-500"
            />
          </div>
          <div className="flex justify-between mt-1">
            <span className="text-[10px] text-slate-600 font-mono">0%</span>
            <span className="text-[10px] text-slate-600 font-mono">100%</span>
          </div>
        </div>
      )}
    </motion.div>
  );
}
