"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Map, AlertTriangle, TrendingUp } from "lucide-react";

const mix = (v: string, pct: number) => `color-mix(in srgb, ${v} ${pct}%, transparent)`;

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
      ? { label: "Minimal change", color: "var(--veg)" }
      : changePercentage < 10
      ? { label: "Minor change", color: "var(--water)" }
      : changePercentage < 30
      ? { label: "Moderate change", color: "var(--sar)" }
      : changePercentage < 60
      ? { label: "Significant change", color: "var(--bare)" }
      : { label: "Extensive change", color: "var(--change)" };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={`overflow-hidden ${className}`}
      style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", borderRadius: "10px" }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-3"
        style={{ borderBottom: "1px solid var(--border)" }}
      >
        <div className="flex items-center gap-2">
          <Map className="w-4 h-4" style={{ color: "var(--change)" }} />
          <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Change Detection Map</h3>
        </div>
        {changePercentage != null && level && (
          <div
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs"
            style={{ background: mix(level.color, 13), border: `1px solid ${mix(level.color, 32)}` }}
          >
            <TrendingUp className="w-3 h-3" style={{ color: level.color }} />
            <span style={{ color: level.color }}>{level.label}</span>
          </div>
        )}
      </div>

      {/* Map image */}
      <div
        className="relative"
        style={{ background: "var(--bg-inset)" }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`data:image/png;base64,${changeMapB64}`}
          alt="Change detection map"
          className="w-full object-contain max-h-72"
        />

        {/* Hover overlay with legend — dark scrim over imagery in both themes */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: hovered ? 1 : 0 }}
          className="absolute inset-0 flex items-end p-3 pointer-events-none"
          style={{ background: "rgba(6,10,14,0.62)" }}
        >
          <div className="flex items-center gap-4 text-xs w-full">
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-sm" style={{ background: "var(--change)" }} />
              <span style={{ color: "#E9F1F7" }}>Changed</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-sm" style={{ background: "var(--veg)" }} />
              <span style={{ color: "#E9F1F7" }}>Unchanged</span>
            </div>
            {changePercentage != null && (
              <div className="ml-auto flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" style={{ color: "#E3A93C" }} />
                <span className="font-semibold" style={{ color: "#E3A93C" }}>
                  {changePercentage.toFixed(1)}% changed
                </span>
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* Stats bar */}
      {changePercentage != null && (
        <div
          className="px-4 py-3"
          style={{ borderTop: "1px solid var(--border)", background: "var(--bg-inset)" }}
        >
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>Changed area</span>
            <span className="text-xs font-semibold" style={{ color: level?.color ?? "var(--text-secondary)" }}>
              {changePercentage.toFixed(1)}%
            </span>
          </div>
          <div
            className="h-1.5 rounded-full overflow-hidden"
            style={{ background: "var(--border-strong)" }}
          >
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(changePercentage, 100)}%` }}
              transition={{ duration: 0.8, ease: "easeOut", delay: 0.2 }}
              className="h-full rounded-full"
              style={{ background: "linear-gradient(to right, var(--veg), var(--change))" }}
            />
          </div>
          <div className="flex justify-between mt-1">
            <span className="text-[10px] font-mono" style={{ color: "var(--text-faint)" }}>0%</span>
            <span className="text-[10px] font-mono" style={{ color: "var(--text-faint)" }}>100%</span>
          </div>
        </div>
      )}
    </motion.div>
  );
}
