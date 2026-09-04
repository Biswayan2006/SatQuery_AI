"use client";

import { motion } from "framer-motion";
import { Database, Cpu, Globe2, Zap } from "lucide-react";
import type { HealthResponse } from "@/types";

const STATS = [
  { label: "Specialist Models",  value: "5",            icon: Database, color: "text-satellite-400" },
  { label: "Supported Tasks",    value: "6",            icon: Zap,      color: "text-violet-400"    },
  { label: "Image Formats",      value: "4",            icon: Globe2,   color: "text-emerald-400"   },
  { label: "RS-Adapted",         value: "BigEarthNet",  icon: Cpu,      color: "text-amber-400"     },
];

export default function StatsBanner({ health }: { health: HealthResponse | null }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      {STATS.map((s, i) => {
        const Icon = s.icon;
        return (
          <motion.div
            key={s.label}
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.08 }}
            className="glass glass-hover p-4 flex flex-col gap-2"
          >
            <div className={`flex items-center gap-2 ${s.color}`}>
              <Icon className="w-4 h-4" />
              <span className="text-lg font-bold tracking-tight">
                {s.label === "Specialist Models" && health
                  ? `${health.models_loaded}/5`
                  : s.value}
              </span>
            </div>
            <p className="text-[11px] text-slate-500 leading-tight">{s.label}</p>
          </motion.div>
        );
      })}
    </div>
  );
}
