"use client";

import { motion } from "framer-motion";
import { Brain, Layers3, Satellite, Zap, Globe2, Wand2 } from "lucide-react";

export type TaskHint =
  | "auto"
  | "SINGLE_VQA"
  | "CAPTIONING"
  | "GROUNDING"
  | "CHANGE_VQA"
  | "CHANGE_DESCRIPTION"
  | "SAR_OPTICAL_FUSION";

const TASKS: { id: TaskHint; label: string; shortLabel: string; icon: React.ReactNode; color: string; border: string; requires: number }[] = [
  { id: "auto",               label: "Auto-detect",       shortLabel: "Auto",    icon: <Wand2 className="w-3.5 h-3.5" />,      color: "text-slate-300",   border: "border-slate-500/30",   requires: 1 },
  { id: "SINGLE_VQA",         label: "Visual Q&A",        shortLabel: "VQA",     icon: <Brain className="w-3.5 h-3.5" />,      color: "text-violet-400",  border: "border-violet-500/30",  requires: 1 },
  { id: "CAPTIONING",         label: "Scene Caption",     shortLabel: "Caption", icon: <Layers3 className="w-3.5 h-3.5" />,    color: "text-sky-400",     border: "border-sky-500/30",     requires: 1 },
  { id: "GROUNDING",          label: "Grounding",         shortLabel: "Ground",  icon: <Satellite className="w-3.5 h-3.5" />,  color: "text-emerald-400", border: "border-emerald-500/30", requires: 1 },
  { id: "CHANGE_VQA",         label: "Change Q&A",        shortLabel: "Change",  icon: <Zap className="w-3.5 h-3.5" />,        color: "text-amber-400",   border: "border-amber-500/30",   requires: 2 },
  { id: "SAR_OPTICAL_FUSION", label: "SAR Fusion",        shortLabel: "SAR",     icon: <Globe2 className="w-3.5 h-3.5" />,     color: "text-rose-400",    border: "border-rose-500/30",    requires: 2 },
];

interface Props {
  value: TaskHint;
  onChange: (v: TaskHint) => void;
  numImages: number;
}

export default function TaskSelector({ value, onChange, numImages }: Props) {
  return (
    <div className="space-y-2">
      <p className="text-[10px] text-slate-600 uppercase tracking-widest font-semibold">Override task (optional)</p>
      <div className="grid grid-cols-3 gap-1.5">
        {TASKS.map(task => {
          const disabled = task.requires > numImages && task.id !== "auto";
          const active = value === task.id;
          return (
            <motion.button
              key={task.id}
              onClick={() => !disabled && onChange(task.id)}
              disabled={disabled}
              whileTap={!disabled ? { scale: 0.95 } : {}}
              className={`
                relative flex flex-col items-center gap-1 py-2 px-1.5 rounded-xl border text-center
                transition-all duration-150 no-tap overflow-hidden
                ${active
                  ? `${task.color} bg-space-700/80 ${task.border} shadow-sm`
                  : disabled
                  ? "text-slate-700 border-white/[0.03] bg-space-900/40 cursor-not-allowed opacity-40"
                  : `text-slate-500 border-white/[0.04] hover:border-white/[0.1] hover:text-slate-300 hover:bg-space-800/40 cursor-pointer`
                }
              `}
            >
              {active && (
                <motion.div
                  layoutId="task-active"
                  className={`absolute inset-0 bg-current opacity-5 rounded-xl`}
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                />
              )}
              {task.icon}
              <span className="text-[9px] font-semibold leading-none">{task.shortLabel}</span>
              {task.requires === 2 && !disabled && (
                <span className="text-[8px] text-slate-700 leading-none">2 imgs</span>
              )}
            </motion.button>
          );
        })}
      </div>
    </div>
  );
}
