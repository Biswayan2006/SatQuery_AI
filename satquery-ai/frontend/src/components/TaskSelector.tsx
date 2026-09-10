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
  { id: "auto",               label: "Auto-detect",       shortLabel: "Auto",    icon: <Wand2 className="w-3.5 h-3.5" />,      color: "text-ink-soft",   border: "border-line",   requires: 1 },
  { id: "SINGLE_VQA",         label: "Visual Q&A",        shortLabel: "VQA",     icon: <Brain className="w-3.5 h-3.5" />,      color: "text-accent",     border: "border-accent/30",  requires: 1 },
  { id: "CAPTIONING",         label: "Scene Caption",     shortLabel: "Caption", icon: <Layers3 className="w-3.5 h-3.5" />,    color: "text-water",      border: "border-water/30",     requires: 1 },
  { id: "GROUNDING",          label: "Grounding",         shortLabel: "Ground",  icon: <Satellite className="w-3.5 h-3.5" />,  color: "text-veg",        border: "border-veg/30", requires: 1 },
  { id: "CHANGE_VQA",         label: "Change Q&A",        shortLabel: "Change",  icon: <Zap className="w-3.5 h-3.5" />,        color: "text-change",     border: "border-change/30",   requires: 2 },
  { id: "SAR_OPTICAL_FUSION", label: "SAR Fusion",        shortLabel: "SAR",     icon: <Globe2 className="w-3.5 h-3.5" />,     color: "text-sar",        border: "border-sar/30",    requires: 2 },
];

interface Props {
  value: TaskHint;
  onChange: (v: TaskHint) => void;
  numImages: number;
}

export default function TaskSelector({ value, onChange, numImages }: Props) {
  return (
    <div className="space-y-2">
      <p className="text-xs font-semibold text-ink-muted">Override task (optional)</p>
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
                  ? `${task.color} bg-raised ${task.border} shadow-sm font-semibold`
                  : disabled
                  ? "text-ink-faint border-line/40 bg-inset cursor-not-allowed opacity-40"
                  : `text-ink-muted border-line hover:border-line-strong hover:text-ink hover:bg-raised cursor-pointer`
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
              <span className="text-[10px] font-semibold leading-none">{task.shortLabel}</span>
              {task.requires === 2 && !disabled && (
                <span className="text-[9px] text-ink-faint leading-none">2 images</span>
              )}
            </motion.button>
          );
        })}
      </div>
    </div>
  );
}
