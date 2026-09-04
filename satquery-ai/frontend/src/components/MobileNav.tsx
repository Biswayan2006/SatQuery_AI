"use client";

import { motion } from "framer-motion";
import { Upload, MessageSquare, BarChart2, Home } from "lucide-react";

interface Props {
  activeStep: number;
  onNav: (step: number) => void;
  hasImages: boolean;
  hasResult: boolean;
}

const TABS = [
  { id: 0, icon: Home,          label: "Home"    },
  { id: 1, icon: Upload,        label: "Upload"  },
  { id: 2, icon: MessageSquare, label: "Query"   },
  { id: 3, icon: BarChart2,     label: "Results" },
];

export default function MobileNav({ activeStep, onNav, hasImages, hasResult }: Props) {
  return (
    <nav className="fixed bottom-0 inset-x-0 z-50 safe-bottom md:hidden">
      <div className="glass border-t border-white/[0.06] rounded-none px-2 py-2">
        <div className="flex items-center justify-around">
          {TABS.map(tab => {
            const Icon = tab.icon;
            const active = activeStep === tab.id;
            const disabled =
              (tab.id === 2 && !hasImages) ||
              (tab.id === 3 && !hasResult);

            return (
              <button
                key={tab.id}
                onClick={() => !disabled && onNav(tab.id)}
                disabled={disabled}
                className={`relative flex flex-col items-center gap-1 px-4 py-1.5 rounded-xl transition-all duration-200 no-tap min-w-[60px]
                  ${active ? "text-satellite-400" : disabled ? "text-slate-700" : "text-slate-500 active:text-slate-300"}
                `}
              >
                {active && (
                  <motion.div
                    layoutId="mobile-nav-pill"
                    className="absolute inset-0 bg-satellite-600/15 border border-satellite-500/20 rounded-xl"
                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                  />
                )}
                <Icon className="w-5 h-5 relative z-10" />
                <span className="text-[10px] font-medium relative z-10">{tab.label}</span>

                {/* Dot indicators */}
                {tab.id === 1 && hasImages && !active && (
                  <span className="absolute top-1 right-3 w-1.5 h-1.5 rounded-full bg-emerald-400" />
                )}
                {tab.id === 3 && hasResult && !active && (
                  <span className="absolute top-1 right-3 w-1.5 h-1.5 rounded-full bg-satellite-400" />
                )}
              </button>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
