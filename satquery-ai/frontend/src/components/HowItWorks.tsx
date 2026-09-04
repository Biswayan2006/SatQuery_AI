"use client";

import { motion } from "framer-motion";
import { Upload, Brain, Layers, BarChart2, ArrowRight } from "lucide-react";

const STEPS = [
  {
    n: "01",
    icon: Upload,
    color: "text-satellite-400",
    bg: "bg-satellite-600/10 border-satellite-500/20",
    title: "Upload Imagery",
    desc: "Drop in a GeoTIFF, TIFF, PNG, or JPEG. Single images, bi-temporal pairs, or co-registered SAR-optical pairs are all supported.",
  },
  {
    n: "02",
    icon: Brain,
    color: "text-violet-400",
    bg: "bg-violet-600/10 border-violet-500/20",
    title: "Ask in Plain English",
    desc: "Type any natural language question — from 'describe the land cover' to 'what changed between these dates?' No special syntax needed.",
  },
  {
    n: "03",
    icon: Layers,
    color: "text-emerald-400",
    bg: "bg-emerald-600/10 border-emerald-500/20",
    title: "Agentic Routing",
    desc: "The controller classifies your task, validates the inputs, selects the right specialist model from the registry, and executes the workflow.",
  },
  {
    n: "04",
    icon: BarChart2,
    color: "text-amber-400",
    bg: "bg-amber-600/10 border-amber-500/20",
    title: "Evidence-Grounded Answer",
    desc: "Get a textual answer with confidence score, visual evidence (change map, grounding boxes, fusion map), and a downloadable PDF report.",
  },
];

export default function HowItWorks() {
  return (
    <section className="relative py-20 px-4 sm:px-6">
      {/* Glow */}
      <div className="absolute inset-0 bg-glow-radial pointer-events-none" />

      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-14"
        >
          <p className="text-xs text-satellite-500 font-bold tracking-widest uppercase mb-3">How it works</p>
          <h2 className="text-3xl sm:text-4xl font-bold text-white tracking-tight mb-4">
            Four steps from image to insight
          </h2>
          <p className="text-slate-500 text-sm sm:text-base max-w-xl mx-auto leading-relaxed">
            SatQuery AI handles everything from input validation to model selection and output synthesis automatically.
          </p>
        </motion.div>

        {/* Steps */}
        <div className="relative">
          {/* Connector line (desktop) */}
          <div className="hidden lg:block absolute top-12 left-[12.5%] right-[12.5%] h-px bg-gradient-to-r from-transparent via-satellite-600/30 to-transparent" />

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {STEPS.map((step, i) => {
              const Icon = step.icon;
              return (
                <motion.div
                  key={step.n}
                  initial={{ opacity: 0, y: 24 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.12 }}
                  className="relative flex flex-col items-center text-center gap-4 group"
                >
                  {/* Step icon */}
                  <div className={`relative w-16 h-16 rounded-2xl border flex items-center justify-center ${step.bg} ${step.color}`}>
                    <Icon className="w-7 h-7" />
                    <div className="absolute -top-2 -right-2 w-5 h-5 rounded-full bg-space-800 border border-white/[0.08] flex items-center justify-center">
                      <span className="text-[9px] font-bold text-slate-500">{step.n}</span>
                    </div>
                  </div>

                  {/* Arrow between steps (mobile/tablet) */}
                  {i < STEPS.length - 1 && (
                    <ArrowRight className="absolute top-6 -right-3 w-4 h-4 text-slate-700 hidden sm:block lg:hidden" />
                  )}

                  <div>
                    <p className={`text-sm font-bold mb-2 ${step.color}`}>{step.title}</p>
                    <p className="text-xs text-slate-500 leading-relaxed">{step.desc}</p>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>

        {/* Model table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2 }}
          className="mt-14 glass overflow-hidden"
        >
          <div className="px-5 py-4 border-b border-white/[0.04]">
            <p className="text-sm font-semibold text-slate-200">Specialist Model Registry</p>
            <p className="text-xs text-slate-500 mt-0.5">Models auto-selected by the agentic controller based on task and input</p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-white/[0.04]">
                  {["Module", "Base Model", "Task", "Input"].map(h => (
                    <th key={h} className="text-left px-5 py-3 text-[10px] font-bold text-slate-600 uppercase tracking-wider">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.03]">
                {[
                  ["VQA",            "blip-vqa-base",            "Visual Question Answering",  "Single image"],
                  ["Captioning",     "blip-image-captioning",    "Scene description",          "Single image"],
                  ["Grounding",      "owlvit-base-patch32",      "Object localisation",        "Single image"],
                  ["Change Detect.", "ResNet-50 (Siamese)",      "Temporal change analysis",   "Image pair"],
                  ["SAR Fusion",     "Dual ResNet-50 + MLP",     "Cross-modal analysis",       "SAR + Optical pair"],
                ].map(([mod, base, task, input], i) => (
                  <tr key={i} className="hover:bg-white/[0.02] transition-colors">
                    <td className="px-5 py-3 font-semibold text-satellite-400">{mod}</td>
                    <td className="px-5 py-3 text-slate-400 font-mono text-[11px]">{base}</td>
                    <td className="px-5 py-3 text-slate-500">{task}</td>
                    <td className="px-5 py-3 text-slate-600">{input}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
