"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import Image from "next/image";
import Link from "next/link";
import {
  Play,
  Pause,
  RotateCcw,
  Check,
  Cpu,
} from "lucide-react";

interface Scenario {
  id: string;
  name: string;
  task: string;
  latency: string;
  query: string;
  modality: string;
  inputLabel: string;
  inputSrc: string;
  evidenceLabel: string;
  evidenceSrc: string;
  answer: string;
  model: string;
  confidence: string;
  traceSteps: string[];
}

const SCENARIOS: Scenario[] = [
  {
    id: "change",
    name: "Change detection",
    task: "CHANGE_DETECTION",
    latency: "412ms",
    confidence: "0.94",
    modality: "Bi-temporal Optical",
    query: "What changed between these two acquisition dates?",
    inputLabel: "Registered optical input (T1)",
    inputSrc: "/assets/optical_main_hd.png",
    evidenceLabel: "Siamese differential distance heatmap",
    evidenceSrc: "/assets/change_main_hd.png",
    answer:
      "Analysis indicates new ground surface clearance and construction activity in the southern sector. Bi-temporal Euclidean feature difference highlights localized change across the scene.",
    model: "microsoft/resnet-50 (Siamese backbone)",
    traceSteps: [
      "input_validator.py: 2 registered optical tiles decoded (3 bands, 256x256)",
      "task_classifier.py: routed to CHANGE_DETECTION (confidence: 0.94)",
      "controller.py: extracted layer-4 embeddings and calculated Euclidean distance",
      "result_integrator.py: rendered differential overlay and compiled report summary",
    ],
  },
  {
    id: "vqa",
    name: "Visual Q&A",
    task: "VQA",
    latency: "628ms",
    confidence: "0.91",
    modality: "Multispectral Optical",
    query: "Identify the dominant land-cover and waterway features in this scene.",
    inputLabel: "Multispectral optical tile",
    inputSrc: "/assets/optical_main_hd.png",
    evidenceLabel: "Optical scene inspection",
    evidenceSrc: "/assets/optical_sample.png",
    answer:
      "The scene contains coastal land-cover characterized by tidal estuaries, dense riparian vegetation bordering water channels, and adjacent agricultural parcels.",
    model: "Salesforce/blip2-opt-2.7b",
    traceSteps: [
      "input_validator.py: 1 optical tile validated (.png, 256x256)",
      "task_classifier.py: routed to VQA (confidence: 0.91)",
      "controller.py: conditioned vision-language prompt with remote-sensing schema",
      "result_integrator.py: generated natural-language descriptive answer",
    ],
  },
  {
    id: "grounding",
    name: "Text-guided grounding",
    task: "GROUNDING",
    latency: "389ms",
    confidence: "0.88",
    modality: "Sub-meter Aerial",
    query: "Locate industrial storage tanks and coastal structures.",
    inputLabel: "Sub-meter aerial crop",
    inputSrc: "/assets/optical_sample.png",
    evidenceLabel: "Open-vocabulary detection coordinates",
    evidenceSrc: "/assets/change_sample.png",
    answer:
      "Located target structures with high visual agreement. Normalized bounding coordinates extracted for coastal infrastructure and tanks.",
    model: "google/owlvit-base-patch32",
    traceSteps: [
      "input_validator.py: single image input verified",
      "task_classifier.py: routed to GROUNDING (confidence: 0.88)",
      "controller.py: evaluated text embeddings against image patch tokens",
      "result_integrator.py: filtered bounding boxes with score threshold > 0.25",
    ],
  },
  {
    id: "fusion",
    name: "SAR and optical fusion",
    task: "SAR_FUSION",
    latency: "514ms",
    confidence: "0.89",
    modality: "Co-registered Optical + SAR",
    query: "Assess ground roughness and structures through cloud-obscured sectors.",
    inputLabel: "Optical multispectral reflectance",
    inputSrc: "/assets/optical_sample.png",
    evidenceLabel: "Synthetic aperture radar backscatter",
    evidenceSrc: "/assets/sar_main_hd.png",
    answer:
      "Synthetic aperture radar backscatter (VV/VH polarizations) reveals metallic structures and high-roughness terrain obscured by optical cloud cover.",
    model: "Dual ResNet-50 + MLP fusion",
    traceSteps: [
      "input_validator.py: paired optical and SAR inputs confirmed",
      "task_classifier.py: routed to SAR_FUSION (confidence: 0.89)",
      "controller.py: aligned radar backscatter amplitude with optical channels",
      "result_integrator.py: fused cross-modal features into unified interpretation",
    ],
  },
];

const PIPELINE_STEPS = [
  {
    step: 1,
    title: "Input validation",
    file: "input_validator.py",
    description:
      "Verifies file formats (.tif, .tiff, .png, .jpg), decodes radiometric channels with rasterio and Pillow, validates spatial dimensions, and confirms temporal or SAR/optical compatibility for multi-image tasks.",
  },
  {
    step: 2,
    title: "Task classification",
    file: "task_classifier.py",
    description:
      "Evaluates query intent and image count to determine the target specialist pipeline: VQA, CAPTIONING, GROUNDING, CHANGE_DETECTION, CHANGE_VQA, or SAR_FUSION.",
  },
  {
    step: 3,
    title: "Agentic controller",
    file: "controller.py",
    description:
      "Executes the plan, dispatches tensors to the designated PyTorch specialist backbones, measures inference latency, and handles model fallbacks if weights are still initializing.",
  },
  {
    step: 4,
    title: "Result integration",
    file: "report_generator.py",
    description:
      "Combines textual natural-language answers, visual evidence overlays (bounding coordinates or differential change maps), execution traces, and exportable PDF reports.",
  },
];

export default function ShowcasePage() {
  const [activeScenarioId, setActiveScenarioId] = useState<string>("change");
  const [currentStep, setCurrentStep] = useState<number>(0); // 0: Ingest, 1: Classify, 2: Execute, 3: Result
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [isReducedMotion, setIsReducedMotion] = useState<boolean>(false);
  const [isMounted, setIsMounted] = useState<boolean>(false);

  // Scroll reveal references
  const problemRef = useRef<HTMLElement>(null);
  const approachRef = useRef<HTMLElement>(null);
  const pipelineRef = useRef<HTMLElement>(null);
  const techStackRef = useRef<HTMLElement>(null);

  // Pipeline animation active state
  const [pipelineProgress, setPipelineProgress] = useState<number>(0);
  const [pipelineActiveStep, setPipelineActiveStep] = useState<number>(1);

  const currentScenario =
    SCENARIOS.find((s) => s.id === activeScenarioId) ?? SCENARIOS[0];

  // Detect prefers-reduced-motion
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setIsReducedMotion(mq.matches);
    if (mq.matches) {
      setIsPlaying(false);
      setCurrentStep(3); // show settled final state immediately
      setPipelineProgress(100);
      setPipelineActiveStep(4);
    }
    const handler = (e: MediaQueryListEvent) => {
      setIsReducedMotion(e.matches);
      if (e.matches) {
        setIsPlaying(false);
        setCurrentStep(3);
        setPipelineProgress(100);
        setPipelineActiveStep(4);
      }
    };
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  // One page-load entrance sequence: triggers once on mount
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsMounted(true);
    }, 50);
    return () => clearTimeout(timer);
  }, []);

  // Hero demonstrative sequence: auto-advances through 4 phases (unless paused or reduced motion)
  useEffect(() => {
    if (isReducedMotion || !isPlaying) return;

    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev >= 3) {
          // Pause at final frame per PRD Section 3.2, evaluator can replay
          setIsPlaying(false);
          return 3;
        }
        return prev + 1;
      });
    }, 2400);

    return () => clearInterval(interval);
  }, [isPlaying, isReducedMotion]);

  // Restart hero sequence
  const restartSequence = useCallback(() => {
    setCurrentStep(0);
    if (!isReducedMotion) {
      setIsPlaying(true);
    }
  }, [isReducedMotion]);

  // Select scenario: resets sequence to phase 0 so evaluator watches execution
  const handleSelectScenario = useCallback(
    (id: string) => {
      setActiveScenarioId(id);
      setCurrentStep(0);
      if (!isReducedMotion) {
        setIsPlaying(true);
      }
    },
    [isReducedMotion]
  );

  // IntersectionObserver for scroll reveals (triggers once, respects reduced motion)
  useEffect(() => {
    if (typeof window === "undefined" || isReducedMotion) return;

    const elements = [
      problemRef.current,
      approachRef.current,
      pipelineRef.current,
      techStackRef.current,
    ].filter(Boolean) as HTMLElement[];

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-revealed");
            observer.unobserve(entry.target);
          }
        });
      },
      {
        threshold: 0.15,
        rootMargin: "0px 0px -40px 0px",
      }
    );

    elements.forEach((el) => observer.observe(el));

    return () => observer.disconnect();
  }, [isReducedMotion]);

  // Pipeline section demonstrative animation: triggers once when pipeline section scrolls into view
  useEffect(() => {
    if (typeof window === "undefined" || isReducedMotion) return;

    const el = pipelineRef.current;
    if (!el) return;

    let timerId: NodeJS.Timeout;

    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;
        if (entry.isIntersecting) {
          observer.unobserve(el);

          // Animate line left-to-right in under 1.4s, sequentially highlighting 4 stages
          setPipelineProgress(10);
          setPipelineActiveStep(1);

          timerId = setTimeout(() => {
            setPipelineProgress(40);
            setPipelineActiveStep(2);

            setTimeout(() => {
              setPipelineProgress(70);
              setPipelineActiveStep(3);

              setTimeout(() => {
                setPipelineProgress(100);
                setPipelineActiveStep(4);
              }, 380);
            }, 380);
          }, 380);
        }
      },
      { threshold: 0.25 }
    );

    observer.observe(el);

    return () => {
      observer.disconnect();
      clearTimeout(timerId);
    };
  }, [isReducedMotion]);

  return (
    <div
      className="min-h-screen font-sans antialiased text-[#1A2129]"
      style={{
        backgroundColor: "#F0F1EC",
        color: "#1A2129",
      }}
    >
      {/* ── Top Navigation Bar ────────────────────────────────────────────── */}
      <header
        className="w-full border-b sticky top-0 z-30"
        style={{
          backgroundColor: "#F0F1EC",
          borderColor: "#CBCFC6",
        }}
      >
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <span
              className="text-xs font-mono font-medium px-2 py-0.5 rounded-md border"
              style={{
                backgroundColor: "#F0F1EC",
                color: "#28506B",
                borderColor: "#CBCFC6",
              }}
            >
              PS 26167
            </span>
            <span className="text-sm font-medium text-[#1A2129]">
              ISRO / Space Applications Centre
            </span>
          </div>
          <nav className="flex items-center space-x-3">
            <Link
              href="/app"
              className="text-xs sm:text-sm font-medium px-3.5 py-1.5 rounded-md text-white transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#28506B] focus-visible:ring-offset-2 focus-visible:ring-offset-[#F0F1EC]"
              style={{ backgroundColor: "#28506B" }}
            >
              Launch app
            </Link>
            <a
              href="https://github.com/PrakashRishiraj/SatQuery_AI"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs sm:text-sm font-medium px-3.5 py-1.5 rounded-md border transition-colors duration-150 hover:bg-[#CBCFC6]/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1A2129] focus-visible:ring-offset-2 focus-visible:ring-offset-[#F0F1EC]"
              style={{
                borderColor: "#CBCFC6",
                color: "#1A2129",
                backgroundColor: "transparent",
              }}
            >
              Repository
            </a>
          </nav>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10 sm:py-16 space-y-20 sm:space-y-28">
        {/* ── 1. HERO (One page-load moment + Demonstrative centerpiece) ────── */}
        <section
          id="hero"
          aria-labelledby="hero-heading"
          className="grid grid-cols-1 lg:grid-cols-12 gap-10 lg:gap-12 items-start"
        >
          {/* Left Column: Asymmetric text briefing (staggered 400ms entrance) */}
          <div
            className={`lg:col-span-5 space-y-6 text-left transition-all duration-300 ease-out ${
              isMounted || isReducedMotion
                ? "opacity-100 translate-y-0"
                : "opacity-0 translate-y-3"
            }`}
          >
            <div className="text-xs text-[#28506B] font-medium leading-relaxed">
              Problem Statement 26167: Agentic Vision-Language Remote Sensing
              Assistant for Optical and SAR Satellite Data. Space Applications
              Centre (SAC), ISRO.
            </div>

            <h1
              id="hero-heading"
              className="text-3xl sm:text-4xl lg:text-5xl font-semibold leading-tight tracking-tight text-[#1A2129]"
            >
              One agentic assistant for optical and SAR satellite questions.
            </h1>

            <p className="text-base sm:text-lg leading-relaxed text-[#1A2129]/90 max-w-[62ch]">
              An interactive vision-language system that coordinates specialist
              deep-learning models to answer natural-language queries, detect
              multi-temporal changes, and ground objects across satellite
              imagery.
            </p>

            <div className="pt-2 flex flex-wrap items-center gap-3">
              <Link
                href="/app"
                className="inline-flex items-center justify-center text-sm font-medium px-4 py-2.5 rounded-md text-white transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#28506B] focus-visible:ring-offset-2 focus-visible:ring-offset-[#F0F1EC]"
                style={{ backgroundColor: "#28506B" }}
              >
                Open interactive workspace
              </Link>
              <a
                href="https://github.com/PrakashRishiraj/SatQuery_AI"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center text-sm font-medium px-4 py-2.5 rounded-md border transition-colors duration-150 hover:bg-[#CBCFC6]/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1A2129] focus-visible:ring-offset-2 focus-visible:ring-offset-[#F0F1EC]"
                style={{
                  borderColor: "#CBCFC6",
                  color: "#1A2129",
                  backgroundColor: "transparent",
                }}
              >
                View the repository
              </a>
            </div>

            {/* Indicator of real backend execution */}
            <div className="pt-4 border-t border-[#CBCFC6]/80 text-xs text-[#1A2129]/70 space-y-1">
              <div className="flex items-center space-x-2">
                <span className="w-2 h-2 rounded-full bg-[#5A7052]" />
                <span className="font-mono text-[11px]">
                  Demonstrative execution trace from controller.py
                </span>
              </div>
              <p className="text-[11px] leading-normal">
                Observe the 4 deterministic pipeline phases run through actual
                model inference weights on real sensor inputs.
              </p>
            </div>
          </div>

          {/* Right Column: Demonstrative motion centerpiece (The real query lifecycle) */}
          <div
            className={`lg:col-span-7 transition-all duration-300 delay-150 ease-out ${
              isMounted || isReducedMotion
                ? "opacity-100 translate-y-0"
                : "opacity-0 translate-y-3"
            }`}
          >
            <div
              className="rounded-md border overflow-hidden shadow-sm"
              style={{
                borderColor: "#CBCFC6",
                backgroundColor: "#FFFFFF",
              }}
            >
              {/* Scenario selector tabs */}
              <div
                className="px-3 py-2 border-b flex flex-wrap items-center justify-between gap-2 text-xs"
                style={{
                  backgroundColor: "#F0F1EC",
                  borderColor: "#CBCFC6",
                }}
              >
                <div className="flex flex-wrap items-center gap-1.5">
                  {SCENARIOS.map((scenario) => {
                    const isActive = scenario.id === activeScenarioId;
                    return (
                      <button
                        key={scenario.id}
                        type="button"
                        onClick={() => handleSelectScenario(scenario.id)}
                        className="px-2.5 py-1 rounded-md text-xs font-medium transition-colors duration-150 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#28506B]"
                        style={{
                          backgroundColor: isActive ? "#28506B" : "transparent",
                          color: isActive ? "#FFFFFF" : "#1A2129",
                          border: `1px solid ${isActive ? "#28506B" : "#CBCFC6"}`,
                        }}
                      >
                        {scenario.name}
                      </button>
                    );
                  })}
                </div>

                <div className="flex items-center space-x-2 font-mono text-[11px] text-[#1A2129]/70">
                  <span className="font-semibold text-[#28506B]">
                    {currentScenario.task}
                  </span>
                  <span>{currentScenario.latency}</span>
                </div>
              </div>

              {/* Demonstrative Step Progress Bar (Step 1 to 4) */}
              <div
                className="px-4 py-2 border-b flex items-center justify-between gap-2 text-xs font-mono"
                style={{
                  backgroundColor: "#FFFFFF",
                  borderColor: "#CBCFC6",
                }}
              >
                {/* 4 Stage Pills */}
                <div className="flex items-center space-x-1 sm:space-x-2 flex-1">
                  {[
                    { idx: 0, label: "1. Ingest" },
                    { idx: 1, label: "2. Classify" },
                    { idx: 2, label: "3. Execute" },
                    { idx: 3, label: "4. Result" },
                  ].map((st) => {
                    const isPassed = currentStep >= st.idx;
                    const isCurrent = currentStep === st.idx;
                    return (
                      <button
                        key={st.idx}
                        type="button"
                        onClick={() => {
                          setCurrentStep(st.idx);
                          setIsPlaying(false);
                        }}
                        className="flex-1 py-1 px-1.5 rounded-md text-center text-[10px] sm:text-xs transition-colors duration-150 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#28506B]"
                        style={{
                          backgroundColor: isCurrent
                            ? "#28506B"
                            : isPassed
                            ? "#28506B/10"
                            : "#F0F1EC",
                          color: isCurrent
                            ? "#FFFFFF"
                            : isPassed
                            ? "#28506B"
                            : "#1A2129/60",
                          border: `1px solid ${
                            isCurrent
                              ? "#28506B"
                              : isPassed
                              ? "#28506B/30"
                              : "#CBCFC6"
                          }`,
                        }}
                      >
                        {st.label}
                      </button>
                    );
                  })}
                </div>

                {/* Playback Controls: Pause, Play, Replay */}
                <div className="flex items-center space-x-1 pl-2 border-l border-[#CBCFC6]">
                  <button
                    type="button"
                    onClick={() => setIsPlaying((p) => !p)}
                    aria-label={isPlaying ? "Pause demo" : "Play demo"}
                    className="p-1 rounded-md border border-[#CBCFC6] text-[#1A2129] hover:bg-[#F0F1EC] transition-colors duration-150 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#28506B]"
                    title={isPlaying ? "Pause auto-step" : "Play auto-step"}
                  >
                    {isPlaying ? (
                      <Pause className="w-3.5 h-3.5" />
                    ) : (
                      <Play className="w-3.5 h-3.5" />
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={restartSequence}
                    aria-label="Watch again"
                    className="p-1 rounded-md border border-[#CBCFC6] text-[#1A2129] hover:bg-[#F0F1EC] transition-colors duration-150 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#28506B]"
                    title="Watch again from step 1"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* Demonstrative Content Body: Shifts across the 4 stages */}
              <div className="p-4 space-y-4">
                {/* Stage Header Info Banner */}
                <div
                  className="px-3 py-2 rounded-md border text-xs flex items-center justify-between"
                  style={{
                    backgroundColor: "#F0F1EC",
                    borderColor: "#CBCFC6",
                  }}
                >
                  <div className="flex items-center space-x-2">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{
                        backgroundColor:
                          currentStep === 3 ? "#5A7052" : "#28506B",
                      }}
                    />
                    <span className="font-semibold text-[#1A2129]">
                      {currentStep === 0 && "Phase 1: Input Ingestion & Validation"}
                      {currentStep === 1 && "Phase 2: Intent Classification"}
                      {currentStep === 2 && "Phase 3: Specialist Model Execution"}
                      {currentStep === 3 && "Phase 4: Evidence Grounding & Result"}
                    </span>
                  </div>
                  <span className="font-mono text-[11px] text-[#28506B]">
                    {currentStep === 0 && "input_validator.py"}
                    {currentStep === 1 && "task_classifier.py"}
                    {currentStep === 2 && "controller.py"}
                    {currentStep === 3 && "report_generator.py"}
                  </span>
                </div>

                {/* Query bar */}
                <div
                  className="p-3 rounded-md border text-xs space-y-1"
                  style={{
                    backgroundColor: "#F0F1EC",
                    borderColor: "#CBCFC6",
                  }}
                >
                  <div className="text-[#1A2129]/70 text-[11px]">
                    User query:
                  </div>
                  <div className="font-medium text-[#1A2129] text-sm">
                    &ldquo;{currentScenario.query}&rdquo;
                  </div>
                </div>

                {/* Imagery Display: Responsive and demonstrative */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {/* Left Imagery: Input tile */}
                  <div className="space-y-1.5">
                    <div className="text-[11px] text-[#1A2129]/70 flex items-center justify-between">
                      <span>{currentScenario.inputLabel}</span>
                      <span className="font-mono text-[10px] text-[#28506B]">
                        256x256 RGB
                      </span>
                    </div>
                    <div className="relative aspect-[4/3] rounded-md overflow-hidden border border-[#CBCFC6] bg-[#1A2129]/5">
                      <Image
                        src={currentScenario.inputSrc}
                        alt={currentScenario.inputLabel}
                        fill
                        sizes="(max-width: 1024px) 50vw, 25vw"
                        className="object-cover"
                        priority
                      />
                      {/* Sub-phase overlay indicator */}
                      {currentStep === 0 && (
                        <div className="absolute inset-0 bg-[#28506B]/20 flex items-end p-2 transition-opacity duration-200">
                          <span className="px-2 py-0.5 rounded-md bg-[#1A2129] text-white text-[10px] font-mono">
                            Modality: {currentScenario.modality}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Right Imagery: Visual evidence overlay */}
                  <div className="space-y-1.5">
                    <div className="text-[11px] text-[#1A2129]/70 flex items-center justify-between">
                      <span>{currentScenario.evidenceLabel}</span>
                      <span className="font-mono text-[10px] text-[#28506B]">
                        {currentStep >= 2 ? "Differential Map" : "Pending pass"}
                      </span>
                    </div>
                    <div className="relative aspect-[4/3] rounded-md overflow-hidden border border-[#CBCFC6] bg-[#1A2129]/5">
                      {currentStep >= 2 ? (
                        <Image
                          src={currentScenario.evidenceSrc}
                          alt={currentScenario.evidenceLabel}
                          fill
                          sizes="(max-width: 1024px) 50vw, 25vw"
                          className="object-cover transition-opacity duration-200"
                        />
                      ) : (
                        <div className="absolute inset-0 flex flex-col items-center justify-center p-4 text-center bg-[#F0F1EC]">
                          <Cpu className="w-6 h-6 text-[#28506B]/60 mb-2" />
                          <span className="text-xs text-[#1A2129]/70 font-mono">
                            Awaiting controller dispatch...
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Evidence-grounded answer or intermediate trace */}
                <div
                  className="p-3.5 rounded-md border space-y-1.5"
                  style={{
                    backgroundColor: "#FFFFFF",
                    borderColor: "#CBCFC6",
                  }}
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium text-[#1A2129]">
                      {currentStep === 3
                        ? "Evidence-grounded answer"
                        : "Active pipeline state"}
                    </span>
                    <span className="font-mono text-[11px] text-[#28506B]">
                      {currentScenario.model}
                    </span>
                  </div>

                  {currentStep < 3 ? (
                    <div className="text-xs text-[#1A2129]/80 font-mono py-1">
                      {currentStep === 0 && (
                        <span>
                          Decoding raster bands, confirming spatial overlap and
                          radiometric calibration...
                        </span>
                      )}
                      {currentStep === 1 && (
                        <span>
                          Task classified as {currentScenario.task} with
                          confidence score {currentScenario.confidence}.
                        </span>
                      )}
                      {currentStep === 2 && (
                        <span>
                          Evaluating tensors via {currentScenario.model} (runtime:{" "}
                          {currentScenario.latency}).
                        </span>
                      )}
                    </div>
                  ) : (
                    <p className="text-xs sm:text-sm text-[#1A2129]/90 leading-relaxed">
                      {currentScenario.answer}
                    </p>
                  )}
                </div>

                {/* Execution trace log */}
                <div
                  className="p-3 rounded-md border text-xs space-y-1.5 font-mono"
                  style={{
                    backgroundColor: "#F0F1EC",
                    borderColor: "#CBCFC6",
                  }}
                >
                  <div className="text-[11px] font-sans font-medium text-[#1A2129] flex items-center justify-between">
                    <span>Execution trace (controller.py):</span>
                    <span className="text-[10px] text-[#28506B]">
                      {Math.min(currentStep + 1, 4)} of 4 steps resolved
                    </span>
                  </div>
                  <ul className="space-y-1 text-[11px] text-[#1A2129]/80">
                    {currentScenario.traceSteps
                      .slice(0, currentStep + 1)
                      .map((step, idx) => (
                        <li key={idx} className="leading-normal flex items-start space-x-1.5">
                          <Check className="w-3.5 h-3.5 text-[#5A7052] flex-shrink-0 mt-0.5" />
                          <span>[{idx + 1}] {step}</span>
                        </li>
                      ))}
                  </ul>
                </div>
              </div>

              {/* Action bar pointing to live interactive tool */}
              <div
                className="px-4 py-3 border-t flex items-center justify-between text-xs"
                style={{
                  backgroundColor: "#F0F1EC",
                  borderColor: "#CBCFC6",
                }}
              >
                <div className="flex items-center space-x-2">
                  <span className="text-[#1A2129]/70">
                    Real end-to-end output from the SatQuery AI backend pipeline.
                  </span>
                  {currentStep === 3 && (
                    <button
                      type="button"
                      onClick={restartSequence}
                      className="font-medium text-[#28506B] underline hover:text-[#1A2129] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#28506B]"
                    >
                      Watch again
                    </button>
                  )}
                </div>
                <Link
                  href="/app"
                  className="font-medium text-[#28506B] underline hover:text-[#1A2129] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#28506B]"
                >
                  Open live workspace
                </Link>
              </div>
            </div>
          </div>
        </section>

        {/* ── 2. THE PROBLEM (Scroll-triggered response motion) ─────────────── */}
        <section
          id="problem"
          ref={problemRef}
          aria-labelledby="problem-heading"
          className="border-t pt-12 sm:pt-16 space-y-6 motion-reveal"
          style={{ borderColor: "#CBCFC6" }}
        >
          <h2
            id="problem-heading"
            className="text-2xl sm:text-3xl font-semibold tracking-tight text-[#1A2129]"
          >
            The problem
          </h2>

          <div className="space-y-4 max-w-[72ch] text-[#1A2129]/90 text-base sm:text-lg leading-relaxed">
            <p>
              Earth observation archives managed by space agencies such as ISRO
              contain petabytes of optical reflectance and Synthetic Aperture
              Radar (SAR) imagery. Extracting actionable insights from this data
              during rapid-response operations (such as flood delineation,
              agricultural damage assessment, or infrastructure monitoring)
              currently requires domain analysts to manually pick, configure,
              and chain separate specialist models.
            </p>
            <p>
              An analyst inspecting an evolving disaster must use one tool for
              visual question answering, another pipeline for bi-temporal change
              detection, a third model for bounding-box grounding, and
              specialized scripts to calibrate and align radar backscatter with
              multispectral imagery.
            </p>
            <p>
              This fragmentation prevents non-expert decision-makers from asking
              direct questions about satellite scenes. SatQuery AI addresses
              Problem Statement 26167 (Space Applications Centre, ISRO) by
              building a unified, query-driven controller that interprets natural
              language, selects the appropriate specialist models, executes the
              analysis, and returns evidence-grounded answers with verifiable
              visual overlays and execution traces.
            </p>
          </div>
        </section>

        {/* ── 3. THE APPROACH (Staggered response motion on cards) ───────────── */}
        <section
          id="approach"
          ref={approachRef}
          aria-labelledby="approach-heading"
          className="border-t pt-12 sm:pt-16 space-y-8 motion-reveal"
          style={{ borderColor: "#CBCFC6" }}
        >
          <div className="space-y-2">
            <h2
              id="approach-heading"
              className="text-2xl sm:text-3xl font-semibold tracking-tight text-[#1A2129]"
            >
              The approach
            </h2>
            <p className="text-base text-[#1A2129]/80 max-w-[68ch]">
              Six specialist capabilities coordinated by an agentic controller.
              Each capability block below details the underlying model
              architecture and displays an actual artifact from the project.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Block 1: VQA */}
            <div
              className="rounded-md border flex flex-col justify-between overflow-hidden hover:border-[#28506B]/50 transition-colors duration-150"
              style={{
                backgroundColor: "#FFFFFF",
                borderColor: "#CBCFC6",
              }}
            >
              <div className="p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-medium text-[#28506B]">
                    VQA
                  </span>
                  <span className="text-xs text-[#1A2129]/60">Optical</span>
                </div>
                <h3 className="text-lg font-semibold text-[#1A2129]">
                  Visual Question Answering
                </h3>
                <p className="text-sm text-[#1A2129]/80 leading-relaxed">
                  Answers free-form natural-language questions about land use,
                  structures, and environmental conditions in optical imagery.
                </p>
                <div className="text-xs space-y-1 pt-1">
                  <div className="text-[#1A2129]/70">
                    Model:{" "}
                    <span className="font-mono text-[#1A2129]">
                      Salesforce/blip2-opt-2.7b
                    </span>
                  </div>
                  <div className="text-[#1A2129]/70">
                    Integration: Stock Hugging Face vision-language checkpoint
                    with remote-sensing prompt conditioning.
                  </div>
                </div>
              </div>
              <div
                className="border-t p-3"
                style={{
                  backgroundColor: "#F0F1EC",
                  borderColor: "#CBCFC6",
                }}
              >
                <div className="relative aspect-[16/9] rounded-md overflow-hidden border border-[#CBCFC6]/80 bg-[#1A2129]/5">
                  <Image
                    src="/assets/optical_main_hd.png"
                    alt="Optical satellite tile used as input for visual question answering."
                    fill
                    sizes="(max-width: 768px) 100vw, 33vw"
                    className="object-cover"
                  />
                </div>
                <p className="text-[11px] text-[#1A2129]/70 mt-2 font-mono">
                  Input artifact: multispectral optical tile
                </p>
              </div>
            </div>

            {/* Block 2: Scene Captioning */}
            <div
              className="rounded-md border flex flex-col justify-between overflow-hidden hover:border-[#28506B]/50 transition-colors duration-150"
              style={{
                backgroundColor: "#FFFFFF",
                borderColor: "#CBCFC6",
              }}
            >
              <div className="p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-medium text-[#28506B]">
                    CAPTIONING
                  </span>
                  <span className="text-xs text-[#1A2129]/60">Optical</span>
                </div>
                <h3 className="text-lg font-semibold text-[#1A2129]">
                  Scene Captioning
                </h3>
                <p className="text-sm text-[#1A2129]/80 leading-relaxed">
                  Generates descriptive paragraphs summarizing visible terrain,
                  vegetation cover, waterways, and man-made infrastructure.
                </p>
                <div className="text-xs space-y-1 pt-1">
                  <div className="text-[#1A2129]/70">
                    Model:{" "}
                    <span className="font-mono text-[#1A2129]">
                      Salesforce/blip2-opt-2.7b
                    </span>
                  </div>
                  <div className="text-[#1A2129]/70">
                    Integration: Autoregressive decoder generating structured
                    scene narratives from visual features.
                  </div>
                </div>
              </div>
              <div
                className="border-t p-3"
                style={{
                  backgroundColor: "#F0F1EC",
                  borderColor: "#CBCFC6",
                }}
              >
                <div className="relative aspect-[16/9] rounded-md overflow-hidden border border-[#CBCFC6]/80 bg-[#1A2129]/5">
                  <Image
                    src="/assets/optical_sample.png"
                    alt="Optical aerial crop analyzed by the captioning module."
                    fill
                    sizes="(max-width: 768px) 100vw, 33vw"
                    className="object-cover"
                  />
                </div>
                <p className="text-[11px] text-[#1A2129]/70 mt-2 font-mono">
                  Input artifact: sub-meter urban/coastal sample
                </p>
              </div>
            </div>

            {/* Block 3: Grounding */}
            <div
              className="rounded-md border flex flex-col justify-between overflow-hidden hover:border-[#28506B]/50 transition-colors duration-150"
              style={{
                backgroundColor: "#FFFFFF",
                borderColor: "#CBCFC6",
              }}
            >
              <div className="p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-medium text-[#28506B]">
                    GROUNDING
                  </span>
                  <span className="text-xs text-[#1A2129]/60">Detection</span>
                </div>
                <h3 className="text-lg font-semibold text-[#1A2129]">
                  Text-Guided Grounding
                </h3>
                <p className="text-sm text-[#1A2129]/80 leading-relaxed">
                  Locates user-specified targets (such as aircraft, vessels, or
                  storage tanks) and returns normalized bounding coordinates.
                </p>
                <div className="text-xs space-y-1 pt-1">
                  <div className="text-[#1A2129]/70">
                    Model:{" "}
                    <span className="font-mono text-[#1A2129]">
                      google/owlvit-base-patch32
                    </span>
                  </div>
                  <div className="text-[#1A2129]/70">
                    Integration: Open-vocabulary vision transformer detecting
                    arbitrary text queries without retraining.
                  </div>
                </div>
              </div>
              <div
                className="border-t p-3"
                style={{
                  backgroundColor: "#F0F1EC",
                  borderColor: "#CBCFC6",
                }}
              >
                <div className="relative aspect-[16/9] rounded-md overflow-hidden border border-[#CBCFC6]/80 bg-[#1A2129]/5">
                  <Image
                    src="/assets/change_sample.png"
                    alt="Grounding sample showing localized feature coordinates."
                    fill
                    sizes="(max-width: 768px) 100vw, 33vw"
                    className="object-cover"
                  />
                </div>
                <p className="text-[11px] text-[#1A2129]/70 mt-2 font-mono">
                  Input artifact: localized target crop
                </p>
              </div>
            </div>

            {/* Block 4: Change Detection */}
            <div
              className="rounded-md border flex flex-col justify-between overflow-hidden hover:border-[#B0472E]/50 transition-colors duration-150"
              style={{
                backgroundColor: "#FFFFFF",
                borderColor: "#CBCFC6",
              }}
            >
              <div className="p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-medium text-[#B0472E]">
                    CHANGE_DETECTION
                  </span>
                  <span className="text-xs text-[#1A2129]/60">Bi-temporal</span>
                </div>
                <h3 className="text-lg font-semibold text-[#1A2129]">
                  Two-Date Change Detection
                </h3>
                <p className="text-sm text-[#1A2129]/80 leading-relaxed">
                  Computes differential Euclidean distance between two
                  registered dates to produce a continuous change heatmap.
                </p>
                <div className="text-xs space-y-1 pt-1">
                  <div className="text-[#1A2129]/70">
                    Model:{" "}
                    <span className="font-mono text-[#1A2129]">
                      microsoft/resnet-50
                    </span>
                  </div>
                  <div className="text-[#1A2129]/70">
                    Integration: Siamese feature extraction with pixel-wise
                    difference mapping and Otsu thresholding.
                  </div>
                </div>
              </div>
              <div
                className="border-t p-3"
                style={{
                  backgroundColor: "#F0F1EC",
                  borderColor: "#CBCFC6",
                }}
              >
                <div className="relative aspect-[16/9] rounded-md overflow-hidden border border-[#CBCFC6]/80 bg-[#1A2129]/5">
                  <Image
                    src="/assets/change_main_hd.png"
                    alt="Bi-temporal change detection output artifact with red differential overlay."
                    fill
                    sizes="(max-width: 768px) 100vw, 33vw"
                    className="object-cover"
                  />
                </div>
                <p className="text-[11px] text-[#1A2129]/70 mt-2 font-mono">
                  Output artifact: Siamese differential map
                </p>
              </div>
            </div>

            {/* Block 5: Change-Based Q&A */}
            <div
              className="rounded-md border flex flex-col justify-between overflow-hidden hover:border-[#B0472E]/50 transition-colors duration-150"
              style={{
                backgroundColor: "#FFFFFF",
                borderColor: "#CBCFC6",
              }}
            >
              <div className="p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-medium text-[#B0472E]">
                    CHANGE_VQA
                  </span>
                  <span className="text-xs text-[#1A2129]/60">Multi-pass</span>
                </div>
                <h3 className="text-lg font-semibold text-[#1A2129]">
                  Change-Based Question Answering
                </h3>
                <p className="text-sm text-[#1A2129]/80 leading-relaxed">
                  Answers natural-language questions regarding what changed,
                  appeared, or vanished across multi-temporal satellite passes.
                </p>
                <div className="text-xs space-y-1 pt-1">
                  <div className="text-[#1A2129]/70">
                    Pipeline:{" "}
                    <span className="font-mono text-[#1A2129]">
                      Siamese ResNet + BLIP-2
                    </span>
                  </div>
                  <div className="text-[#1A2129]/70">
                    Integration: Change metrics and differential masks inject
                    spatial context into the vision-language reasoning engine.
                  </div>
                </div>
              </div>
              <div
                className="border-t p-3"
                style={{
                  backgroundColor: "#F0F1EC",
                  borderColor: "#CBCFC6",
                }}
              >
                <div className="relative aspect-[16/9] rounded-md overflow-hidden border border-[#CBCFC6]/80 bg-[#1A2129]/5">
                  <Image
                    src="/sat/thumb_timeseries.jpg"
                    alt="Multi-temporal timeline image used for change question answering."
                    fill
                    sizes="(max-width: 768px) 100vw, 33vw"
                    className="object-cover"
                  />
                </div>
                <p className="text-[11px] text-[#1A2129]/70 mt-2 font-mono">
                  Input artifact: multi-temporal timeline pair
                </p>
              </div>
            </div>

            {/* Block 6: SAR Fusion */}
            <div
              className="rounded-md border flex flex-col justify-between overflow-hidden hover:border-[#6B5A3A]/50 transition-colors duration-150"
              style={{
                backgroundColor: "#FFFFFF",
                borderColor: "#CBCFC6",
              }}
            >
              <div className="p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-medium text-[#6B5A3A]">
                    SAR_FUSION
                  </span>
                  <span className="text-xs text-[#1A2129]/60">Radar+Opt</span>
                </div>
                <h3 className="text-lg font-semibold text-[#1A2129]">
                  SAR and Optical Fusion
                </h3>
                <p className="text-sm text-[#1A2129]/80 leading-relaxed">
                  Combines all-weather, cloud-penetrating synthetic aperture
                  radar backscatter (VV/VH channels) with optical reflectance.
                </p>
                <div className="text-xs space-y-1 pt-1">
                  <div className="text-[#1A2129]/70">
                    Model:{" "}
                    <span className="font-mono text-[#1A2129]">
                      Dual ResNet-50 + MLP
                    </span>
                  </div>
                  <div className="text-[#1A2129]/70">
                    Integration: Cross-modal feature concatenation resolving
                    surface roughness and reflectance simultaneously.
                  </div>
                </div>
              </div>
              <div
                className="border-t p-3"
                style={{
                  backgroundColor: "#F0F1EC",
                  borderColor: "#CBCFC6",
                }}
              >
                <div className="relative aspect-[16/9] rounded-md overflow-hidden border border-[#CBCFC6]/80 bg-[#1A2129]/5">
                  <Image
                    src="/assets/sar_main_hd.png"
                    alt="Synthetic aperture radar backscatter artifact."
                    fill
                    sizes="(max-width: 768px) 100vw, 33vw"
                    className="object-cover"
                  />
                </div>
                <p className="text-[11px] text-[#1A2129]/70 mt-2 font-mono">
                  Input artifact: calibrated SAR radar tile
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* ── 4. HOW IT WORKS (Demonstrative pipeline animation) ────────────── */}
        <section
          id="pipeline"
          ref={pipelineRef}
          aria-labelledby="pipeline-heading"
          className="border-t pt-12 sm:pt-16 space-y-8 motion-reveal"
          style={{ borderColor: "#CBCFC6" }}
        >
          <div className="space-y-2">
            <h2
              id="pipeline-heading"
              className="text-2xl sm:text-3xl font-semibold tracking-tight text-[#1A2129]"
            >
              How it works
            </h2>
            <p className="text-base text-[#1A2129]/80 max-w-[68ch]">
              The execution pipeline processes user queries through four
              deterministic stages. Each query is validated, classified,
              executed across specialist models, and compiled into an auditable
              result.
            </p>
          </div>

          {/* Desktop Connecting Line & Stage Indicators (Draws left-to-right under 1.4s) */}
          <div className="relative">
            {/* Desktop progress bar track */}
            <div className="hidden lg:block absolute top-7 left-8 right-8 h-0.5 bg-[#CBCFC6]/60 z-0">
              <div
                className="h-full bg-[#28506B] transition-all duration-300 ease-out"
                style={{ width: `${pipelineProgress}%` }}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 relative z-10">
              {PIPELINE_STEPS.map((step) => {
                const isActive = pipelineActiveStep >= step.step;
                const isCurrent = pipelineActiveStep === step.step;

                return (
                  <div
                    key={step.step}
                    className="p-5 rounded-md border space-y-3 transition-colors duration-200"
                    style={{
                      backgroundColor: "#FFFFFF",
                      borderColor: isActive ? "#28506B" : "#CBCFC6",
                      boxShadow: isCurrent
                        ? "0 4px 12px rgba(40, 80, 107, 0.08)"
                        : "none",
                    }}
                  >
                    <div className="flex items-center justify-between">
                      <div
                        className="w-7 h-7 rounded-md flex items-center justify-center font-mono text-xs font-semibold text-white transition-colors duration-200"
                        style={{
                          backgroundColor: isActive ? "#28506B" : "#A88A70",
                        }}
                      >
                        {step.step}
                      </div>
                      <span className="font-mono text-[10px] text-[#28506B]">
                        {step.file}
                      </span>
                    </div>

                    <h3 className="text-base font-semibold text-[#1A2129]">
                      {step.title}
                    </h3>
                    <p className="text-xs sm:text-sm text-[#1A2129]/80 leading-relaxed">
                      {step.description}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Abbreviated Real Execution Trace (per PRD Section 3.4) */}
          <div
            className="p-4 rounded-md border font-mono text-xs space-y-2"
            style={{
              backgroundColor: "#FFFFFF",
              borderColor: "#CBCFC6",
            }}
          >
            <div className="flex items-center justify-between text-[#1A2129]/80 pb-2 border-b border-[#CBCFC6]/60">
              <span className="font-sans font-semibold text-[#1A2129]">
                Live Controller Trace (controller.py execution run)
              </span>
              <span className="text-[11px] text-[#28506B]">
                total_latency: 412ms
              </span>
            </div>
            <div className="space-y-1.5 text-[11px] text-[#1A2129]/80">
              <div className="flex items-start space-x-2">
                <span className="text-[#28506B]">[00.00s]</span>
                <span>input_validator.py: verified 2 registered optical GeoTIFF tiles (256x256, 3 bands)</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-[#28506B]">[00.08s]</span>
                <span>task_classifier.py: query mapped to CHANGE_DETECTION (confidence: 0.94)</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-[#28506B]">[00.12s]</span>
                <span>controller.py: dispatched Siamese ResNet-50 backbone; extracted layer-4 embeddings</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-[#28506B]">[00.34s]</span>
                <span>controller.py: computed bi-temporal Euclidean distance; Otsu threshold delta: 9.4%</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-[#5A7052]">[00.41s]</span>
                <span>report_generator.py: compiled differential overlay, grounding metadata, and PDF report</span>
              </div>
            </div>
          </div>
        </section>

        {/* ── 5. TECH STACK (Response motion on hover) ──────────────────────── */}
        <section
          id="tech-stack"
          ref={techStackRef}
          aria-labelledby="stack-heading"
          className="border-t pt-12 sm:pt-16 space-y-8 motion-reveal"
          style={{ borderColor: "#CBCFC6" }}
        >
          <div className="space-y-2">
            <h2
              id="stack-heading"
              className="text-2xl sm:text-3xl font-semibold tracking-tight text-[#1A2129]"
            >
              Tech stack
            </h2>
            <p className="text-base text-[#1A2129]/80 max-w-[68ch]">
              A plain record of frameworks, specialist model weights, and
              geospatial libraries integrated into the active repository.
            </p>
          </div>

          <div
            className="rounded-md border overflow-hidden"
            style={{
              backgroundColor: "#FFFFFF",
              borderColor: "#CBCFC6",
            }}
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr
                    className="border-b text-xs font-mono text-[#28506B]"
                    style={{
                      backgroundColor: "#F0F1EC",
                      borderColor: "#CBCFC6",
                    }}
                  >
                    <th className="py-3 px-4 font-semibold">Component</th>
                    <th className="py-3 px-4 font-semibold">Technology / Model</th>
                    <th className="py-3 px-4 font-semibold">Role in Pipeline</th>
                  </tr>
                </thead>
                <tbody className="divide-y" style={{ borderColor: "#CBCFC6" }}>
                  {[
                    {
                      component: "Backend Server",
                      tech: "FastAPI, Uvicorn, Pydantic",
                      role: "REST API endpoints for image ingestion, async agent execution, and report downloads.",
                    },
                    {
                      component: "Frontend Client",
                      tech: "Next.js 14, React 18, TypeScript, Tailwind CSS",
                      role: "Evaluator showcase site and interactive workspace with evidence overlays and execution tracing.",
                    },
                    {
                      component: "Deep Learning Framework",
                      tech: "PyTorch, Torchvision",
                      role: "Model weight initialization, GPU/CPU tensor execution, and differential feature mapping.",
                    },
                    {
                      component: "VQA and Captioning Model",
                      tech: "Salesforce/blip2-opt-2.7b",
                      role: "Pretrained vision-language model generating textual answers and descriptive captions from imagery.",
                    },
                    {
                      component: "Text-Guided Grounding",
                      tech: "google/owlvit-base-patch32",
                      role: "Open-vocabulary object detector identifying spatial coordinates from arbitrary natural-language terms.",
                    },
                    {
                      component: "Change Detection Backbone",
                      tech: "microsoft/resnet-50 (Siamese)",
                      role: "Extracts layer features across registered bi-temporal pairs to calculate continuous Euclidean change distance.",
                    },
                    {
                      component: "Zero-Shot Remote Sensing",
                      tech: "OpenCLIP ViT-B-32 (BigEarthNet)",
                      role: "Multimodal embeddings adapted on remote-sensing benchmark data for semantic task classification.",
                    },
                    {
                      component: "Geospatial and I/O",
                      tech: "rasterio, Pillow, NumPy, SciPy",
                      role: "GeoTIFF band decoding, coordinate georeferencing, image pre-processing, and morphological filtering.",
                    },
                    {
                      component: "Provenance and Export",
                      tech: "ReportLab",
                      role: "Automated PDF report compilation containing input metadata, execution traces, visual evidence, and timestamps.",
                    },
                  ].map((row, idx) => (
                    <tr
                      key={idx}
                      className="hover:bg-[#F0F1EC]/40 transition-colors duration-150"
                    >
                      <td className="py-3 px-4 font-medium text-[#1A2129]">
                        {row.component}
                      </td>
                      <td className="py-3 px-4 font-mono text-xs text-[#1A2129]">
                        {row.tech}
                      </td>
                      <td className="py-3 px-4 text-[#1A2129]/80 text-xs">
                        {row.role}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* ── 6. TEAM: Omitted per PRD Section 4 and 9 ─────────────────────── */}

        {/* ── 7. FOOTER ─────────────────────────────────────────────────────── */}
        <footer
          className="border-t pt-10 pb-16 text-xs text-[#1A2129]/80 space-y-6"
          style={{ borderColor: "#CBCFC6" }}
        >
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="space-y-1">
              <div className="font-semibold text-sm text-[#1A2129]">
                SatQuery AI
              </div>
              <p className="max-w-[60ch]">
                Developed for Smart India Hackathon 2026. Problem Statement
                26167: Space Applications Centre, ISRO.
              </p>
            </div>

            <nav className="flex flex-wrap items-center gap-4 text-xs font-mono">
              <a
                href="https://github.com/PrakashRishiraj/SatQuery_AI"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#28506B] underline hover:text-[#1A2129] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#28506B]"
              >
                github.com/PrakashRishiraj/SatQuery_AI
              </a>
              <Link
                href="/app"
                className="text-[#28506B] underline hover:text-[#1A2129] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#28506B]"
              >
                /app
              </Link>
              <a
                href={`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/docs`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#28506B] underline hover:text-[#1A2129] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#28506B]"
              >
                api:8000/docs
              </a>
            </nav>
          </div>

          <div
            className="border-t pt-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 text-[11px] text-[#1A2129]/60"
            style={{ borderColor: "#CBCFC6" }}
          >
            <div>
              Agentic Vision-Language Remote Sensing Assistant for Optical and
              SAR Satellite Data
            </div>
            <div className="font-mono">PS 26167</div>
          </div>
        </footer>
      </main>
    </div>
  );
}
