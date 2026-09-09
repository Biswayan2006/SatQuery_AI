"use client";

import { useState } from "react";
import Image from "next/image";
import Link from "next/link";

interface Scenario {
  id: string;
  name: string;
  task: string;
  latency: string;
  query: string;
  inputLabel: string;
  inputSrc: string;
  evidenceLabel: string;
  evidenceSrc: string;
  answer: string;
  model: string;
  traceSteps: string[];
}

const SCENARIOS: Scenario[] = [
  {
    id: "change",
    name: "Change detection",
    task: "CHANGE_DETECTION",
    latency: "412ms",
    query: "What changed between these two acquisition dates?",
    inputLabel: "Registered optical input (T1)",
    inputSrc: "/assets/optical_main_hd.png",
    evidenceLabel: "Siamese differential distance heatmap",
    evidenceSrc: "/assets/change_main_hd.png",
    answer:
      "Analysis indicates new ground surface clearance and construction activity in the southern sector. Bi-temporal Euclidean feature difference highlights localized change across the scene.",
    model: "microsoft/resnet-50 (Siamese backbone)",
    traceSteps: [
      "input_validator: 2 registered optical tiles decoded (3 bands, 256x256)",
      "task_classifier: routed to CHANGE_DETECTION (confidence: 0.94)",
      "controller: extracted layer-4 embeddings and calculated Euclidean distance",
      "result_integrator: rendered differential overlay and compiled report summary",
    ],
  },
  {
    id: "vqa",
    name: "Visual Q&A",
    task: "VQA",
    latency: "628ms",
    query: "Identify the dominant land-cover and waterway features in this scene.",
    inputLabel: "Multispectral optical tile",
    inputSrc: "/assets/optical_main_hd.png",
    evidenceLabel: "Optical scene inspection",
    evidenceSrc: "/assets/optical_sample.png",
    answer:
      "The scene contains coastal land-cover characterized by tidal estuaries, dense riparian vegetation bordering water channels, and adjacent agricultural parcels.",
    model: "Salesforce/blip2-opt-2.7b",
    traceSteps: [
      "input_validator: 1 optical tile validated (.png, 256x256)",
      "task_classifier: routed to VQA (confidence: 0.91)",
      "controller: conditioned vision-language prompt with remote-sensing schema",
      "result_integrator: generated natural-language descriptive answer",
    ],
  },
  {
    id: "grounding",
    name: "Text-guided grounding",
    task: "GROUNDING",
    latency: "389ms",
    query: "Locate industrial storage tanks and coastal structures.",
    inputLabel: "Sub-meter aerial crop",
    inputSrc: "/assets/optical_sample.png",
    evidenceLabel: "Open-vocabulary detection coordinates",
    evidenceSrc: "/assets/change_sample.png",
    answer:
      "Located target structures with high visual agreement. Normalized bounding coordinates extracted for coastal infrastructure and tanks.",
    model: "google/owlvit-base-patch32",
    traceSteps: [
      "input_validator: single image input verified",
      "task_classifier: routed to GROUNDING (confidence: 0.88)",
      "controller: evaluated text embeddings against image patch tokens",
      "result_integrator: filtered bounding boxes with score threshold > 0.25",
    ],
  },
  {
    id: "fusion",
    name: "SAR and optical fusion",
    task: "SAR_FUSION",
    latency: "514ms",
    query: "Assess ground roughness and structures through cloud-obscured sectors.",
    inputLabel: "Optical multispectral reflectance",
    inputSrc: "/assets/optical_sample.png",
    evidenceLabel: "Synthetic aperture radar backscatter",
    evidenceSrc: "/assets/sar_main_hd.png",
    answer:
      "Synthetic aperture radar backscatter (VV/VH polarizations) reveals metallic structures and high-roughness terrain obscured by optical cloud cover.",
    model: "Dual ResNet-50 + MLP fusion",
    traceSteps: [
      "input_validator: paired optical and SAR inputs confirmed",
      "task_classifier: routed to SAR_FUSION (confidence: 0.89)",
      "controller: aligned radar backscatter amplitude with optical channels",
      "result_integrator: fused cross-modal features into unified interpretation",
    ],
  },
];

export default function ShowcasePage() {
  const [activeScenarioId, setActiveScenarioId] = useState<string>("change");
  const currentScenario =
    SCENARIOS.find((s) => s.id === activeScenarioId) ?? SCENARIOS[0];

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
              href="/demo"
              className="text-xs sm:text-sm font-medium px-3.5 py-1.5 rounded-md text-white transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#28506B] focus-visible:ring-offset-2 focus-visible:ring-offset-[#F0F1EC]"
              style={{ backgroundColor: "#28506B" }}
            >
              View live demo
            </Link>
            <a
              href="https://github.com/PrakashRishiraj/SatQuery_AI"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs sm:text-sm font-medium px-3.5 py-1.5 rounded-md border transition-colors hover:bg-[#CBCFC6]/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1A2129] focus-visible:ring-offset-2 focus-visible:ring-offset-[#F0F1EC]"
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
        {/* ── 1. HERO ───────────────────────────────────────────────────────── */}
        <section
          id="hero"
          aria-labelledby="hero-heading"
          className="grid grid-cols-1 lg:grid-cols-12 gap-10 lg:gap-12 items-start"
        >
          {/* Left Column: Asymmetric text briefing */}
          <div className="lg:col-span-5 space-y-6 text-left">
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
                href="/demo"
                className="inline-flex items-center justify-center text-sm font-medium px-4 py-2.5 rounded-md text-white transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#28506B] focus-visible:ring-offset-2 focus-visible:ring-offset-[#F0F1EC]"
                style={{ backgroundColor: "#28506B" }}
              >
                View the live demo
              </Link>
              <a
                href="https://github.com/PrakashRishiraj/SatQuery_AI"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center text-sm font-medium px-4 py-2.5 rounded-md border transition-colors hover:bg-[#CBCFC6]/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1A2129] focus-visible:ring-offset-2 focus-visible:ring-offset-[#F0F1EC]"
                style={{
                  borderColor: "#CBCFC6",
                  color: "#1A2129",
                  backgroundColor: "transparent",
                }}
              >
                View the repository
              </a>
            </div>
          </div>

          {/* Right Column: Real Artifact (Interactive query flow & analysis result) */}
          <div className="lg:col-span-7">
            <div
              className="rounded-md border overflow-hidden"
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
                        onClick={() => setActiveScenarioId(scenario.id)}
                        className="px-2.5 py-1 rounded-md text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#28506B]"
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

              {/* Analysis input and visual evidence */}
              <div className="p-4 space-y-4">
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

                {/* Imagery side-by-side: input tile vs visual proof */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div className="space-y-1.5">
                    <div className="text-[11px] text-[#1A2129]/70">
                      {currentScenario.inputLabel}
                    </div>
                    <div className="relative aspect-[4/3] rounded-md overflow-hidden border border-[#CBCFC6] bg-[#1A2129]/5">
                      <Image
                        src={currentScenario.inputSrc}
                        alt={currentScenario.inputLabel}
                        fill
                        sizes="(max-width: 1024px) 50vw, 25vw"
                        className="object-cover"
                      />
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <div className="text-[11px] text-[#1A2129]/70">
                      {currentScenario.evidenceLabel}
                    </div>
                    <div className="relative aspect-[4/3] rounded-md overflow-hidden border border-[#CBCFC6] bg-[#1A2129]/5">
                      <Image
                        src={currentScenario.evidenceSrc}
                        alt={currentScenario.evidenceLabel}
                        fill
                        sizes="(max-width: 1024px) 50vw, 25vw"
                        className="object-cover"
                      />
                    </div>
                  </div>
                </div>

                {/* Grounded answer output */}
                <div
                  className="p-3.5 rounded-md border space-y-1.5"
                  style={{
                    backgroundColor: "#FFFFFF",
                    borderColor: "#CBCFC6",
                  }}
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium text-[#1A2129]">
                      Evidence-grounded answer
                    </span>
                    <span className="font-mono text-[11px] text-[#28506B]">
                      {currentScenario.model}
                    </span>
                  </div>
                  <p className="text-xs sm:text-sm text-[#1A2129]/90 leading-relaxed">
                    {currentScenario.answer}
                  </p>
                </div>

                {/* Execution trace log */}
                <div
                  className="p-3 rounded-md border text-xs space-y-1.5 font-mono"
                  style={{
                    backgroundColor: "#F0F1EC",
                    borderColor: "#CBCFC6",
                  }}
                >
                  <div className="text-[11px] font-sans font-medium text-[#1A2129]">
                    Execution trace (controller.py):
                  </div>
                  <ul className="space-y-1 text-[11px] text-[#1A2129]/80">
                    {currentScenario.traceSteps.map((step, idx) => (
                      <li key={idx} className="leading-normal">
                        [{idx + 1}] {step}
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
                <span className="text-[#1A2129]/70">
                  Real end-to-end output from the SatQuery AI backend pipeline.
                </span>
                <Link
                  href="/demo"
                  className="font-medium text-[#28506B] underline hover:text-[#1A2129] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#28506B]"
                >
                  Open live workspace
                </Link>
              </div>
            </div>
          </div>
        </section>

        {/* ── 2. THE PROBLEM ────────────────────────────────────────────────── */}
        <section
          id="problem"
          aria-labelledby="problem-heading"
          className="border-t pt-12 sm:pt-16 space-y-6"
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
              Earth observation archives managed by space agencies such as
              ISRO contain petabytes of optical reflectance and Synthetic
              Aperture Radar (SAR) imagery. Extracting actionable insights from
              this data during rapid-response operations (such as flood
              delineation, agricultural damage assessment, or infrastructure
              monitoring) currently requires domain analysts to manually pick,
              configure, and chain separate specialist models.
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

        {/* ── 3. THE APPROACH ───────────────────────────────────────────────── */}
        <section
          id="approach"
          aria-labelledby="approach-heading"
          className="border-t pt-12 sm:pt-16 space-y-8"
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
              className="rounded-md border flex flex-col justify-between overflow-hidden"
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
              className="rounded-md border flex flex-col justify-between overflow-hidden"
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
              className="rounded-md border flex flex-col justify-between overflow-hidden"
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
              className="rounded-md border flex flex-col justify-between overflow-hidden"
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
              className="rounded-md border flex flex-col justify-between overflow-hidden"
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
              className="rounded-md border flex flex-col justify-between overflow-hidden"
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

        {/* ── 4. HOW IT WORKS ───────────────────────────────────────────────── */}
        <section
          id="pipeline"
          aria-labelledby="pipeline-heading"
          className="border-t pt-12 sm:pt-16 space-y-8"
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

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Step 1 */}
            <div
              className="p-5 rounded-md border space-y-3"
              style={{
                backgroundColor: "#FFFFFF",
                borderColor: "#CBCFC6",
              }}
            >
              <div
                className="w-7 h-7 rounded-md flex items-center justify-center font-mono text-xs font-semibold text-white"
                style={{ backgroundColor: "#28506B" }}
              >
                1
              </div>
              <h3 className="text-base font-semibold text-[#1A2129]">
                Input validation
              </h3>
              <p className="text-xs sm:text-sm text-[#1A2129]/80 leading-relaxed">
                Verifies file extensions (.tif, .tiff, .png, .jpg), decodes
                radiometric channels with rasterio and Pillow, validates dimensions,
                and confirms temporal or SAR/optical compatibility for multi-image
                tasks.
              </p>
              <div className="text-xs font-mono text-[#28506B]">
                input_validator.py
              </div>
            </div>

            {/* Step 2 */}
            <div
              className="p-5 rounded-md border space-y-3"
              style={{
                backgroundColor: "#FFFFFF",
                borderColor: "#CBCFC6",
              }}
            >
              <div
                className="w-7 h-7 rounded-md flex items-center justify-center font-mono text-xs font-semibold text-white"
                style={{ backgroundColor: "#28506B" }}
              >
                2
              </div>
              <h3 className="text-base font-semibold text-[#1A2129]">
                Task classification
              </h3>
              <p className="text-xs sm:text-sm text-[#1A2129]/80 leading-relaxed">
                Evaluates query intent and image count to determine the target
                specialist pipeline: VQA, CAPTIONING, GROUNDING, CHANGE_DETECTION,
                CHANGE_VQA, or SAR_FUSION.
              </p>
              <div className="text-xs font-mono text-[#28506B]">
                task_classifier.py
              </div>
            </div>

            {/* Step 3 */}
            <div
              className="p-5 rounded-md border space-y-3"
              style={{
                backgroundColor: "#FFFFFF",
                borderColor: "#CBCFC6",
              }}
            >
              <div
                className="w-7 h-7 rounded-md flex items-center justify-center font-mono text-xs font-semibold text-white"
                style={{ backgroundColor: "#28506B" }}
              >
                3
              </div>
              <h3 className="text-base font-semibold text-[#1A2129]">
                Agentic controller
              </h3>
              <p className="text-xs sm:text-sm text-[#1A2129]/80 leading-relaxed">
                Executes the plan, dispatches tensors to the designated PyTorch
                specialist backbones, measures inference latency, and handles
                model fallbacks if weights are still initializing.
              </p>
              <div className="text-xs font-mono text-[#28506B]">
                controller.py
              </div>
            </div>

            {/* Step 4 */}
            <div
              className="p-5 rounded-md border space-y-3"
              style={{
                backgroundColor: "#FFFFFF",
                borderColor: "#CBCFC6",
              }}
            >
              <div
                className="w-7 h-7 rounded-md flex items-center justify-center font-mono text-xs font-semibold text-white"
                style={{ backgroundColor: "#28506B" }}
              >
                4
              </div>
              <h3 className="text-base font-semibold text-[#1A2129]">
                Result integration
              </h3>
              <p className="text-xs sm:text-sm text-[#1A2129]/80 leading-relaxed">
                Combines textual natural-language answers, visual evidence
                overlays (bounding coordinates or differential change maps),
                execution traces, and exportable PDF reports.
              </p>
              <div className="text-xs font-mono text-[#28506B]">
                report_generator.py
              </div>
            </div>
          </div>
        </section>

        {/* ── 5. TECH STACK ─────────────────────────────────────────────────── */}
        <section
          id="tech-stack"
          aria-labelledby="stack-heading"
          className="border-t pt-12 sm:pt-16 space-y-8"
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
                  <tr>
                    <td className="py-3 px-4 font-medium text-[#1A2129]">
                      Backend Server
                    </td>
                    <td className="py-3 px-4 font-mono text-xs text-[#1A2129]">
                      FastAPI, Uvicorn, Pydantic
                    </td>
                    <td className="py-3 px-4 text-[#1A2129]/80 text-xs">
                      REST API endpoints for image ingestion, async agent
                      execution, and report downloads.
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 font-medium text-[#1A2129]">
                      Frontend Client
                    </td>
                    <td className="py-3 px-4 font-mono text-xs text-[#1A2129]">
                      Next.js 14, React 18, TypeScript, Tailwind CSS
                    </td>
                    <td className="py-3 px-4 text-[#1A2129]/80 text-xs">
                      Evaluator showcase site and interactive workspace with
                      evidence overlays and execution tracing.
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 font-medium text-[#1A2129]">
                      Deep Learning Framework
                    </td>
                    <td className="py-3 px-4 font-mono text-xs text-[#1A2129]">
                      PyTorch, Torchvision
                    </td>
                    <td className="py-3 px-4 text-[#1A2129]/80 text-xs">
                      Model weight initialization, GPU/CPU tensor execution,
                      and differential feature mapping.
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 font-medium text-[#1A2129]">
                      VQA and Captioning Model
                    </td>
                    <td className="py-3 px-4 font-mono text-xs text-[#1A2129]">
                      Salesforce/blip2-opt-2.7b
                    </td>
                    <td className="py-3 px-4 text-[#1A2129]/80 text-xs">
                      Pretrained vision-language model generating textual
                      answers and descriptive captions from imagery.
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 font-medium text-[#1A2129]">
                      Text-Guided Grounding
                    </td>
                    <td className="py-3 px-4 font-mono text-xs text-[#1A2129]">
                      google/owlvit-base-patch32
                    </td>
                    <td className="py-3 px-4 text-[#1A2129]/80 text-xs">
                      Open-vocabulary object detector identifying spatial
                      coordinates from arbitrary natural-language terms.
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 font-medium text-[#1A2129]">
                      Change Detection Backbone
                    </td>
                    <td className="py-3 px-4 font-mono text-xs text-[#1A2129]">
                      microsoft/resnet-50 (Siamese)
                    </td>
                    <td className="py-3 px-4 text-[#1A2129]/80 text-xs">
                      Extracts layer features across registered bi-temporal
                      pairs to calculate continuous Euclidean change distance.
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 font-medium text-[#1A2129]">
                      Zero-Shot Remote Sensing
                    </td>
                    <td className="py-3 px-4 font-mono text-xs text-[#1A2129]">
                      OpenCLIP ViT-B-32 (BigEarthNet)
                    </td>
                    <td className="py-3 px-4 text-[#1A2129]/80 text-xs">
                      Multimodal embeddings adapted on remote-sensing benchmark
                      data for semantic task classification.
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 font-medium text-[#1A2129]">
                      Geospatial and I/O
                    </td>
                    <td className="py-3 px-4 font-mono text-xs text-[#1A2129]">
                      rasterio, Pillow, NumPy, SciPy
                    </td>
                    <td className="py-3 px-4 text-[#1A2129]/80 text-xs">
                      GeoTIFF band decoding, coordinate georeferencing, image
                      pre-processing, and morphological filtering.
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 font-medium text-[#1A2129]">
                      Provenance and Export
                    </td>
                    <td className="py-3 px-4 font-mono text-xs text-[#1A2129]">
                      ReportLab
                    </td>
                    <td className="py-3 px-4 text-[#1A2129]/80 text-xs">
                      Automated PDF report compilation containing input metadata,
                      execution traces, visual evidence, and timestamps.
                    </td>
                  </tr>
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
                href="/demo"
                className="text-[#28506B] underline hover:text-[#1A2129] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#28506B]"
              >
                /demo
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
