"use client";

import { useRouter } from "next/navigation";
import {
  ArrowRight,
  Brain,
  Globe,
  Image,
  MapPin,
  Search,
  Shield,
  Satellite,
  Sparkles,
  CheckCircle2,
  Layers,
  BarChart3,
  AlertTriangle,
  ChevronRight,
} from "lucide-react";

function SatIcon({ size = 18, color = "currentColor" }: { size?: number; color?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2a2.236 2.236 0 0 0-3-3" />
      <path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z" />
      <path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0" />
      <path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5" />
    </svg>
  );
}

const CAPABILITIES = [
  {
    icon: Search,
    title: "Natural-Language Querying",
    desc: "Ask questions about satellite imagery in plain English. No GIS expertise required.",
    color: "var(--accent)",
  },
  {
    icon: Image,
    title: "Image Captioning",
    desc: "Automatic scene description for optical and multispectral remote-sensing imagery.",
    color: "var(--veg)",
  },
  {
    icon: Brain,
    title: "Visual Question Answering",
    desc: "Targeted answers about land cover, infrastructure, vegetation, water bodies, and more.",
    color: "var(--water)",
  },
  {
    icon: Layers,
    title: "Land-Cover Classification",
    desc: "Zero-shot classification into semantic land-cover categories using RS-CLIP embeddings.",
    color: "var(--bare)",
  },
  {
    icon: BarChart3,
    title: "Spectral Statistics",
    desc: "Deterministic band-level statistics computed directly from raster data — no ML model needed.",
    color: "var(--sar)",
  },
  {
    icon: MapPin,
    title: "Deterministic Geolocation",
    desc: "When geospatial metadata is present, extract precise coordinates and place names without any ML model.",
    color: "var(--water)",
  },
  {
    icon: Shield,
    title: "Verification-Aware Responses",
    desc: "When confidence is low or metadata is unavailable, SatQuery AI says so — it does not fabricate answers.",
    color: "var(--change)",
  },
  {
    icon: Globe,
    title: "Agentic Task Routing",
    desc: "An intelligent classifier routes each query to the optimal specialist model or deterministic tool.",
    color: "var(--accent)",
  },
];

const PIPELINE_STEPS = [
  { label: "Your Query", detail: "\"What land cover types are present?\"" },
  { label: "Task Classification", detail: "Intent detection + confidence scoring" },
  { label: "Agentic Routing", detail: "Specialist model or deterministic tool selection" },
  { label: "Execution", detail: "Model inference or geospatial computation" },
  { label: "Evidence Integration", detail: "Confidence assessment + tool evidence" },
  { label: "Natural-Language Result", detail: "Answer with evidence and uncertainty" },
];

const MODALITIES = [
  { name: "Optical", examples: "Sentinel-2, Landsat, high-res RGB", status: "supported" },
  { name: "Multispectral", examples: "Band-composite analysis, NDVI-adjacent", status: "supported" },
  { name: "SAR", examples: "Synthetic Aperture Radar workflows", status: "architectural" },
];

export default function LandingPage() {
  const router = useRouter();

  const handleLaunch = () => {
    router.push("/app");
  };

  const handleSignIn = () => {
    router.push("/login");
  };

  return (
    <div className="min-h-screen" style={{ background: "#080B0F", color: "#E8EDF2" }}>
      {/* ── Navbar ─────────────────────────────────────────────────────── */}
      <nav className="fixed top-0 inset-x-0 z-50 h-16 flex items-center justify-between px-6 sm:px-10" style={{ background: "rgba(8,11,15,0.85)", backdropFilter: "blur(12px)", borderBottom: "1px solid rgba(232,237,242,0.06)" }}>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: "rgba(81,141,178,0.15)", border: "1px solid rgba(81,141,178,0.3)" }}>
            <SatIcon size={16} color="#9BD5E8" />
          </div>
          <span className="text-sm font-semibold tracking-tight text-white">SatQuery AI</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={handleSignIn}
            className="text-xs font-medium px-3 py-1.5 rounded-lg transition-colors duration-150"
            style={{ color: "rgba(232,237,242,0.5)" }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "white")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "rgba(232,237,242,0.5)")}
          >
            Sign in
          </button>
          <button
            type="button"
            onClick={handleLaunch}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#9BD5E8]"
            style={{ background: "#518DB2", color: "white" }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "#3D7396")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "#518DB2")}
          >
            Launch SatQuery AI
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </nav>

      {/* ── Hero ───────────────────────────────────────────────────────── */}
      <section className="relative pt-32 pb-20 px-6 sm:px-10 overflow-hidden">
        {/* Subtle radial glow */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[600px] pointer-events-none" style={{ background: "radial-gradient(ellipse at center, rgba(81,141,178,0.08) 0%, transparent 70%)" }} />

        <div className="relative max-w-3xl mx-auto text-center space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-md text-xs font-mono" style={{ background: "rgba(81,141,178,0.1)", border: "1px solid rgba(81,141,178,0.25)", color: "#9BD5E8" }}>
            <SatIcon size={12} color="#9BD5E8" />
            <span>ISRO / SAC &middot; Smart India Hackathon 2026</span>
          </div>

          <h1 className="text-4xl sm:text-6xl font-bold tracking-tight text-white leading-[1.1]">
            Ask Questions.<br />
            <span style={{ color: "#9BD5E8" }}>Unlock Intelligence From Earth.</span>
          </h1>

          <p className="text-base sm:text-lg max-w-[52ch] mx-auto leading-relaxed" style={{ color: "rgba(232,237,242,0.6)" }}>
            SatQuery AI is an agentic vision-language system that lets you query satellite and remote-sensing imagery using natural language — and get evidence-backed answers.
          </p>

          <div className="pt-4 flex flex-col sm:flex-row items-center justify-center gap-3">
            <button
              type="button"
              onClick={handleLaunch}
              className="inline-flex items-center gap-2 px-6 py-3 rounded-lg text-base font-semibold transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#9BD5E8]"
              style={{ background: "#518DB2", color: "white" }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "#3D7396")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "#518DB2")}
            >
              Launch SatQuery AI
              <ArrowRight className="w-4 h-4" />
            </button>
            <a
              href="#capabilities"
              className="inline-flex items-center gap-2 px-5 py-3 rounded-lg text-sm font-medium transition-colors duration-150"
              style={{ color: "rgba(232,237,242,0.5)", border: "1px solid rgba(232,237,242,0.1)" }}
              onMouseEnter={(e) => { e.currentTarget.style.color = "white"; e.currentTarget.style.borderColor = "rgba(232,237,242,0.2)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.color = "rgba(232,237,242,0.5)"; e.currentTarget.style.borderColor = "rgba(232,237,242,0.1)"; }}
            >
              Explore capabilities
              <ChevronRight className="w-3.5 h-3.5" />
            </a>
          </div>
        </div>
      </section>

      {/* ── What Is SatQuery AI ────────────────────────────────────────── */}
      <section className="py-20 px-6 sm:px-10" style={{ borderTop: "1px solid rgba(232,237,242,0.06)" }}>
        <div className="max-w-3xl mx-auto text-center space-y-4">
          <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">What Is SatQuery AI?</h2>
          <p className="text-sm sm:text-base leading-relaxed" style={{ color: "rgba(232,237,242,0.55)" }}>
            SatQuery AI is a prototype intelligent assistant for Earth observation. It combines vision-language models
            with deterministic remote-sensing tools to answer natural-language questions about satellite imagery.
            Built for ISRO&apos;s Space Applications Centre as Problem Statement 26167, it demonstrates how agentic AI
            can make remote-sensing analysis accessible to anyone who can ask a question.
          </p>
        </div>
      </section>

      {/* ── Capabilities ───────────────────────────────────────────────── */}
      <section id="capabilities" className="py-20 px-6 sm:px-10" style={{ borderTop: "1px solid rgba(232,237,242,0.06)" }}>
        <div className="max-w-5xl mx-auto space-y-12">
          <div className="text-center space-y-3">
            <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">Capabilities</h2>
            <p className="text-sm max-w-[48ch] mx-auto" style={{ color: "rgba(232,237,242,0.45)" }}>
              Each capability is backed by a real implementation in the SatQuery AI pipeline.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {CAPABILITIES.map((cap) => (
              <div
                key={cap.title}
                className="rounded-xl p-5 space-y-3 transition-colors duration-200"
                style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}
                onMouseEnter={(e) => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.12)"; e.currentTarget.style.background = "rgba(255,255,255,0.05)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.06)"; e.currentTarget.style.background = "rgba(255,255,255,0.03)"; }}
              >
                <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ background: `color-mix(in srgb, ${cap.color} 12%, transparent)`, border: `1px solid color-mix(in srgb, ${cap.color} 25%, transparent)` }}>
                  <cap.icon className="w-4 h-4" style={{ color: cap.color }} />
                </div>
                <h3 className="text-sm font-semibold text-white">{cap.title}</h3>
                <p className="text-xs leading-relaxed" style={{ color: "rgba(232,237,242,0.45)" }}>{cap.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How It Works ───────────────────────────────────────────────── */}
      <section className="py-20 px-6 sm:px-10" style={{ borderTop: "1px solid rgba(232,237,242,0.06)" }}>
        <div className="max-w-3xl mx-auto space-y-12">
          <div className="text-center space-y-3">
            <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">How It Works</h2>
            <p className="text-sm max-w-[48ch] mx-auto" style={{ color: "rgba(232,237,242,0.45)" }}>
              From natural-language query to evidence-backed answer in seconds.
            </p>
          </div>

          <div className="space-y-3">
            {PIPELINE_STEPS.map((step, i) => (
              <div
                key={step.label}
                className="flex items-start gap-4 rounded-xl px-5 py-4"
                style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}
              >
                <div
                  className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold mt-0.5"
                  style={{ background: "rgba(81,141,178,0.15)", color: "#9BD5E8", border: "1px solid rgba(81,141,178,0.3)" }}
                >
                  {i + 1}
                </div>
                <div>
                  <p className="text-sm font-semibold text-white">{step.label}</p>
                  <p className="text-xs mt-0.5" style={{ color: "rgba(232,237,242,0.4)" }}>{step.detail}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Multimodal Remote Sensing ──────────────────────────────────── */}
      <section className="py-20 px-6 sm:px-10" style={{ borderTop: "1px solid rgba(232,237,242,0.06)" }}>
        <div className="max-w-3xl mx-auto space-y-12">
          <div className="text-center space-y-3">
            <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">Multimodal Remote Sensing</h2>
            <p className="text-sm max-w-[52ch] mx-auto" style={{ color: "rgba(232,237,242,0.45)" }}>
              Designed for the workflows that matter in Earth observation.
            </p>
          </div>

          <div className="space-y-3">
            {MODALITIES.map((m) => (
              <div
                key={m.name}
                className="flex items-center justify-between rounded-xl px-5 py-4"
                style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}
              >
                <div>
                  <p className="text-sm font-semibold text-white">{m.name}</p>
                  <p className="text-xs mt-0.5" style={{ color: "rgba(232,237,242,0.4)" }}>{m.examples}</p>
                </div>
                <span
                  className="text-[10px] font-mono px-2 py-1 rounded"
                  style={m.status === "supported"
                    ? { background: "rgba(90,112,82,0.15)", color: "#5A7052", border: "1px solid rgba(90,112,82,0.3)" }
                    : { background: "rgba(95,108,116,0.15)", color: "#5F6C74", border: "1px solid rgba(95,108,116,0.3)" }
                  }
                >
                  {m.status === "supported" ? "SUPPORTED" : "ARCHITECTURAL"}
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Trust & Safety ─────────────────────────────────────────────── */}
      <section className="py-20 px-6 sm:px-10" style={{ borderTop: "1px solid rgba(232,237,242,0.06)" }}>
        <div className="max-w-3xl mx-auto space-y-8">
          <div className="text-center space-y-3">
            <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">Trust & Safety</h2>
            <p className="text-sm max-w-[52ch] mx-auto" style={{ color: "rgba(232,237,242,0.45)" }}>
              SatQuery AI does not blindly fabricate geographic or analytical information.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="rounded-xl p-5 space-y-3" style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}>
              <CheckCircle2 className="w-5 h-5" style={{ color: "#5A7052" }} />
              <h3 className="text-sm font-semibold text-white">Deterministic When Possible</h3>
              <p className="text-xs leading-relaxed" style={{ color: "rgba(232,237,242,0.4)" }}>
                Geolocation uses CRS + affine transform metadata directly — no ML model guesses coordinates.
              </p>
            </div>
            <div className="rounded-xl p-5 space-y-3" style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}>
              <AlertTriangle className="w-5 h-5" style={{ color: "#A88A70" }} />
              <h3 className="text-sm font-semibold text-white">Honest About Uncertainty</h3>
              <p className="text-xs leading-relaxed" style={{ color: "rgba(232,237,242,0.4)" }}>
                When metadata is unavailable, location is reported as undetermined — with explicit verification requests.
              </p>
            </div>
            <div className="rounded-xl p-5 space-y-3" style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}>
              <Shield className="w-5 h-5" style={{ color: "#9BD5E8" }} />
              <h3 className="text-sm font-semibold text-white">Evidence-Oriented</h3>
              <p className="text-xs leading-relaxed" style={{ color: "rgba(232,237,242,0.4)" }}>
                Every answer includes tool evidence, confidence scores, and execution traces for auditability.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Why SatQuery AI ────────────────────────────────────────────── */}
      <section className="py-20 px-6 sm:px-10" style={{ borderTop: "1px solid rgba(232,237,242,0.06)" }}>
        <div className="max-w-3xl mx-auto space-y-8">
          <div className="text-center space-y-3">
            <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">Why SatQuery AI?</h2>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {[
              { icon: Search, text: "Natural-language interaction — no GIS expertise required" },
              { icon: Brain, text: "Specialist routing — right model for every query type" },
              { icon: Satellite, text: "Remote-sensing aware — optical, multispectral, SAR-ready" },
              { icon: Sparkles, text: "Evidence-oriented — confidence scores and tool provenance" },
              { icon: MapPin, text: "Geospatial intelligence — deterministic coordinate extraction" },
              { icon: Layers, text: "Extensible architecture — new specialists plug in cleanly" },
            ].map((item) => (
              <div key={item.text} className="flex items-start gap-3 px-4 py-3 rounded-lg" style={{ background: "rgba(255,255,255,0.02)" }}>
                <item.icon className="w-4 h-4 mt-0.5 flex-shrink-0" style={{ color: "#9BD5E8" }} />
                <p className="text-sm" style={{ color: "rgba(232,237,242,0.6)" }}>{item.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Final CTA ──────────────────────────────────────────────────── */}
      <section className="py-24 px-6 sm:px-10" style={{ borderTop: "1px solid rgba(232,237,242,0.06)" }}>
        <div className="max-w-xl mx-auto text-center space-y-6">
          <h2 className="text-3xl sm:text-4xl font-bold text-white tracking-tight">
            Ready to analyze?
          </h2>
          <p className="text-sm" style={{ color: "rgba(232,237,242,0.45)" }}>
            Upload satellite imagery and ask questions in natural language.
          </p>
          <div className="pt-2 flex flex-col items-center gap-3">
            <button
              type="button"
              onClick={handleLaunch}
              className="inline-flex items-center gap-2 px-6 py-3 rounded-lg text-base font-semibold transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#9BD5E8]"
              style={{ background: "#518DB2", color: "white" }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "#3D7396")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "#518DB2")}
            >
              Launch SatQuery AI
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={handleSignIn}
              className="text-xs font-medium transition-colors duration-150"
              style={{ color: "rgba(232,237,242,0.35)" }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "rgba(232,237,242,0.6)")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "rgba(232,237,242,0.35)")}
            >
              or sign in with Google to save your analyses
            </button>
          </div>
        </div>
      </section>

      {/* ── Footer ─────────────────────────────────────────────────────── */}
      <footer className="py-8 px-6 sm:px-10" style={{ borderTop: "1px solid rgba(232,237,242,0.06)" }}>
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <SatIcon size={14} color="#518DB2" />
            <span className="text-xs font-mono" style={{ color: "rgba(232,237,242,0.3)" }}>
              SatQuery AI &middot; PS 26167 &middot; ISRO / SAC
            </span>
          </div>
          <span className="text-[11px] font-mono" style={{ color: "rgba(232,237,242,0.2)" }}>
            Smart India Hackathon 2026
          </span>
        </div>
      </footer>
    </div>
  );
}
