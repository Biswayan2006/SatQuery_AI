"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { ArrowRight, ChevronDown, Check } from "lucide-react";

/* ── Inline SVG Satellite for consistent styling ─────────────────────────── */
function SatelliteIcon({ size = 18, color = "currentColor" }: { size?: number; color?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2a2.236 2.236 0 0 0-3-3" />
      <path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z" />
      <path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0" />
      <path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5" />
    </svg>
  );
}

export default function LandingPage() {
  const router = useRouter();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [scrollProgress, setScrollProgress] = useState(0);
  const [isReducedMotion, setIsReducedMotion] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [isLaunching, setIsLaunching] = useState(false);

  const isLaunchingRef = useRef(false);
  isLaunchingRef.current = isLaunching;

  // Check prefers-reduced-motion
  useEffect(() => {
    setMounted(true);
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setIsReducedMotion(mq.matches);
    if (mq.matches) {
      setScrollProgress(1);
    }
    const handler = (e: MediaQueryListEvent) => {
      setIsReducedMotion(e.matches);
      if (e.matches) {
        setScrollProgress(1);
      }
    };
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  // Pre-fetch the /app dashboard route for zero-latency transition
  useEffect(() => {
    try {
      router.prefetch("/app");
    } catch {
      // ignore
    }
  }, [router]);

  // Track window scroll progress between 0 and 1
  useEffect(() => {
    if (isReducedMotion || isLaunching) return;

    const handleScroll = () => {
      const scrollY = window.scrollY;
      const maxScroll =
        document.documentElement.scrollHeight - window.innerHeight;
      if (maxScroll <= 0) {
        setScrollProgress(0);
        return;
      }
      const p = Math.min(Math.max(scrollY / maxScroll, 0), 1);
      setScrollProgress(p);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
  }, [isReducedMotion, isLaunching]);

  // Launch transition into /app dashboard
  const handleLaunch = useCallback(() => {
    if (isReducedMotion) {
      router.push("/app");
      return;
    }
    setIsLaunching(true);
    setTimeout(() => {
      router.push("/app");
    }, 600);
  }, [isReducedMotion, router]);

  // Smooth scroll helper to advance to launch stage
  const scrollToLaunch = useCallback(() => {
    window.scrollTo({
      top: document.documentElement.scrollHeight,
      behavior: "smooth",
    });
  }, []);

  // Canvas Earth & Space Animation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;
    let rotation = 0;
    let launchZoom = 0;

    // Fixed stars list
    const starsCount = 200;
    const stars: { x: number; y: number; r: number; a: number; speed: number }[] = [];

    const initStars = (w: number, h: number) => {
      stars.length = 0;
      for (let i = 0; i < starsCount; i++) {
        stars.push({
          x: Math.random() * w,
          y: Math.random() * h,
          r: Math.random() * 1.2 + 0.3,
          a: Math.random() * 0.7 + 0.2,
          speed: Math.random() * 0.02 + 0.005,
        });
      }
    };

    // Resize handling
    const resizeCanvas = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = window.innerWidth;
      const h = window.innerHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.scale(dpr, dpr);
      initStars(w, h);
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    // Continental landmass coordinates (approximate polygonal clusters)
    const landmasses = [
      // Eurasia / Africa
      [
        { lat: 60, lng: 30 },
        { lat: 55, lng: 70 },
        { lat: 40, lng: 110 },
        { lat: 20, lng: 80 },
        { lat: 10, lng: 50 },
        { lat: 0, lng: 20 },
        { lat: -30, lng: 25 },
        { lat: -25, lng: 35 },
        { lat: 5, lng: 40 },
        { lat: 35, lng: 30 },
      ],
      // Americas
      [
        { lat: 65, lng: -100 },
        { lat: 50, lng: -80 },
        { lat: 30, lng: -85 },
        { lat: 10, lng: -75 },
        { lat: -10, lng: -55 },
        { lat: -45, lng: -65 },
        { lat: -20, lng: -40 },
        { lat: 5, lng: -50 },
        { lat: 25, lng: -100 },
      ],
      // Australia / Pacific
      [
        { lat: -15, lng: 130 },
        { lat: -25, lng: 150 },
        { lat: -35, lng: 140 },
        { lat: -30, lng: 115 },
      ],
    ];

    // 3D Sphere projection
    function project3D(
      lat: number,
      lng: number,
      rotY: number,
      R: number,
      cx: number,
      cy: number
    ) {
      const phi = ((90 - lat) * Math.PI) / 180;
      const theta = ((lng + rotY) * Math.PI) / 180;
      const x3 = R * Math.sin(phi) * Math.cos(theta);
      const y3 = R * Math.cos(phi);
      const z3 = R * Math.sin(phi) * Math.sin(theta);
      return { x: cx + x3, y: cy - y3, z: z3, visible: z3 > -R * 0.1 };
    }

    let lastTime = performance.now();

    const render = (time: number) => {
      const dt = (time - lastTime) / 1000;
      lastTime = time;

      const w = window.innerWidth;
      const h = window.innerHeight;

      // Accelerate camera if launch transition is triggered
      if (isLaunchingRef.current) {
        launchZoom += dt * 3.2;
      }

      // Clear space background
      ctx.fillStyle = "#080B0F";
      ctx.fillRect(0, 0, w, h);

      // Starfield rendering with subtle twinkle
      stars.forEach((s) => {
        const twinkle = Math.sin(time * s.speed + s.x) * 0.25;
        const alpha = Math.min(Math.max(s.a + twinkle, 0.1), 0.9);
        ctx.fillStyle = `rgba(180, 210, 230, ${alpha})`;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fill();
      });

      // Zoom interpolation based on scroll progress and launch sequence
      const currentP = isReducedMotion ? 0.85 : scrollProgress;
      const baseR = Math.min(w, h) * 0.34;
      const maxR = Math.min(w, h) * 1.7;
      let R = baseR + (maxR - baseR) * Math.pow(currentP, 1.4);

      if (launchZoom > 0) {
        R *= 1 + launchZoom * 2.2;
      }

      // Center shifts slightly for cinematic asymmetry
      const cx = w / 2 - (w * 0.12) * currentP;
      const cy = h / 2 + (h * 0.05) * currentP;

      // Earth rotation
      rotation += dt * (isLaunchingRef.current ? 35 : 8);
      const rotY = rotation;

      // 1. Atmosphere halo (soft blue glow)
      const atmoGrad = ctx.createRadialGradient(cx, cy, R * 0.95, cx, cy, R * 1.25);
      atmoGrad.addColorStop(0, "rgba(61, 115, 150, 0.28)");
      atmoGrad.addColorStop(0.5, "rgba(40, 80, 107, 0.12)");
      atmoGrad.addColorStop(1, "rgba(8, 11, 15, 0)");

      ctx.beginPath();
      ctx.arc(cx, cy, R * 1.25, 0, Math.PI * 2);
      ctx.fillStyle = atmoGrad;
      ctx.fill();

      // 2. Earth base sphere with spherical shading
      const globeGrad = ctx.createRadialGradient(
        cx - R * 0.35,
        cy - R * 0.35,
        R * 0.05,
        cx,
        cy,
        R
      );
      globeGrad.addColorStop(0, "#22394A");
      globeGrad.addColorStop(0.65, "#121E27");
      globeGrad.addColorStop(1, "#090F14");

      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, Math.PI * 2);
      ctx.fillStyle = globeGrad;
      ctx.fill();

      // Clip subsequent continent & grid rendering inside the Earth sphere
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, Math.PI * 2);
      ctx.clip();

      // 3. Latitude & Longitude grid lines
      ctx.strokeStyle = "rgba(110, 165, 195, 0.14)";
      ctx.lineWidth = 1;

      // Latitudes
      for (let lat = -60; lat <= 60; lat += 30) {
        ctx.beginPath();
        let started = false;
        for (let lng = 0; lng <= 360; lng += 10) {
          const pt = project3D(lat, lng, rotY, R, cx, cy);
          if (pt.visible) {
            if (!started) {
              ctx.moveTo(pt.x, pt.y);
              started = true;
            } else {
              ctx.lineTo(pt.x, pt.y);
            }
          } else {
            started = false;
          }
        }
        ctx.stroke();
      }

      // Longitudes
      for (let lng = 0; lng < 360; lng += 30) {
        ctx.beginPath();
        let started = false;
        for (let lat = -80; lat <= 80; lat += 10) {
          const pt = project3D(lat, lng, rotY, R, cx, cy);
          if (pt.visible) {
            if (!started) {
              ctx.moveTo(pt.x, pt.y);
              started = true;
            } else {
              ctx.lineTo(pt.x, pt.y);
            }
          } else {
            started = false;
          }
        }
        ctx.stroke();
      }

      // 4. Continents / Landmasses
      ctx.fillStyle = "rgba(65, 95, 80, 0.45)";
      ctx.strokeStyle = "rgba(120, 170, 140, 0.35)";
      ctx.lineWidth = 1.2;

      landmasses.forEach((poly) => {
        ctx.beginPath();
        let anyVisible = false;
        poly.forEach((coord, idx) => {
          const pt = project3D(coord.lat, coord.lng, rotY, R, cx, cy);
          if (pt.visible) {
            anyVisible = true;
            if (idx === 0) ctx.moveTo(pt.x, pt.y);
            else ctx.lineTo(pt.x, pt.y);
          }
        });
        if (anyVisible) {
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }
      });

      // 5. Day/Night Shadow Terminator overlay
      const shadowGrad = ctx.createLinearGradient(
        cx - R,
        cy - R,
        cx + R * 0.8,
        cy + R * 0.8
      );
      shadowGrad.addColorStop(0, "rgba(255, 255, 255, 0.06)");
      shadowGrad.addColorStop(0.4, "rgba(0, 0, 0, 0)");
      shadowGrad.addColorStop(1, "rgba(4, 6, 8, 0.75)");

      ctx.fillStyle = shadowGrad;
      ctx.fillRect(cx - R, cy - R, R * 2, R * 2);

      ctx.restore(); // end clip

      // Earth limb edge glow
      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(120, 185, 220, 0.35)";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // 6. Satellite Orbit & Moving Satellite
      const orbitA = R * 1.38;
      const orbitB = R * 0.52;
      const orbitAngle = -0.35;

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(orbitAngle);

      // Faint orbital ellipse track
      ctx.beginPath();
      ctx.ellipse(0, 0, orbitA, orbitB, 0, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(81, 141, 178, ${Math.max(0.28 - currentP * 0.2, 0.08)})`;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 6]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Satellite position along orbit
      const satSpeed = 0.45;
      const satAngle = (time / 1000) * satSpeed;
      const satX = orbitA * Math.cos(satAngle);
      const satY = orbitB * Math.sin(satAngle);

      // Nadir sensor beam projected from satellite to Earth surface
      if (currentP < 0.8 && !isLaunchingRef.current) {
        ctx.beginPath();
        ctx.moveTo(satX, satY);
        const footX = satX * 0.72;
        const footY = satY * 0.72;
        ctx.lineTo(footX - 12, footY);
        ctx.lineTo(footX + 12, footY);
        ctx.closePath();
        ctx.fillStyle = "rgba(155, 213, 232, 0.08)";
        ctx.fill();

        ctx.beginPath();
        ctx.ellipse(footX, footY, 14, 5, 0, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(155, 213, 232, 0.32)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Draw Satellite icon & solar panels
      ctx.translate(satX, satY);

      // Satellite body
      ctx.fillStyle = "#E8EDF2";
      ctx.fillRect(-3, -3, 6, 6);

      // Solar panel wings
      ctx.fillStyle = "#518DB2";
      ctx.fillRect(-10, -2, 5, 4);
      ctx.fillRect(5, -2, 5, 4);

      // Antenna beacon
      ctx.strokeStyle = "#9BD5E8";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, -3);
      ctx.lineTo(0, -6);
      ctx.stroke();

      // Satellite pulse
      const pulseR = 8 + Math.sin(time * 0.005) * 3;
      ctx.beginPath();
      ctx.arc(0, 0, pulseR, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(155, 213, 232, 0.28)";
      ctx.stroke();

      ctx.restore();

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", resizeCanvas);
    };
  }, [scrollProgress, isReducedMotion]);

  // Optical imagery transition opacity: emerges as camera approaches Earth
  const imageryOpacity = isReducedMotion
    ? 0.75
    : Math.min(Math.max((scrollProgress - 0.4) / 0.35, 0), 0.85);

  // Text stage visibility calculations
  const isOrbitState = scrollProgress < 0.35 && !isLaunching;
  const isObservationState = scrollProgress >= 0.4 && scrollProgress < 0.78 && !isLaunching;
  const isLaunchState = (scrollProgress >= 0.78 || isReducedMotion) && !isLaunching;

  return (
    <div
      ref={containerRef}
      className="relative bg-[#080B0F] text-[#E8EDF2] select-none font-sans"
      style={{
        height: isReducedMotion ? "100vh" : "280vh",
      }}
    >
      {/* ── Fixed Canvas Viewport ─────────────────────────────────────────── */}
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
        <canvas ref={canvasRef} className="w-full h-full block" />

        {/* Remote Sensing Satellite Imagery Layer (Blends in upon approach) */}
        <div
          className="absolute inset-0 transition-opacity duration-500 flex items-center justify-center pointer-events-none"
          style={{ opacity: imageryOpacity }}
        >
          {/* High-res remote sensing optical crop with subtle vignette */}
          <div className="relative w-full max-w-4xl h-[70vh] rounded-2xl overflow-hidden border border-[#518DB2]/30 shadow-2xl shadow-black/80">
            <Image
              src="/assets/optical_main_hd.png"
              alt="Optical Earth observation imagery"
              fill
              className="object-cover brightness-95 contrast-105"
              priority
            />
            {/* Dark vignette blending into space */}
            <div className="absolute inset-0 bg-gradient-to-t from-[#080B0F] via-transparent to-[#080B0F]/80" />
            <div className="absolute inset-0 bg-gradient-to-r from-[#080B0F] via-transparent to-[#080B0F]/80" />

            {/* Sensor telemetry brackets */}
            <div className="absolute top-4 left-5 font-mono text-[11px] text-[#9BD5E8]/80 flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-[#518DB2] animate-pulse" />
              <span>SENSOR: SENTINEL-2 MSI / 10M GSD</span>
            </div>
            <div className="absolute bottom-4 right-5 font-mono text-[11px] text-[#9BD5E8]/70">
              LAT 19°04&apos;N &middot; LON 72°52&apos;E
            </div>
          </div>
        </div>
      </div>

      {/* ── Launch HUD Transition Overlay ─────────────────────────────────── */}
      {isLaunching && (
        <div className="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-[#080B0F]/80 backdrop-blur-md transition-opacity duration-300">
          <div className="max-w-md w-full px-6 text-center space-y-4 font-mono">
            <div className="w-12 h-12 mx-auto rounded-lg bg-[#518DB2]/20 border border-[#518DB2]/50 flex items-center justify-center text-[#9BD5E8] animate-pulse">
              <SatelliteIcon size={24} color="#9BD5E8" />
            </div>
            <div className="space-y-1">
              <p className="text-xs font-semibold tracking-wider text-[#9BD5E8]">
                ESTABLISHING SENSOR LINK
              </p>
              <p className="text-sm text-white">
                Connecting to SatQuery AI Workspace...
              </p>
            </div>
            <div className="w-full h-1.5 bg-[#1A2129] rounded-full overflow-hidden border border-[#518DB2]/30">
              <div className="h-full bg-[#518DB2] animate-pulse w-full transition-all duration-500" />
            </div>
            <div className="flex justify-between text-[10px] text-[#9BD5E8]/60">
              <span>CONTROLLER: 8000</span>
              <span>ORBIT: LOCKED</span>
              <span>SCHEMA: RS-VLM</span>
            </div>
          </div>
        </div>
      )}

      {/* ── Minimal Header (Always Accessible) ────────────────────────────── */}
      <header className="fixed top-0 inset-x-0 z-50 h-16 flex items-center justify-between px-6 sm:px-10 border-b border-[#E8EDF2]/10 bg-[#080B0F]/40 backdrop-blur-md">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-md bg-[#518DB2]/20 border border-[#518DB2]/40 flex items-center justify-center text-[#9BD5E8]">
            <SatelliteIcon size={18} color="#9BD5E8" />
          </div>
          <span className="text-base font-semibold tracking-tight text-white">
            SatQuery AI
          </span>
        </div>

        <nav>
          <button
            type="button"
            onClick={handleLaunch}
            onMouseEnter={() => router.prefetch("/app")}
            className="text-xs sm:text-sm font-medium px-4 py-2 rounded-md bg-[#518DB2] text-white hover:bg-[#3D7396] transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#9BD5E8] focus-visible:ring-offset-2 focus-visible:ring-offset-[#080B0F]"
          >
            Launch App
          </button>
        </nav>
      </header>

      {/* ── Foreground Content Layers (Sticky Fullscreen) ─────────────────── */}
      <div className="sticky top-0 h-screen w-full flex flex-col justify-between p-6 sm:p-12 pointer-events-none z-10 pt-24">
        {/* Top Spacer */}
        <div />

        {/* Center Dynamic Brand Experience */}
        <div className="max-w-xl mx-auto text-center space-y-6 pointer-events-auto">
          {/* State 1: Orbit / Arrival */}
          {isOrbitState && (
            <div
              className={`space-y-4 transition-all duration-500 ease-out ${
                mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
              }`}
            >
              <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-md border border-[#518DB2]/30 bg-[#518DB2]/10 text-xs font-mono text-[#9BD5E8]">
                <SatelliteIcon size={13} color="#9BD5E8" />
                <span>ORBITAL RECONNAISSANCE</span>
              </div>

              <h1 className="text-4xl sm:text-6xl font-bold tracking-tight text-white leading-tight">
                Ask Earth.
              </h1>

              <p className="text-base sm:text-lg text-[#E8EDF2]/80 max-w-[42ch] mx-auto leading-relaxed">
                Understand satellite imagery through natural language.
              </p>

              <div className="pt-2">
                <button
                  type="button"
                  onClick={handleLaunch}
                  onMouseEnter={() => router.prefetch("/app")}
                  className="inline-flex items-center space-x-2 px-5 py-3 rounded-md bg-[#518DB2] text-white text-sm font-semibold hover:bg-[#3D7396] transition-colors duration-150 shadow-lg shadow-[#518DB2]/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#9BD5E8]"
                >
                  <span>Launch SatQuery AI</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}

          {/* State 2: Atmospheric Micro-Interaction during Approach */}
          {isObservationState && (
            <div className="space-y-3 transition-all duration-500 ease-out animate-fade-in">
              <div className="inline-block px-4 py-2 rounded-md bg-[#080B0F]/80 border border-[#518DB2]/40 backdrop-blur-md font-mono text-sm sm:text-base text-white shadow-xl">
                &ldquo;What&apos;s changed here?&rdquo;
              </div>
              <div className="flex items-center justify-center space-x-2 text-xs font-mono text-[#9BD5E8]">
                <span className="w-1.5 h-1.5 rounded-full bg-[#9BD5E8] animate-ping" />
                <span>Analyzing multi-temporal sensor pass...</span>
              </div>
            </div>
          )}

          {/* State 3: Launch Entry Stage */}
          {isLaunchState && (
            <div className="space-y-6 transition-all duration-500 ease-out animate-fade-in">
              <div className="space-y-2">
                <h2 className="text-3xl sm:text-5xl font-bold text-white tracking-tight">
                  SatQuery AI
                </h2>
                <p className="text-base sm:text-lg text-[#E8EDF2]/90 max-w-[40ch] mx-auto">
                  Understand satellite imagery through natural language.
                </p>
                <p className="text-xs text-[#9BD5E8]/80 font-mono">
                  Multimodal intelligence for Earth observation.
                </p>
              </div>

              <div className="pt-2">
                <button
                  type="button"
                  onClick={handleLaunch}
                  onMouseEnter={() => router.prefetch("/app")}
                  className="inline-flex items-center space-x-2 px-6 py-3.5 rounded-md bg-[#518DB2] text-white text-base font-semibold hover:bg-[#3D7396] transition-colors duration-150 shadow-xl shadow-[#518DB2]/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#9BD5E8]"
                >
                  <span>Launch SatQuery AI</span>
                  <ArrowRight className="w-5 h-5" />
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Bottom Scroll Indicator (Visible on initial orbit screen) */}
        <div className="flex items-center justify-between text-xs text-[#E8EDF2]/60 font-mono pointer-events-auto">
          <div>
            <span>EARTH OBSERVATION</span>
          </div>

          {!isLaunchState && (
            <button
              type="button"
              onClick={scrollToLaunch}
              className="flex items-center space-x-1.5 hover:text-white transition-colors duration-150 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#9BD5E8]"
            >
              <span>Scroll to approach</span>
              <ChevronDown className="w-3.5 h-3.5 animate-bounce" />
            </button>
          )}

          {isLaunchState && (
            <button
              type="button"
              onClick={handleLaunch}
              className="hover:text-white transition-colors duration-150"
            >
              Ready for analysis &rarr;
            </button>
          )}

          <div>
            <span>PS 26167</span>
          </div>
        </div>
      </div>
    </div>
  );
}
