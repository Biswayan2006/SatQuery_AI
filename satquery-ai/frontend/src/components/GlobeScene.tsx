"use client";

import { useEffect, useRef, useState } from "react";

interface Props {
  className?: string;
}

type Palette = {
  star: (a: number) => string;
  atmo: string;
  globe: [string, string, string];
  grid1: string;
  grid2: string;
  border: string;
  shine: string;
  orbit: string;
  satGlow: (a: number) => string;
  satCore: string;
};

const DARK: Palette = {
  star: (a) => `rgba(148,212,255,${a})`,
  atmo: "rgba(26,106,255,0.10)",
  globe: ["#0d2a50", "#091a35", "#04091a"],
  grid1: "rgba(58,171,255,0.18)",
  grid2: "rgba(58,171,255,0.13)",
  border: "rgba(58,171,255,0.25)",
  shine: "rgba(255,255,255,0.07)",
  orbit: "rgba(58,171,255,0.12)",
  satGlow: (a) => `rgba(96,197,255,${a})`,
  satCore: "#93d5ff",
};

const LIGHT: Palette = {
  star: (a) => `rgba(90,120,155,${a * 0.7})`,
  atmo: "rgba(82,80,221,0.08)",
  globe: ["#e2ecf8", "#bcd2ea", "#93b2d4"],
  grid1: "rgba(31,99,192,0.24)",
  grid2: "rgba(31,99,192,0.15)",
  border: "rgba(31,99,192,0.32)",
  shine: "rgba(255,255,255,0.55)",
  orbit: "rgba(31,99,192,0.15)",
  satGlow: (a) => `rgba(82,80,221,${a})`,
  satCore: "#5250DD",
};

export default function GlobeScene({ className = "" }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDark, setIsDark] = useState(true);

  // Track theme so the globe repaints its palette on light/dark flips
  useEffect(() => {
    const read = () =>
      setIsDark(document.documentElement.getAttribute("data-theme") !== "light");
    read();
    const obs = new MutationObserver(read);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const P = isDark ? DARK : LIGHT;

    let animId: number;
    let t = 0;

    // ── Geometry helpers ──────────────────────────────────────────────────────
    const W = canvas.width  = canvas.offsetWidth  || 500;
    const H = canvas.height = canvas.offsetHeight || 500;
    const cx = W / 2;
    const cy = H / 2;
    const R  = Math.min(W, H) * 0.32;

    // Lat/lng grid lines
    function latLngToXY(lat: number, lng: number, rotY: number) {
      const phi   = ((90 - lat)  * Math.PI) / 180;
      const theta = ((lng + rotY) * Math.PI) / 180;
      const x3 = R * Math.sin(phi) * Math.cos(theta);
      const y3 = R * Math.cos(phi);
      const z3 = R * Math.sin(phi) * Math.sin(theta);
      return { x: cx + x3, y: cy - y3, z: z3 };
    }

    // Stars (static, generated once)
    const stars: { x: number; y: number; r: number; a: number }[] = [];
    for (let i = 0; i < 220; i++) {
      stars.push({
        x: Math.random() * W,
        y: Math.random() * H,
        r: Math.random() * 1.2 + 0.2,
        a: Math.random() * 0.7 + 0.15,
      });
    }

    // Satellite dots on orbits
    const sats = [
      { orbitR: R * 1.35, speed: 0.012, tilt: 0.45, phase: 0,    size: 2.5 },
      { orbitR: R * 1.55, speed: 0.007, tilt: 1.15, phase: 2.1,  size: 2   },
      { orbitR: R * 1.75, speed: -0.009, tilt: 0.75, phase: 4.2, size: 1.8 },
    ];

    function drawFrame() {
      ctx!.clearRect(0, 0, W, H);

      // ── Stars ──────────────────────────────────────────────────────────────
      stars.forEach(s => {
        const twinkle = 0.5 + 0.5 * Math.sin(t * 0.8 + s.x);
        ctx!.beginPath();
        ctx!.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx!.fillStyle = P.star(s.a * twinkle);
        ctx!.fill();
      });

      const rotDeg = (t * 12) % 360; // degrees rotation

      // ── Atmosphere glow ────────────────────────────────────────────────────
      const grad = ctx!.createRadialGradient(cx, cy, R * 0.85, cx, cy, R * 1.18);
      grad.addColorStop(0, P.atmo);
      grad.addColorStop(1, "rgba(0,0,0,0)");
      ctx!.beginPath();
      ctx!.arc(cx, cy, R * 1.18, 0, Math.PI * 2);
      ctx!.fillStyle = grad;
      ctx!.fill();

      // ── Globe fill ─────────────────────────────────────────────────────────
      const globeGrad = ctx!.createRadialGradient(cx - R * 0.25, cy - R * 0.25, R * 0.05, cx, cy, R);
      globeGrad.addColorStop(0, P.globe[0]);
      globeGrad.addColorStop(0.5, P.globe[1]);
      globeGrad.addColorStop(1, P.globe[2]);
      ctx!.beginPath();
      ctx!.arc(cx, cy, R, 0, Math.PI * 2);
      ctx!.fillStyle = globeGrad;
      ctx!.fill();

      // ── Clip to globe for grid lines ───────────────────────────────────────
      ctx!.save();
      ctx!.beginPath();
      ctx!.arc(cx, cy, R, 0, Math.PI * 2);
      ctx!.clip();

      // Grid lines — only draw visible (z >= 0) segments
      const LATS = [-60, -30, 0, 30, 60];
      const LNGS = Array.from({ length: 12 }, (_, i) => i * 30);

      ctx!.lineWidth = 0.5;

      // Lat lines
      LATS.forEach(lat => {
        let first = true;
        for (let lng = 0; lng <= 360; lng += 3) {
          const p = latLngToXY(lat, lng, rotDeg);
          if (p.z < 0) { first = true; continue; }
          if (first) { ctx!.beginPath(); ctx!.moveTo(p.x, p.y); first = false; }
          else ctx!.lineTo(p.x, p.y);
        }
        ctx!.strokeStyle = P.grid1;
        ctx!.stroke();
      });

      // Lng lines
      LNGS.forEach(lng => {
        let first = true;
        for (let lat = -90; lat <= 90; lat += 3) {
          const p = latLngToXY(lat, lng, rotDeg);
          if (p.z < 0) { first = true; continue; }
          if (first) { ctx!.beginPath(); ctx!.moveTo(p.x, p.y); first = false; }
          else ctx!.lineTo(p.x, p.y);
        }
        ctx!.strokeStyle = P.grid2;
        ctx!.stroke();
      });

      ctx!.restore();

      // ── Globe border ───────────────────────────────────────────────────────
      ctx!.beginPath();
      ctx!.arc(cx, cy, R, 0, Math.PI * 2);
      ctx!.strokeStyle = P.border;
      ctx!.lineWidth = 1;
      ctx!.stroke();

      // ── Shine highlight ────────────────────────────────────────────────────
      const shine = ctx!.createRadialGradient(cx - R * 0.3, cy - R * 0.35, 0, cx - R * 0.1, cy - R * 0.15, R * 0.7);
      shine.addColorStop(0, P.shine);
      shine.addColorStop(1, "rgba(255,255,255,0)");
      ctx!.beginPath();
      ctx!.arc(cx, cy, R, 0, Math.PI * 2);
      ctx!.fillStyle = shine;
      ctx!.fill();

      // ── Orbit rings + satellite dots ───────────────────────────────────────
      sats.forEach(sat => {
        const r = sat.orbitR;
        // Elliptical projection of orbit
        const ry = r * Math.abs(Math.cos(sat.tilt));

        ctx!.beginPath();
        ctx!.ellipse(cx, cy, r, ry, 0, 0, Math.PI * 2);
        ctx!.strokeStyle = P.orbit;
        ctx!.lineWidth = 0.8;
        ctx!.stroke();

        // Satellite position
        const angle = t * sat.speed + sat.phase;
        const sx = cx + r * Math.cos(angle);
        const sy = cy + ry * Math.sin(angle);

        // Glow
        const gGrad = ctx!.createRadialGradient(sx, sy, 0, sx, sy, sat.size * 4);
        gGrad.addColorStop(0, P.satGlow(0.6));
        gGrad.addColorStop(1, P.satGlow(0));
        ctx!.beginPath();
        ctx!.arc(sx, sy, sat.size * 4, 0, Math.PI * 2);
        ctx!.fillStyle = gGrad;
        ctx!.fill();

        // Core dot
        ctx!.beginPath();
        ctx!.arc(sx, sy, sat.size, 0, Math.PI * 2);
        ctx!.fillStyle = P.satCore;
        ctx!.fill();
      });

      t += 0.016;
      animId = requestAnimationFrame(drawFrame);
    }

    drawFrame();

    const ro = new ResizeObserver(() => {
      canvas.width  = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    });
    ro.observe(canvas);

    return () => {
      cancelAnimationFrame(animId);
      ro.disconnect();
    };
  }, [isDark]);

  return (
    <canvas
      ref={canvasRef}
      className={`${className} pointer-events-none select-none`}
      style={{ background: "transparent" }}
      aria-hidden="true"
    />
  );
}
