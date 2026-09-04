"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import { SlidersHorizontal } from "lucide-react";

interface Props {
  beforeSrc: string; // base64 or URL
  afterSrc: string;
  beforeLabel?: string;
  afterLabel?: string;
  className?: string;
}

export default function CompareSlider({
  beforeSrc, afterSrc,
  beforeLabel = "Before", afterLabel = "After",
  className = "",
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState(50); // percentage
  const dragging = useRef(false);

  const update = useCallback((clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const pct = Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100));
    setPos(pct);
  }, []);

  const onMouseDown = (e: React.MouseEvent) => { dragging.current = true; update(e.clientX); };
  const onTouchStart = (e: React.TouchEvent) => { dragging.current = true; update(e.touches[0].clientX); };

  useEffect(() => {
    const up = () => { dragging.current = false; };
    const move = (e: MouseEvent) => { if (dragging.current) update(e.clientX); };
    const touch = (e: TouchEvent) => { if (dragging.current) update(e.touches[0].clientX); };
    window.addEventListener("mouseup", up);
    window.addEventListener("mousemove", move);
    window.addEventListener("touchend", up);
    window.addEventListener("touchmove", touch, { passive: true });
    return () => {
      window.removeEventListener("mouseup", up);
      window.removeEventListener("mousemove", move);
      window.removeEventListener("touchend", up);
      window.removeEventListener("touchmove", touch);
    };
  }, [update]);

  const isSrc = (s: string) => s.startsWith("http") || s.startsWith("/") || s.startsWith("data:");
  const beforeImg = isSrc(beforeSrc) ? beforeSrc : `data:image/png;base64,${beforeSrc}`;
  const afterImg  = isSrc(afterSrc)  ? afterSrc  : `data:image/png;base64,${afterSrc}`;

  return (
    <div
      ref={containerRef}
      className={`relative overflow-hidden rounded-lg select-none cursor-col-resize ${className}`}
      style={{
        background: "var(--bg-inset)",
        border: "1px solid var(--border)",
      }}
      onMouseDown={onMouseDown}
      onTouchStart={onTouchStart}
    >
      {/* After (full) */}
      <img src={afterImg} alt={afterLabel} className="w-full h-full object-cover block" draggable={false} />

      {/* Before (clipped) */}
      <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        <img src={beforeImg} alt={beforeLabel} className="w-full h-full object-cover block" draggable={false} />
      </div>

      {/* Divider line */}
      <div
        className="absolute top-0 bottom-0"
        style={{
          left: `${pos}%`,
          transform: "translateX(-50%)",
          width: "2px",
          background: "rgba(255,255,255,0.85)",
          boxShadow: "0 0 0 1px rgba(0,0,0,0.3)",
        }}
      >
        {/* Handle */}
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full flex items-center justify-center"
          style={{
            background: "rgba(255,255,255,0.92)",
            border: "2px solid color-mix(in srgb, var(--accent) 65%, transparent)",
            boxShadow: "0 2px 8px rgba(0,0,0,0.4), 0 0 0 1px rgba(0,0,0,0.2)",
          }}
        >
          <SlidersHorizontal className="w-3.5 h-3.5" style={{ color: "#161B22" }} />
        </div>
      </div>

      {/* Labels */}
      <span
        className="absolute top-2 left-3 text-[10px] font-bold rounded-full"
        style={{
          color: "#fff",
          background: "rgba(0,0,0,0.55)",
          backdropFilter: "blur(4px)",
          padding: "2px 8px",
        }}
      >
        {beforeLabel}
      </span>
      <span
        className="absolute top-2 right-3 text-[10px] font-bold rounded-full"
        style={{
          color: "#fff",
          background: "rgba(0,0,0,0.55)",
          backdropFilter: "blur(4px)",
          padding: "2px 8px",
        }}
      >
        {afterLabel}
      </span>
    </div>
  );
}
