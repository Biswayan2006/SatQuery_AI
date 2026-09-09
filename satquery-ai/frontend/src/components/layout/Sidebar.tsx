"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  ChevronDown,
  Folder,
  History,
  Database,
  FileText,
  Settings,
  User,
} from "lucide-react";
import ProjectSelector from "@/components/layout/ProjectSelector";

/* ── Navigation items ───────────────────────────────────────────────────── */
const NAV_ITEMS = [
  { href: "/app",      label: "Analyze New Query"  },
  { href: "/history",  label: "Execution History"  },
  { href: "/datasets", label: "Datasets"           },
  { href: "/reports",  label: "Reports"            },
  { href: "/settings", label: "Settings"           },
] as const;

const NAV_ICONS: Record<string, React.ReactNode> = {
  "/app":      <SatelliteIcon />,
  "/history":  <History   className="w-[18px] h-[18px] flex-shrink-0" />,
  "/datasets": <Database  className="w-[18px] h-[18px] flex-shrink-0" />,
  "/reports":  <FileText  className="w-[18px] h-[18px] flex-shrink-0" />,
  "/settings": <Settings  className="w-[18px] h-[18px] flex-shrink-0" />,
};

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside
      className="hidden md:flex flex-col fixed inset-y-0 left-0 z-50 w-64 select-none transition-colors duration-300"
      style={{ background: "var(--bg-base)", borderRight: "1px solid var(--border)" }}
    >
      {/* ── Logo ─────────────────────────────────────────────────────────── */}
      <div
        className="flex items-center gap-3 px-5 h-16 flex-shrink-0"
        style={{ borderBottom: "1px solid var(--border)" }}
      >
        <div
          className="w-9 h-9 flex-shrink-0 flex items-center justify-center rounded-lg"
          style={{ background: "var(--accent-soft)", border: "1px solid var(--accent-border)" }}
        >
          <SatelliteIcon size={20} color="var(--accent)" />
        </div>
        <span
          className="text-[17px] font-semibold tracking-tightest leading-none"
          style={{ color: "var(--text-primary)" }}
        >
          SatQuery AI
        </span>
      </div>

      {/* ── Project selector ─────────────────────────────────────────────── */}
      <div className="px-4 py-4 flex-shrink-0" style={{ borderBottom: "1px solid var(--border)" }}>
        <p className="text-xs font-medium mb-2 px-1" style={{ color: "var(--text-muted)" }}>
          Project
        </p>
        <ProjectSelector />
      </div>

      {/* ── Navigation ───────────────────────────────────────────────────── */}
      <nav className="flex-1 overflow-y-auto scrollbar-none px-3 py-3 space-y-0.5">
        {NAV_ITEMS.map(({ href, label }) => {
          const active = pathname === href;
          return (
            <Link key={href} href={href} className="no-tap block">
              <div
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors duration-100 ${
                  active ? "" : "hover:bg-surface"
                }`}
                style={
                  active
                    ? { background: "var(--accent)", color: "var(--accent-contrast)" }
                    : { color: "var(--text-muted)" }
                }
              >
                <span style={{ color: active ? "var(--accent-contrast)" : "var(--text-muted)" }}>
                  {NAV_ICONS[href]}
                </span>
                {label}
              </div>
            </Link>
          );
        })}
      </nav>

      {/* ── Bottom user row with online status ─────────────────────────── */}
      <div className="sidebar-bottom-map flex-shrink-0 relative">
        <div className="px-3 py-3" style={{ borderTop: "1px solid var(--border)" }}>
          <Link
            href="/settings"
            className="w-full flex items-center gap-3 px-2 py-2.5 rounded-lg hover:bg-raised transition-colors no-tap block"
          >
            {/* Avatar with attached online status dot */}
            <div className="relative flex-shrink-0">
              <div
                className="w-9 h-9 rounded-full flex items-center justify-center"
                style={{ background: "var(--accent-soft)", border: "1px solid var(--accent-border)" }}
              >
                <User className="w-4 h-4" style={{ color: "var(--accent)" }} />
              </div>
              <span
                className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full ring-2"
                style={{
                  background: "var(--veg)",
                  boxShadow: "0 0 6px var(--veg)",
                  borderColor: "var(--bg-base)",
                }}
                title="Online"
              />
            </div>
            <div className="flex-1 min-w-0 text-left leading-tight">
              <div className="flex items-center gap-1.5">
                <span className="text-[11px] font-medium" style={{ color: "var(--text-muted)" }}>User</span>
                <span
                  className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] font-semibold"
                  style={{
                    background: "var(--veg-soft)",
                    color: "var(--veg)",
                    border: "1px solid color-mix(in srgb, var(--veg) 30%, transparent)",
                  }}
                >
                  <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: "var(--veg)" }} />
                  Online
                </span>
              </div>
              <p className="text-sm font-semibold mt-0.5 truncate" style={{ color: "var(--text-secondary)" }}>
                ISRO Scientist
              </p>
            </div>
            <ChevronDown className="w-4 h-4 flex-shrink-0" style={{ color: "var(--text-muted)" }} />
          </Link>
        </div>
      </div>
    </aside>
  );
}

/* ── Satellite SVG icon (inline: avoids import conflicts) ───────────────── */
function SatelliteIcon({ size = 18, color = "currentColor" }: { size?: number; color?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="flex-shrink-0"
    >
      <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2a2.236 2.236 0 0 0-3-3" />
      <path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z" />
      <path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0" />
      <path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5" />
    </svg>
  );
}
