"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { History, Database, FileText, Settings } from "lucide-react";

/* Inline SVG satellite for consistency */
function SatIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2a2.236 2.236 0 0 0-3-3" />
      <path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z" />
      <path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0" />
      <path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5" />
    </svg>
  );
}

const TABS = [
  { href: "/",         label: "Analyze",  Icon: SatIcon   },
  { href: "/history",  label: "History",  Icon: History   },
  { href: "/datasets", label: "Data",     Icon: Database  },
  { href: "/reports",  label: "Reports",  Icon: FileText  },
  { href: "/settings", label: "Settings", Icon: Settings  },
] as const;

export default function MobileBottomNav() {
  const pathname = usePathname();

  return (
    <nav
      className="md:hidden fixed bottom-0 inset-x-0 z-50 safe-bottom transition-colors duration-300"
      style={{
        background: "color-mix(in srgb, var(--bg-base) 90%, transparent)",
        borderTop: "1px solid var(--border)",
        backdropFilter: "blur(12px)",
        WebkitBackdropFilter: "blur(12px)",
      }}
    >
      <div className="flex items-center justify-around h-16">
        {TABS.map(({ href, label, Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className="no-tap flex flex-col items-center gap-1 px-3 py-2 rounded-lg
                         transition-colors min-w-[52px]"
            >
              <span style={{ color: active ? "var(--accent)" : "var(--text-muted)" }}>
                <Icon />
              </span>
              <span
                className="text-[10px] font-semibold"
                style={{ color: active ? "var(--accent)" : "var(--text-muted)" }}
              >
                {label}
              </span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
