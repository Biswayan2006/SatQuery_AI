"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X, ChevronDown, User } from "lucide-react";
import { useSession } from "next-auth/react";
import ThemeToggle from "@/components/theme/ThemeToggle";
import NotificationsPopover from "@/components/layout/NotificationsPopover";
import ProjectSelector from "@/components/layout/ProjectSelector";

const NAV_ITEMS = [
  { href: "/app",      label: "Analyze New Query"  },
  { href: "/history",  label: "Execution History"  },
  { href: "/datasets", label: "Datasets"           },
  { href: "/reports",  label: "Reports"            },
  { href: "/settings", label: "Settings"           },
];

export default function MobileHeader() {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();
  const { data: session } = useSession();

  return (
    <>
      {/* Top bar */}
      <div
        className="md:hidden fixed top-0 inset-x-0 z-50 h-14 flex items-center justify-between px-4 transition-colors duration-300"
        style={{
          background: "color-mix(in srgb, var(--bg-base) 88%, transparent)",
          borderBottom: "1px solid var(--border)",
          backdropFilter: "blur(12px)",
          WebkitBackdropFilter: "blur(12px)",
        }}
      >
        {/* Hamburger */}
        <button
          onClick={() => setOpen(true)}
          className="w-10 h-10 flex items-center justify-center rounded-lg no-tap
                     text-ink-muted hover:text-ink hover:bg-raised transition-colors"
          style={{ border: "1px solid var(--border)" }}
          aria-label="Open navigation"
        >
          <Menu className="w-5 h-5" />
        </button>

        {/* Logo */}
        <span className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
          SatQuery AI
        </span>

        {/* Theme, Notifications, avatar */}
        <div className="flex items-center gap-2">
          <ThemeToggle />
          <NotificationsPopover />
          <Link
            href="/settings"
            className="w-9 h-9 rounded-full flex items-center justify-center no-tap"
            style={{ background: "var(--accent-soft)", border: "1px solid var(--accent-border)" }}
            aria-label="Account Settings"
          >
            <User className="w-4 h-4" style={{ color: "var(--accent)" }} />
          </Link>
        </div>
      </div>

      {/* Overlay */}
      {open && (
        <div
          className="md:hidden fixed inset-0 z-[60] backdrop-blur-sm"
          style={{ background: "var(--overlay)" }}
          onClick={() => setOpen(false)}
        />
      )}

      {/* Slide-over drawer */}
      {open && (
        <div
          className="md:hidden fixed inset-y-0 left-0 z-[70] w-72 flex flex-col"
          style={{
            background: "var(--bg-base)",
            borderRight: "1px solid var(--border)",
            animation: "slideIn 0.2s ease-out both",
          }}
        >
          {/* Drawer header */}
          <div
            className="flex items-center justify-between px-5 h-14 flex-shrink-0"
            style={{ borderBottom: "1px solid var(--border)" }}
          >
            <span className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
              SatQuery AI
            </span>
            <button
              onClick={() => setOpen(false)}
              className="w-9 h-9 flex items-center justify-center rounded-lg no-tap
                         text-ink-muted hover:text-ink hover:bg-raised transition-colors"
              aria-label="Close navigation"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Project selector */}
          <div className="px-4 py-4" style={{ borderBottom: "1px solid var(--border)" }}>
            <p className="text-xs font-medium mb-2" style={{ color: "var(--text-muted)" }}>
              Project
            </p>
            <ProjectSelector />
          </div>

          {/* Nav */}
          <nav className="flex-1 px-3 py-3 space-y-0.5 overflow-y-auto scrollbar-none">
            {NAV_ITEMS.map(({ href, label }) => {
              const active = pathname === href;
              return (
                <Link key={href} href={href} onClick={() => setOpen(false)} className="no-tap block">
                  <div
                    className={`flex items-center gap-3 px-3 py-3 rounded-lg text-sm font-medium transition-colors ${
                      active ? "" : "hover:bg-surface"
                    }`}
                    style={
                      active
                        ? { background: "var(--accent)", color: "var(--accent-contrast)" }
                        : { color: "var(--text-muted)" }
                    }
                  >
                    {label}
                  </div>
                </Link>
              );
            })}
          </nav>

          {/* User */}
          <div className="px-3 py-3" style={{ borderTop: "1px solid var(--border)" }}>
            {session ? (
              <div className="flex items-center gap-3 px-2 py-2.5">
                <div
                  className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0"
                  style={{ background: "var(--accent-soft)", border: "1px solid var(--accent-border)" }}
                >
                  <User className="w-4 h-4" style={{ color: "var(--accent)" }} />
                </div>
                <div className="leading-tight">
                  <p className="text-[11px]" style={{ color: "var(--text-muted)" }}>Signed in</p>
                  <p className="text-sm font-semibold" style={{ color: "var(--text-secondary)" }}>{session.user?.name ?? session.user?.email ?? "User"}</p>
                </div>
              </div>
            ) : (
              <Link href="/login" onClick={() => setOpen(false)} className="flex items-center gap-3 px-2 py-2.5 no-tap rounded-lg transition-colors" style={{ color: "var(--text-muted)" }}>
                <div
                  className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0"
                  style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
                >
                  <User className="w-4 h-4" style={{ color: "var(--text-muted)" }} />
                </div>
                <div className="leading-tight">
                  <p className="text-[11px]" style={{ color: "var(--text-faint)" }}>Guest mode</p>
                  <p className="text-sm font-semibold" style={{ color: "var(--text-muted)" }}>Sign in with Google</p>
                </div>
              </Link>
            )}
          </div>
        </div>
      )}
    </>
  );
}
