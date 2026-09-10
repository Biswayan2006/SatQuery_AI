"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Search, User } from "lucide-react";
import { useSession } from "next-auth/react";
import ThemeToggle from "@/components/theme/ThemeToggle";
import NotificationsPopover from "@/components/layout/NotificationsPopover";

const PAGE_TITLES: Record<string, string> = {
  "/app":      "SatQuery AI: Interactive Vision-Language Assistant",
  "/history":  "Execution History",
  "/datasets": "Datasets",
  "/reports":  "Generated Reports",
  "/settings": "Settings",
};

export default function Header() {
  const pathname = usePathname();
  const { data: session } = useSession();
  const title = PAGE_TITLES[pathname] ?? "SatQuery AI";

  return (
    <header
      className="hidden md:flex fixed top-0 right-0 z-40 h-16 items-center justify-between px-5 lg:px-7 transition-colors duration-300"
      style={{
        left: "256px", /* matches sidebar w-64 */
        background: "color-mix(in srgb, var(--bg-base) 82%, transparent)",
        borderBottom: "1px solid var(--border)",
        backdropFilter: "blur(12px)",
        WebkitBackdropFilter: "blur(12px)",
      }}
    >
      {/* Page title */}
      <div className="flex items-center gap-4 min-w-0 pr-4">
        <h1
          className="text-[17px] font-semibold truncate leading-none"
          style={{ color: "var(--text-primary)" }}
        >
          {title}
        </h1>
      </div>

      {/* Right controls */}
      <div className="flex items-center gap-3 flex-shrink-0">
        {/* Global search */}
        <div
          className="hidden lg:flex items-center gap-2.5 px-3.5 py-2.5 rounded-lg h-10"
          style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", minWidth: "220px" }}
        >
          <Search className="w-4 h-4 flex-shrink-0" style={{ color: "var(--text-muted)" }} />
          <input
            type="text"
            placeholder="Global search"
            className="bg-transparent text-sm text-ink-soft placeholder:text-ink-faint outline-none w-full"
          />
        </div>

        {/* Theme toggle */}
        <ThemeToggle />

        {/* Notifications Popover */}
        <NotificationsPopover />

        {/* Auth status indicator */}
        {session ? (
          <span
            className="hidden lg:inline-flex items-center gap-1.5 text-[10px] font-mono px-2 py-1 rounded"
            style={{ color: "var(--veg)", background: "var(--veg-soft)", border: "1px solid color-mix(in srgb, var(--veg) 25%, transparent)" }}
          >
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: "var(--veg)" }} />
            Signed in
          </span>
        ) : (
          <Link
            href="/login"
            className="hidden lg:inline-flex items-center gap-1.5 text-[10px] font-mono px-2 py-1 rounded transition-colors duration-150"
            style={{ color: "var(--text-muted)", background: "var(--bg-surface)", border: "1px solid var(--border)" }}
          >
            Guest mode
          </Link>
        )}

        {/* User avatar linking to settings */}
        <Link
          href="/settings"
          className="relative flex items-center justify-center w-10 h-10 rounded-full no-tap hover:opacity-90 transition-opacity"
          style={{ background: "var(--accent-soft)", border: "1px solid var(--accent-border)" }}
          aria-label="Account Settings"
          title="Account settings (Online)"
        >
          <User className="w-5 h-5" style={{ color: "var(--accent)" }} />
          <span
            className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full ring-2"
            style={{
              background: "var(--veg)",
              boxShadow: "0 0 6px var(--veg)",
              borderColor: "var(--bg-base)",
            }}
          />
        </Link>
      </div>
    </header>
  );
}
