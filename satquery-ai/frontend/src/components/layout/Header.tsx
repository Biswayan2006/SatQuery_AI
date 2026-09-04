"use client";

import { usePathname } from "next/navigation";
import { Search, Bell, User } from "lucide-react";
import ThemeToggle from "@/components/theme/ThemeToggle";

const PAGE_TITLES: Record<string, string> = {
  "/":         "SatQuery AI: Interactive Vision-Language Assistant",
  "/history":  "Execution History",
  "/datasets": "Datasets",
  "/reports":  "Generated Reports",
  "/settings": "Settings",
};

export default function Header() {
  const pathname = usePathname();
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
      <h1
        className="text-[17px] font-semibold truncate pr-6 leading-none"
        style={{ color: "var(--text-primary)" }}
      >
        {title}
      </h1>

      {/* Right controls */}
      <div className="flex items-center gap-3 flex-shrink-0">
        {/* Global search */}
        <div
          className="hidden lg:flex items-center gap-2.5 px-3.5 py-2.5 rounded-lg h-10"
          style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", minWidth: "260px" }}
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

        {/* Notifications */}
        <button
          className="relative flex items-center justify-center w-10 h-10 rounded-lg
                     text-ink-muted hover:text-ink hover:bg-raised transition-colors no-tap"
          style={{ border: "1px solid var(--border)" }}
          aria-label="Notifications"
        >
          <Bell className="w-[18px] h-[18px]" />
          <span
            className="absolute top-2 right-2 w-2 h-2 rounded-full"
            style={{ background: "var(--accent)" }}
          />
        </button>

        {/* User avatar */}
        <button
          className="flex items-center justify-center w-10 h-10 rounded-full no-tap hover:opacity-90 transition-opacity"
          style={{ background: "var(--accent-soft)", border: "1px solid var(--accent-border)" }}
          aria-label="Account"
        >
          <User className="w-5 h-5" style={{ color: "var(--accent)" }} />
        </button>
      </div>
    </header>
  );
}
