"use client";

import { useState } from "react";
import Sidebar from "./Sidebar";
import Header from "./Header";
import MobileBottomNav from "./MobileBottomNav";
import MobileHeader from "./MobileHeader";

export default function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen" style={{ background: "var(--bg-base)" }}>
      {/* Persistent left sidebar: desktop only */}
      <Sidebar />

      {/* Top header: desktop only (md+) */}
      <Header />

      {/* Top bar: mobile only (< md) */}
      <MobileHeader />

      {/* ── Main content ───────────────────────────────────────────────────
          md+  : offset left by sidebar width (w-64 = 256px),
                 offset top by header height (h-16 = 64px)
          < md : offset top by mobile header height (h-14 = 56px),
                 add bottom padding for the bottom nav bar
      ────────────────────────────────────────────────────────────────────── */}
      <main
        className="
          md:ml-64 md:pt-16
          pt-14 pb-20 md:pb-0
          min-h-screen
        "
      >
        {/* Inner content wrapper: matches subtle frame in reference */}
        <div
          className="p-4 md:p-5 lg:p-6 min-h-[calc(100vh-4rem)]"
          style={{ background: "var(--bg-base)" }}
        >
          {children}
        </div>
      </main>

      {/* Bottom tab bar: mobile only */}
      <MobileBottomNav />
    </div>
  );
}
