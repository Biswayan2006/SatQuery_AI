"use client";

import { signIn } from "next-auth/react";

function SatelliteIcon({ size = 18, color = "currentColor" }: { size?: number; color?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2a2.236 2.236 0 0 0-3-3" />
      <path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z" />
      <path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0" />
      <path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5" />
    </svg>
  );
}

export default function LoginPage() {
  return (
    <div className="min-h-screen flex items-center justify-center px-4" style={{ background: "#080B0F" }}>
      <div className="w-full max-w-sm space-y-8">
        {/* Logo */}
        <div className="text-center space-y-4">
          <div className="w-14 h-14 mx-auto rounded-xl flex items-center justify-center" style={{ background: "rgba(81,141,178,0.15)", border: "1px solid rgba(81,141,178,0.35)" }}>
            <SatelliteIcon size={26} color="#9BD5E8" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">SatQuery AI</h1>
            <p className="text-sm text-white/50 mt-1 font-mono">PS 26167 &middot; ISRO / SAC</p>
          </div>
        </div>

        {/* Sign In Card */}
        <div className="rounded-xl p-6 space-y-5" style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)" }}>
          <div className="space-y-1">
            <h2 className="text-base font-semibold text-white">Sign in to continue</h2>
            <p className="text-xs text-white/40">Authenticate to access the analysis workspace</p>
          </div>

          <button
            type="button"
            onClick={() => signIn("google", { callbackUrl: "/app" })}
            className="w-full flex items-center justify-center gap-3 px-4 py-3 rounded-lg text-sm font-semibold transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#9BD5E8]"
            style={{ background: "#518DB2", color: "white" }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "#3D7396")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "#518DB2")}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" fill="#4285F4"/>
              <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
              <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
              <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
            </svg>
            Continue with Google
          </button>

          <div className="flex items-center gap-3">
            <div className="flex-1 h-px" style={{ background: "rgba(255,255,255,0.08)" }} />
            <span className="text-[10px] text-white/30 font-mono uppercase tracking-wider">secure</span>
            <div className="flex-1 h-px" style={{ background: "rgba(255,255,255,0.08)" }} />
          </div>

          <p className="text-[11px] text-center text-white/30 leading-relaxed">
            By signing in, you agree to use SatQuery AI for authorized remote-sensing analysis purposes.
          </p>
        </div>

        {/* Footer */}
        <p className="text-center text-[11px] text-white/20 font-mono">
          Smart India Hackathon 2026 &middot; Problem Statement 26167
        </p>
      </div>
    </div>
  );
}
