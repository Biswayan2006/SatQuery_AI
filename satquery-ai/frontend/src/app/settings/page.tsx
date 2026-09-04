"use client";
import { useState } from "react";
import { motion } from "framer-motion";
import { User, Monitor, Database, Shield, RefreshCw, ToggleLeft, ToggleRight } from "lucide-react";
import { ThemeSegment } from "@/components/theme/ThemeToggle";

type Tab = "profile" | "application" | "models" | "security";

const TABS = [
  { id: "profile"     as Tab, icon: User,     label: "Profile"     },
  { id: "application" as Tab, icon: Monitor,  label: "Application" },
  { id: "models"      as Tab, icon: Database, label: "Data Models" },
  { id: "security"    as Tab, icon: Shield,   label: "Security"    },
];

interface ModelToggles {
  vqa: boolean;
  captioning: boolean;
  grounding: boolean;
  change: boolean;
  sar: boolean;
}

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<Tab>("profile");
  const [emailNotif, setEmailNotif] = useState(true);
  const [models, setModels] = useState<ModelToggles>({
    vqa: true, captioning: true, grounding: true, change: true, sar: true,
  });

  return (
    <div className="max-w-4xl mx-auto space-y-5 animate-fade-in">
      <div>
        <h1 className="text-lg font-bold text-ink">Settings</h1>
        <p className="text-xs text-ink-muted mt-0.5">Configure your SatQuery AI environment</p>
      </div>

      <div className="flex flex-col md:flex-row gap-5">
        {/* Tab list */}
        <div className="md:w-48 flex-shrink-0">
          <nav className="card p-2 flex md:flex-col gap-1">
            {TABS.map(tab => {
              const Icon = tab.icon;
              const active = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-semibold transition-all no-tap text-left w-full ${
                    active
                      ? "bg-accent text-accent-contrast"
                      : "text-ink-muted hover:text-ink hover:bg-raised"
                  }`}
                >
                  <Icon className="w-4 h-4 flex-shrink-0" />
                  <span className="hidden md:inline">{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Tab content */}
        <div className="flex-1 card p-6">
          {activeTab === "profile" && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-5">
              <h2 className="text-sm font-bold text-ink">Profile</h2>
              <div className="flex items-center gap-4">
                <div
                  className="w-16 h-16 rounded-full flex items-center justify-center text-2xl font-bold"
                  style={{
                    background: "color-mix(in srgb, var(--accent) 16%, transparent)",
                    border: "2px solid var(--accent-border)",
                    color: "var(--accent)",
                  }}
                >
                  IS
                </div>
                <div>
                  <p className="text-sm font-bold text-ink">ISRO Scientist</p>
                  <p className="text-xs text-ink-muted">scientist@isro.gov.in</p>
                  <button className="text-xs text-accent mt-1 hover:underline no-tap">
                    Change photo
                  </button>
                </div>
              </div>
              {[
                { label: "Full Name",     value: "ISRO Scientist"            },
                { label: "Role",          value: "Remote Sensing Researcher" },
                { label: "Organization",  value: "ISRO / SAC"                },
                { label: "Email",         value: "scientist@isro.gov.in"     },
              ].map(f => (
                <div key={f.label}>
                  <label className="block text-xs font-semibold text-ink-muted mb-1.5">
                    {f.label}
                  </label>
                  <input
                    defaultValue={f.value}
                    className="w-full px-3 py-2.5 rounded-xl bg-raised border border-line text-sm text-ink outline-none focus:border-accent-border transition-colors"
                  />
                </div>
              ))}
              <button className="px-5 py-2.5 rounded-xl bg-accent hover:bg-accent-hover text-accent-contrast text-sm font-bold transition-all no-tap">
                Save Changes
              </button>
            </motion.div>
          )}

          {activeTab === "application" && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-5">
              <h2 className="text-sm font-bold text-ink">Application</h2>
              <SettingRow label="Theme" desc="Application colour theme">
                <ThemeSegment />
              </SettingRow>
              <SettingRow label="Email Notifications" desc="Receive email on report completion">
                <Toggle value={emailNotif} onChange={setEmailNotif} />
              </SettingRow>
              <SettingRow
                label="Auto-advance on upload"
                desc="Automatically move to query step after upload"
              >
                <Toggle value={true} onChange={() => {}} />
              </SettingRow>
            </motion.div>
          )}

          {activeTab === "models" && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-5">
              <h2 className="text-sm font-bold text-ink">Data Models</h2>
              <div className="p-3 rounded-xl bg-raised border border-line">
                <p className="text-xs font-bold text-accent mb-1">RS Fine-tuning Status</p>
                <p className="text-xs text-ink-muted">Last adapted: Sep 1, 2026, BigEarthNet v2.0</p>
                <button className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink-soft mt-2 transition-colors no-tap">
                  <RefreshCw className="w-3 h-3" />Re-run fine-tuning
                </button>
              </div>
              <div className="space-y-3">
                {[
                  { key: "vqa",        label: "Visual Question Answering", model: "blip-vqa-base"         },
                  { key: "captioning", label: "Scene Captioning",          model: "blip-image-captioning" },
                  { key: "grounding",  label: "Text-Guided Grounding",     model: "owlvit-base-patch32"   },
                  { key: "change",     label: "Change Detection",          model: "ResNet-50 (Siamese)"   },
                  { key: "sar",        label: "SAR-Optical Fusion",        model: "Dual ResNet-50 + MLP"  },
                ].map(m => (
                  <div
                    key={m.key}
                    className="flex items-center justify-between p-3 rounded-xl bg-raised border border-line"
                  >
                    <div>
                      <p className="text-xs font-semibold text-ink">{m.label}</p>
                      <p className="text-[10px] text-ink-faint font-mono">{m.model}</p>
                    </div>
                    <Toggle
                      value={models[m.key as keyof ModelToggles]}
                      onChange={v => setModels(p => ({ ...p, [m.key]: v }))}
                    />
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {activeTab === "security" && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-5">
              <h2 className="text-sm font-bold text-ink">Security</h2>
              <div>
                <label className="block text-xs font-semibold text-ink-muted mb-1.5">
                  Current Password
                </label>
                <input
                  type="password"
                  className="w-full px-3 py-2.5 rounded-xl bg-raised border border-line text-sm text-ink outline-none focus:border-accent-border transition-colors"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-ink-muted mb-1.5">
                  New Password
                </label>
                <input
                  type="password"
                  className="w-full px-3 py-2.5 rounded-xl bg-raised border border-line text-sm text-ink outline-none focus:border-accent-border transition-colors"
                />
              </div>
              <div className="p-4 rounded-xl bg-raised border border-line">
                <p className="text-xs font-bold text-ink mb-1">API Key</p>
                <div className="flex items-center gap-2 mt-2">
                  <code className="flex-1 text-[10px] text-ink-muted font-mono bg-inset rounded-lg px-3 py-2 truncate border border-line">
                    sq_live_••••••••••••••••••••••
                  </code>
                  <button className="px-3 py-2 rounded-lg bg-raised border border-line text-xs text-ink-muted hover:text-ink transition-colors no-tap">
                    Reveal
                  </button>
                  <button className="px-3 py-2 rounded-lg bg-raised border border-line text-xs text-ink-muted hover:text-ink transition-colors no-tap">
                    Regenerate
                  </button>
                </div>
              </div>
              <button className="px-5 py-2.5 rounded-xl bg-accent hover:bg-accent-hover text-accent-contrast text-sm font-bold transition-all no-tap">
                Update Password
              </button>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Shared sub-components ─────────────────────────────────────────────────────

function SettingRow({
  label,
  desc,
  children,
}: {
  label: string;
  desc: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div>
        <p className="text-xs font-semibold text-ink">{label}</p>
        <p className="text-[10px] text-ink-muted mt-0.5">{desc}</p>
      </div>
      {children}
    </div>
  );
}

function Toggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button onClick={() => onChange(!value)} className="flex-shrink-0 no-tap">
      {value ? (
        <ToggleRight className="w-7 h-7 text-accent" />
      ) : (
        <ToggleLeft className="w-7 h-7 text-ink-faint" />
      )}
    </button>
  );
}
