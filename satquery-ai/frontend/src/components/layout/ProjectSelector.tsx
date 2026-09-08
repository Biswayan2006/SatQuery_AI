"use client";

import { useState, useRef, useEffect } from "react";
import { Folder, ChevronDown, Check, Plus, Layers, X } from "lucide-react";
import { useProject } from "@/context/ProjectContext";

export default function ProjectSelector() {
  const { projects, activeProject, setActiveProjectId, createProject } = useProject();
  const [open, setOpen] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [newModality, setNewModality] = useState("Optical");
  const [newSensor, setNewSensor] = useState("Sentinel-2");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleOutside = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    if (open) document.addEventListener("mousedown", handleOutside);
    return () => document.removeEventListener("mousedown", handleOutside);
  }, [open]);

  const handleCreate = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newProjectName.trim()) return;
    createProject(newProjectName.trim(), newModality, newSensor);
    setNewProjectName("");
    setShowModal(false);
    setOpen(false);
  };

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((prev) => !prev)}
        className="w-full flex items-center justify-between gap-2 px-3 py-2.5 rounded-lg
                   text-sm font-medium transition-colors duration-150 hover:bg-raised no-tap"
        style={{
          border: "1px solid var(--border)",
          background: "var(--bg-raised)",
          color: "var(--text-secondary)",
        }}
        aria-expanded={open}
        aria-label="Select project"
      >
        <span className="flex items-center gap-2 truncate">
          <Folder className="w-4 h-4 flex-shrink-0" style={{ color: "var(--accent)" }} />
          <span className="truncate font-semibold">{activeProject.name}</span>
        </span>
        <ChevronDown
          className={`w-4 h-4 flex-shrink-0 transition-transform duration-200 ${
            open ? "rotate-180" : ""
          }`}
          style={{ color: "var(--text-muted)" }}
        />
      </button>

      {/* Dropdown Menu */}
      {open && (
        <div
          className="absolute left-0 right-0 mt-1.5 rounded-xl shadow-2xl z-50 overflow-hidden animate-fade-in"
          style={{
            background: "var(--bg-elevated)",
            border: "1px solid var(--border)",
            boxShadow: "0 10px 25px rgba(0,0,0,0.35)",
          }}
        >
          <div className="p-1.5 space-y-0.5 max-h-60 overflow-y-auto">
            <div className="px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
              Switch Workspace
            </div>
            {projects.map((proj) => {
              const isSelected = proj.id === activeProject.id;
              return (
                <button
                  key={proj.id}
                  onClick={() => {
                    setActiveProjectId(proj.id);
                    setOpen(false);
                  }}
                  className={`w-full flex items-center justify-between gap-2 px-2.5 py-2 rounded-lg text-left text-xs transition-colors no-tap ${
                    isSelected
                      ? "bg-accent text-accent-contrast font-semibold"
                      : "hover:bg-raised text-ink-soft hover:text-ink"
                  }`}
                >
                  <div className="min-w-0 flex-1">
                    <p className="truncate leading-snug">{proj.name}</p>
                    <p
                      className="text-[10px] leading-tight truncate mt-0.5 opacity-80"
                      style={{
                        color: isSelected ? "var(--accent-contrast)" : "var(--text-muted)",
                      }}
                    >
                      {proj.sensor} ({proj.modality})
                    </p>
                  </div>
                  {isSelected && <Check className="w-3.5 h-3.5 flex-shrink-0" />}
                </button>
              );
            })}
          </div>

          <div
            className="p-1.5"
            style={{ borderTop: "1px solid var(--border)", background: "var(--bg-surface)" }}
          >
            <button
              onClick={() => {
                setOpen(false);
                setShowModal(true);
              }}
              className="w-full flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold text-accent hover:bg-raised transition-colors no-tap"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Create New Project</span>
            </button>
          </div>
        </div>
      )}

      {/* New Project Modal */}
      {showModal && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center p-4 backdrop-blur-sm animate-fade-in"
          style={{ background: "var(--overlay)" }}
          onClick={() => setShowModal(false)}
        >
          <div
            className="w-full max-w-md rounded-xl p-5 shadow-2xl space-y-4"
            style={{
              background: "var(--bg-elevated)",
              border: "1px solid var(--border)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Folder className="w-5 h-5 text-accent" />
                <h3 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
                  Create New Project
                </h3>
              </div>
              <button
                onClick={() => setShowModal(false)}
                className="p-1 rounded-lg hover:bg-raised text-ink-muted transition-colors no-tap"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleCreate} className="space-y-3.5">
              <div>
                <label className="block text-xs font-semibold mb-1" style={{ color: "var(--text-secondary)" }}>
                  Project Name
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Coastal Mangrove Survey"
                  value={newProjectName}
                  onChange={(e) => setNewProjectName(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg text-xs outline-none transition-colors"
                  style={{
                    background: "var(--bg-surface)",
                    border: "1px solid var(--border)",
                    color: "var(--text-primary)",
                  }}
                  autoFocus
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold mb-1" style={{ color: "var(--text-secondary)" }}>
                    Primary Modality
                  </label>
                  <select
                    value={newModality}
                    onChange={(e) => setNewModality(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg text-xs outline-none"
                    style={{
                      background: "var(--bg-surface)",
                      border: "1px solid var(--border)",
                      color: "var(--text-primary)",
                    }}
                  >
                    <option value="Optical">Optical (RGB)</option>
                    <option value="SAR">SAR (Radar)</option>
                    <option value="Multispectral">Multispectral</option>
                    <option value="SAR + Optical">SAR + Optical Fusion</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-semibold mb-1" style={{ color: "var(--text-secondary)" }}>
                    Sensor Constellation
                  </label>
                  <select
                    value={newSensor}
                    onChange={(e) => setNewSensor(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg text-xs outline-none"
                    style={{
                      background: "var(--bg-surface)",
                      border: "1px solid var(--border)",
                      color: "var(--text-primary)",
                    }}
                  >
                    <option value="Sentinel-2">Sentinel-2 (MSI)</option>
                    <option value="Sentinel-1">Sentinel-1 (C-band SAR)</option>
                    <option value="Landsat-8/9">Landsat-8/9 (OLI/TIRS)</option>
                    <option value="Cartosat-3">Cartosat-3 (High-res)</option>
                    <option value="RISAT-1A">RISAT-1A (C-band SAR)</option>
                  </select>
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowModal(false)}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold text-ink-muted hover:text-ink hover:bg-raised transition-colors no-tap"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-semibold bg-accent text-accent-contrast hover:bg-accent-hover transition-colors no-tap"
                >
                  <Check className="w-3.5 h-3.5" />
                  Create Project
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
