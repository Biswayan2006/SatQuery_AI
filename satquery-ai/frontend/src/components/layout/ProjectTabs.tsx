"use client";

import { useState } from "react";
import { Folder, Plus, Check, X } from "lucide-react";
import { useProject } from "@/context/ProjectContext";

export default function ProjectTabs({ compact = false }: { compact?: boolean }) {
  const { projects, activeProject, setActiveProjectId, createProject } = useProject();
  const [showModal, setShowModal] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [newModality, setNewModality] = useState("Optical");
  const [newSensor, setNewSensor] = useState("Sentinel-2");

  const handleCreate = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newProjectName.trim()) return;
    createProject(newProjectName.trim(), newModality, newSensor);
    setNewProjectName("");
    setShowModal(false);
  };

  return (
    <>
      <div className="flex items-center gap-1.5 overflow-x-auto scrollbar-none py-1">
        {projects.map((project) => {
          const isActive = project.id === activeProject.id;
          return (
            <button
              key={project.id}
              onClick={() => setActiveProjectId(project.id)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all no-tap whitespace-nowrap ${
                isActive
                  ? "bg-accent text-accent-contrast shadow-sm"
                  : "bg-surface hover:bg-raised text-ink-muted hover:text-ink border border-line"
              }`}
              title={`${project.name}: ${project.description}`}
            >
              <Folder
                className="w-3.5 h-3.5 flex-shrink-0"
                style={{
                  color: isActive ? "var(--accent-contrast)" : "var(--text-muted)",
                }}
              />
              <span>{project.name}</span>
              {isActive && (
                <span
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ background: "var(--accent-contrast)" }}
                />
              )}
            </button>
          );
        })}

        <button
          onClick={() => setShowModal(true)}
          className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-surface hover:bg-raised text-ink-muted hover:text-ink border border-dashed border-line-strong transition-colors no-tap whitespace-nowrap"
          title="Create new project"
        >
          <Plus className="w-3.5 h-3.5" />
          {!compact && <span>New project</span>}
        </button>
      </div>

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
                  placeholder="e.g. Himalaya Glacial Monitor"
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
    </>
  );
}
