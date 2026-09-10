"use client";

import { useState, useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { Folder, Plus, Check, X, Trash2 } from "lucide-react";
import { useProject, Project } from "@/context/ProjectContext";

export default function ProjectTabs({ compact = false }: { compact?: boolean }) {
  const { projects, activeProject, setActiveProjectId, createProject, deleteProject } = useProject();
  const [showModal, setShowModal] = useState(false);
  const [projectToDelete, setProjectToDelete] = useState<Project | null>(null);
  const [mounted, setMounted] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [newModality, setNewModality] = useState("Optical");
  const [newSensor, setNewSensor] = useState("Sentinel-2");

  const triggerButtonRef = useRef<HTMLButtonElement | null>(null);
  const cancelButtonRef = useRef<HTMLButtonElement | null>(null);
  const dialogRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  const closeDeleteModal = () => {
    const trigger = triggerButtonRef.current;
    setProjectToDelete(null);
    requestAnimationFrame(() => {
      trigger?.focus();
      triggerButtonRef.current = null;
    });
  };

  // Accessibility: focus management & Escape handler for delete modal
  useEffect(() => {
    if (!projectToDelete) return;

    const focusTimer = setTimeout(() => {
      cancelButtonRef.current?.focus();
    }, 50);

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        closeDeleteModal();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      clearTimeout(focusTimer);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [projectToDelete]);

  const handleDialogKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Tab") {
      const focusable = dialogRef.current?.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [tabindex]:not([tabindex="-1"])'
      );
      if (!focusable || focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (e.shiftKey) {
        if (document.activeElement === first) {
          e.preventDefault();
          last.focus();
        }
      } else {
        if (document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    }
  };

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
            <div
              key={project.id}
              role="button"
              tabIndex={0}
              onClick={() => setActiveProjectId(project.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setActiveProjectId(project.id);
                }
              }}
              className={`group flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all no-tap whitespace-nowrap cursor-pointer select-none ${
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
              {isActive && projects.length === 1 && (
                <span
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ background: "var(--accent-contrast)" }}
                />
              )}
              {projects.length > 1 && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    triggerButtonRef.current = e.currentTarget;
                    setProjectToDelete(project);
                  }}
                  onKeyDown={(e) => {
                    e.stopPropagation();
                  }}
                  className={`p-0.5 rounded transition-opacity ${
                    isActive
                      ? "hover:bg-white/20 text-white/80 hover:text-white"
                      : "opacity-40 group-hover:opacity-100 hover:bg-raised text-ink-muted hover:text-danger"
                  }`}
                  title={`Delete ${project.name}`}
                  aria-label={`Delete ${project.name}`}
                >
                  <X className="w-3 h-3" />
                </button>
              )}
            </div>
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

      {/* Delete Confirmation Accessible Modal Dialog */}
      {mounted &&
        projectToDelete &&
        createPortal(
          <div
            className="fixed inset-0 z-[100] flex items-center justify-center p-4 backdrop-blur-sm animate-fade-in"
            style={{ background: "var(--overlay)" }}
            onClick={closeDeleteModal}
          >
            <div
              ref={dialogRef}
              role="dialog"
              aria-modal="true"
              aria-labelledby="delete-project-title"
              aria-describedby="delete-project-description"
              onKeyDown={handleDialogKeyDown}
              className="w-full max-w-sm rounded-xl p-5 shadow-2xl space-y-4"
              style={{
                background: "var(--bg-elevated)",
                border: "1px solid var(--border)",
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center gap-3">
                <div
                  className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
                  style={{ background: "color-mix(in srgb, var(--danger) 15%, transparent)" }}
                >
                  <Trash2 className="w-5 h-5 text-danger" />
                </div>
                <div>
                  <h3
                    id="delete-project-title"
                    className="text-sm font-semibold"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Delete Project
                  </h3>
                  <p
                    id="delete-project-description"
                    className="text-xs mt-0.5"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Delete project <span className="font-semibold" style={{ color: "var(--text-primary)" }}>{projectToDelete.name}</span>? This action cannot be undone.
                  </p>
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-1">
                <button
                  ref={cancelButtonRef}
                  type="button"
                  onClick={closeDeleteModal}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold text-ink-muted hover:text-ink hover:bg-raised transition-colors no-tap"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={() => {
                    deleteProject(projectToDelete.id);
                    closeDeleteModal();
                  }}
                  className="px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-danger text-white hover:opacity-90 transition-opacity no-tap"
                >
                  Delete Project
                </button>
              </div>
            </div>
          </div>,
          document.body
        )}

      {/* New Project Modal rendered via portal to escape parent stacking contexts */}
      {mounted &&
        showModal &&
        createPortal(
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
          </div>,
          document.body
        )}
    </>
  );
}
