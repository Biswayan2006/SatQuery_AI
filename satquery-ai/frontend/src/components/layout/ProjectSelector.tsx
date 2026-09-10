"use client";

import { useState, useRef, useEffect } from "react";
import { createPortal } from "react-dom";
import { Folder, ChevronDown, Check, Layers, Trash2 } from "lucide-react";
import { useProject, Project } from "@/context/ProjectContext";

export default function ProjectSelector() {
  const { projects, activeProject, setActiveProjectId, deleteProject } = useProject();
  const [open, setOpen] = useState(false);
  const [projectToDelete, setProjectToDelete] = useState<Project | null>(null);
  const [mounted, setMounted] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    const handleOutside = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    if (open) document.addEventListener("mousedown", handleOutside);
    return () => document.removeEventListener("mousedown", handleOutside);
  }, [open]);

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
                <div
                  key={proj.id}
                  className={`group w-full flex items-center justify-between gap-2 px-2.5 py-2 rounded-lg text-left text-xs transition-colors no-tap cursor-pointer ${
                    isSelected
                      ? "bg-accent text-accent-contrast font-semibold"
                      : "hover:bg-raised text-ink-soft hover:text-ink"
                  }`}
                  onClick={() => {
                    setActiveProjectId(proj.id);
                    setOpen(false);
                  }}
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
                  <div className="flex items-center gap-1.5 flex-shrink-0">
                    {isSelected && <Check className="w-3.5 h-3.5" />}
                    {projects.length > 1 && (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setProjectToDelete(proj);
                          setOpen(false);
                        }}
                        className={`p-1 rounded transition-opacity ${
                          isSelected
                            ? "hover:bg-white/20 text-white/80 hover:text-white"
                            : "opacity-0 group-hover:opacity-100 hover:bg-raised text-ink-faint hover:text-danger"
                        }`}
                        title={`Delete ${proj.name}`}
                        aria-label={`Delete ${proj.name}`}
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {mounted &&
        projectToDelete &&
        createPortal(
          <div
            className="fixed inset-0 z-[100] flex items-center justify-center p-4 backdrop-blur-sm animate-fade-in"
            style={{ background: "var(--overlay)" }}
            onClick={() => setProjectToDelete(null)}
          >
            <div
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
                  <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                    Delete Project
                  </h3>
                  <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
                    Delete project <span className="font-semibold" style={{ color: "var(--text-primary)" }}>{projectToDelete.name}</span>? This action cannot be undone.
                  </p>
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-1">
                <button
                  type="button"
                  onClick={() => setProjectToDelete(null)}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold text-ink-muted hover:text-ink hover:bg-raised transition-colors no-tap"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={() => {
                    deleteProject(projectToDelete.id);
                    setProjectToDelete(null);
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
    </div>
  );
}
