"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import type { UploadedImage, AnalysisResponse } from "@/types";

export interface ProjectWorkspaceState {
  images: UploadedImage[];
  query: string;
  result: AnalysisResponse | null;
  imageRevision: number;
}

export interface Project {
  id: string;
  name: string;
  description: string;
  modality: string;
  sensor: string;
  scenes: number;
  lastActive: string;
}

export const DEFAULT_PROJECTS: Project[] = [
  {
    id: "urban-watch",
    name: "Urban Watch",
    description: "Built-up area and infrastructure change analysis",
    modality: "Optical",
    sensor: "Sentinel-2",
    scenes: 1240,
    lastActive: "14m ago",
  },
  {
    id: "forest-monitor",
    name: "Forest Monitor",
    description: "Canopy cover and deforestation tracking",
    modality: "Multispectral",
    sensor: "Landsat-8",
    scenes: 890,
    lastActive: "2h ago",
  },
  {
    id: "coastal-watch",
    name: "Coastal Watch",
    description: "Flood extent and shoreline water analysis",
    modality: "SAR + Optical",
    sensor: "Sentinel-1/2",
    scenes: 340,
    lastActive: "1d ago",
  },
  {
    id: "agriculture-survey",
    name: "Agriculture Survey",
    description: "Crop health and seasonal vegetation analysis",
    modality: "Multispectral",
    sensor: "Cartosat",
    scenes: 2150,
    lastActive: "3d ago",
  },
];

interface ProjectContextType {
  projects: Project[];
  activeProject: Project;
  setActiveProjectId: (id: string) => void;
  createProject: (name: string, modality: string, sensor: string, description?: string) => void;
  deleteProject: (id: string) => void;
  workspaces: Record<string, ProjectWorkspaceState>;
  setWorkspaces: React.Dispatch<React.SetStateAction<Record<string, ProjectWorkspaceState>>>;
}

const ProjectContext = createContext<ProjectContextType | undefined>(undefined);

const STORAGE_KEY = "sq-active-project-id";
const PROJECTS_STORAGE_KEY = "sq-projects-list";

export function ProjectProvider({ children }: { children: React.ReactNode }) {
  const [projects, setProjects] = useState<Project[]>(DEFAULT_PROJECTS);
  const [workspaces, setWorkspaces] = useState<Record<string, ProjectWorkspaceState>>({});
  const [activeProjectId, setActiveProjectIdState] = useState<string>("urban-watch");

  // Load saved projects list on mount if available
  useEffect(() => {
    try {
      const savedProjects = localStorage.getItem(PROJECTS_STORAGE_KEY);
      if (savedProjects) {
        const parsed = JSON.parse(savedProjects);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setProjects(parsed);
        }
      }
    } catch {
      // localStorage unavailable or restricted
    }
  }, []);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved && projects.some((p) => p.id === saved)) {
        setActiveProjectIdState(saved);
      }
    } catch {
      // localStorage unavailable or restricted
    }
  }, [projects]);

  const setActiveProjectId = (id: string) => {
    setActiveProjectIdState(id);
    try {
      localStorage.setItem(STORAGE_KEY, id);
    } catch {
      // ignore
    }
  };

  const createProject = (name: string, modality: string, sensor: string, description?: string) => {
    const baseSlug = name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
    const id = baseSlug ? `${baseSlug}-${Date.now()}` : `project-${Date.now()}`;
    const newProj: Project = {
      id,
      name,
      modality: modality || "Optical",
      sensor: sensor || "Sentinel-2",
      description: description || "Custom satellite remote sensing project",
      scenes: 0,
      lastActive: "Just now",
    };
    const updated = [newProj, ...projects];
    setProjects(updated);
    try {
      localStorage.setItem(PROJECTS_STORAGE_KEY, JSON.stringify(updated));
    } catch {
      // ignore
    }
    setActiveProjectId(id);
  };

  const deleteProject = (id: string) => {
    if (projects.length <= 1) return;

    // Remove deleted project's workspace from shared workspace owner and revoke preview URLs
    setWorkspaces((prev) => {
      const targetWs = prev[id];
      if (targetWs) {
        targetWs.images.forEach((img) => {
          if (img.previewUrl) {
            try {
              URL.revokeObjectURL(img.previewUrl);
            } catch {
              // ignore
            }
          }
        });
      }
      const next = { ...prev };
      delete next[id];
      return next;
    });

    const remaining = projects.filter((p) => p.id !== id);
    setProjects(remaining);
    try {
      localStorage.setItem(PROJECTS_STORAGE_KEY, JSON.stringify(remaining));
    } catch {
      // ignore
    }
    if (activeProjectId === id && remaining.length > 0) {
      setActiveProjectId(remaining[0].id);
    }
  };

  const activeProject =
    projects.find((p) => p.id === activeProjectId) ?? projects[0] ?? DEFAULT_PROJECTS[0];

  return (
    <ProjectContext.Provider
      value={{
        projects,
        activeProject,
        setActiveProjectId,
        createProject,
        deleteProject,
        workspaces,
        setWorkspaces,
      }}
    >
      {children}
    </ProjectContext.Provider>
  );
}

export function useProject() {
  const context = useContext(ProjectContext);
  if (!context) {
    throw new Error("useProject must be used within a ProjectProvider");
  }
  return context;
}
