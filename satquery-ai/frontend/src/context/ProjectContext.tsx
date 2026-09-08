"use client";

import React, { createContext, useContext, useState, useEffect } from "react";

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
}

const ProjectContext = createContext<ProjectContextType | undefined>(undefined);

const STORAGE_KEY = "sq-active-project-id";

export function ProjectProvider({ children }: { children: React.ReactNode }) {
  const [projects, setProjects] = useState<Project[]>(DEFAULT_PROJECTS);
  const [activeProjectId, setActiveProjectIdState] = useState<string>("urban-watch");

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
    const id = name.toLowerCase().replace(/[^a-z0-9]+/g, "-") || `project-${Date.now()}`;
    const newProj: Project = {
      id,
      name,
      modality: modality || "Optical",
      sensor: sensor || "Sentinel-2",
      description: description || "Custom satellite remote sensing project",
      scenes: 0,
      lastActive: "Just now",
    };
    setProjects((prev) => [newProj, ...prev]);
    setActiveProjectId(id);
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
