# SatQuery AI - Proposed Production Repository Structure

## Current Issues Identified

1. **Mixed Concerns**: Production runtime, training scripts, evaluation code, and notebooks all in same directory
2. **Runtime Data in Git**: User uploads, generated reports, model cache currently tracked by Git
3. **Development Artifacts**: Python caches, test caches, virtual environments in repository
4. **Secret Files**: `.env` files with actual secrets
5. **Frontend/Backend Mix**: Both in same repository but not cleanly separated

## Proposed Clean Structure

```
satquery-ai/
│
├── backend/                    # PRODUCTION BACKEND APPLICATION
│   ├── app/                   # Core application package
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI entry point
│   │   ├── config.py         # Configuration (pydantic-settings)
│   │   │
│   │   ├── api/              # API layer
│   │   │   ├── __init__.py
│   │   │   ├── routes.py     # All API routes
│   │   │   ├── schemas.py    # Pydantic models
│   │   │   └── dependencies.py
│   │   │
│   │   ├── agent/            # Agentic controller
│   │   │   ├── __init__.py
│   │   │   ├── controller.py
│   │   │   ├── task_classifier.py
│   │   │   ├── tool_planner.py
│   │   │   ├── input_validator.py
│   │   │   ├── query_intent.py
│   │   │   ├── report_generator.py
│   │   │   └── execution_trace.py
│   │   │
│   │   ├── models/           # Model implementations
│   │   │   ├── __init__.py
│   │   │   ├── registry.py   # Model registry singleton
│   │   │   ├── vqa_model.py
│   │   │   ├── captioning_model.py
│   │   │   ├── grounding_model.py
│   │   │   ├── change_model.py
│   │   │   ├── sar_fusion_model.py
│   │   │   └── _mock.py      # Mock models (dev/testing only)
│   │   │
│   │   ├── services/         # Background services
│   │   │   ├── __init__.py
│   │   │   ├── cleanup.py    # TTL-based cleanup
│   │   │   ├── inference.py  # Inference queue/semaphore
│   │   │   ├── degradation_monitor.py
│   │   │   └── confidence_service.py
│   │   │
│   │   ├── middleware/       # FastAPI middleware
│   │   │   ├── __init__.py
│   │   │   ├── security.py
│   │   │   ├── logging_middleware.py
│   │   │   ├── exception_handler.py
│   │   │   └── rate_limiter.py
│   │   │
│   │   ├── tools/           # Specialized analysis tools
│   │   │   ├── __init__.py
│   │   │   ├── spectral.py
│   │   │   ├── geospatial.py
│   │   │   ├── change_analysis.py
│   │   │   └── sar_analysis.py
│   │   │
│   │   └── utils/           # Utilities
│   │       ├── __init__.py
│   │       ├── image_utils.py
│   │       ├── visualization.py
│   │       ├── geo_utils.py
│   │       └── logging.py
│   │
│   ├── tests/               # Test suite (NOT in app package)
│   │   ├── __init__.py
│   │   ├── test_api.py
│   │   ├── test_upload.py
│   │   ├── test_models.py
│   │   ├── test_cleanup.py
│   │   ├── test_security.py
│   │   └── test_integration.py
│   │
│   ├── data/               # RUNTIME DATA (NOT in Git, Docker volume)
│   │   ├── uploads/       # User uploaded images
│   │   ├── reports/       # Generated PDF reports
│   │   └── model_cache/   # HuggingFace model cache
│   │
│   ├── Dockerfile         # Production Dockerfile
│   ├── requirements.txt   # Production dependencies only
│   ├── .env.example      # Example environment (NO SECRETS)
│   ├── .dockerignore     # Docker ignore patterns
│   └── pytest.ini        # Test configuration
│
├── frontend/              # PRODUCTION FRONTEND APPLICATION
│   ├── src/
│   ├── public/
│   ├── package.json
│   ├── Dockerfile
│   ├── next.config.js
│   └── .env.example
│
├── training/              # MODEL TRAINING (Separate from production)
│   ├── datasets/         # Dataset preparation scripts
│   ├── scripts/          # Training scripts
│   │   ├── train_clip.py
│   │   ├── train_vqa.py
│   │   └── train_fusion.py
│   ├── configs/          # Training configurations
│   └── README.md
│
├── evaluation/           # BENCHMARK EVALUATION (Separate from production)
│   ├── vqa/             # VQA evaluation
│   ├── grounding/       # Grounding evaluation
│   ├── captioning/      # Captioning evaluation
│   ├── change/          # Change detection evaluation
│   ├── fusion/          # SAR-optical fusion evaluation
│   ├── datasets/        # Evaluation dataset scripts
│   ├── configs/         # Evaluation configurations
│   ├── results/         # Evaluation results (NOT in Git)
│   └── README.md
│
├── research/             # RESEARCH & EXPERIMENTS (Separate from production)
│   ├── notebooks/       # Jupyter notebooks
│   │   ├── 01_bigearthnet_exploration.ipynb
│   │   ├── 02_clip_finetuning.ipynb
│   │   └── 03_benchmark_evaluation.ipynb
│   ├── experiments/     # Experimental code
│   └── README.md
│
├── deployment/          # DEPLOYMENT CONFIGURATION
│   ├── docker/         # Docker deployment files
│   │   ├── docker-compose.yml
│   │   ├── docker-compose.dev.yml
│   │   └── nginx/
│   ├── digitalocean/   # DigitalOcean specific
│   │   ├── droplet-setup.sh
│   │   ├── app-platform.yaml
│   │   └── README.md
│   ├── kubernetes/     # Kubernetes manifests (optional)
│   ├── scripts/        # Deployment scripts
│   │   ├── smoke_test.py
│   │   ├── health_check.py
│   │   └── backup.sh
│   └── README.md
│
├── docs/               # DOCUMENTATION
│   ├── ARCHITECTURE.md
│   ├── API.md
│   ├── MODELS.md
│   ├── DEPLOYMENT.md
│   ├── SECURITY.md
│   └── TROUBLESHOOTING.md
│
├── .gitignore          # Comprehensive .gitignore
├── .dockerignore       # Root .dockerignore
├── docker-compose.yml  # Root Docker Compose for full stack
├── LICENSE
└── README.md
```

## Migration Plan

### Phase 1: Backend Restructuring
1. Move existing `backend/` files to `backend/app/` package
2. Create proper Python package structure with `__init__.py` files
3. Move tests to `backend/tests/`
4. Move data directories to `backend/data/` (will be .gitignored)

### Phase 2: Separate Concerns
1. Move training code from `backend/training/` to `training/`
2. Move evaluation code from `backend/evaluation/` to `evaluation/`
3. Move notebooks from root `notebooks/` to `research/notebooks/`

### Phase 3: Deployment Configuration
1. Create `deployment/` directory with DigitalOcean configs
2. Update Docker Compose for new structure
3. Create deployment scripts and documentation

### Phase 4: Cleanup
1. Remove all runtime data from Git
2. Delete development artifacts
3. Update all imports to new structure
4. Verify all functionality works

## Key Changes from Current Structure

1. **Clear Separation**: Production runtime vs training vs evaluation vs research
2. **Python Package**: Proper `app/` package with `__init__.py` files
3. **Data Isolation**: Runtime data in `backend/data/` (Docker volume, .gitignored)
4. **Deployment Ready**: All deployment configs in `deployment/` directory
5. **DigitalOcean Focus**: Specific configuration for DigitalOcean deployment

## Import Updates Required

Current imports like:
```python
from config import get_settings
from api.routes import router
from models.registry import ModelRegistry
```

Will become:
```python
from app.config import get_settings
from app.api.routes import router
from app.models.registry import ModelRegistry
```

## Docker Changes

Current structure mounts volumes at container paths:
- `/app/uploads` → `./backend/data/uploads`
- `/app/reports` → `./backend/data/reports`
- `/app/model_cache` → `./backend/data/model_cache`

## Benefits

1. **Production Clean**: Only production code in `backend/app/`
2. **Training Separate**: Training scripts don't pollute production
3. **Evaluation Separate**: Benchmark code separate from runtime
4. **Research Tracked**: Notebooks and experiments preserved but separate
5. **Deployment Ready**: All configs in one place
6. **DigitalOcean Optimized**: Specific deployment guidance
7. **Git Clean**: No runtime data, no secrets, no development artifacts

## Next Steps

1. Create detailed migration script
2. Test import updates
3. Verify Docker volume mounts
4. Test complete deployment flow