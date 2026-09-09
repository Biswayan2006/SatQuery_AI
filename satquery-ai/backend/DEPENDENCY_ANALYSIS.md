# Python Dependency Analysis

## Current Structure

### 1. requirements.txt (Production Runtime)
**Total:** 36 packages
**Purpose:** Backend API server with model inference

### 2. requirements-dev.txt (Development)
**Total:** 19 packages (includes requirements.txt)
**Purpose:** Testing, linting, documentation, debugging

## Dependency Analysis

### Production Dependencies Analysis

#### ✅ Definitely Required for Production
| Package | Version | Purpose | Notes |
|---------|---------|---------|-------|
| fastapi | 0.115.0 | Web framework | Core |
| uvicorn[standard] | 0.32.0 | ASGI server | Core |
| python-multipart | 0.0.12 | File uploads | Core |
| aiofiles | 24.1.0 | Async file I/O | Core |
| torch | 2.6.0 | Deep learning | Core (matches Docker) |
| torchvision | 0.21.0 | Computer vision | Core |
| transformers | 4.47.0 | HuggingFace models | Core |
| sentence-transformers | 3.3.1 | Embedding models | Core |
| accelerate | 1.2.1 | Model optimization | Core |
| open-clip-torch | 2.29.0 | CLIP models | Core |
| Pillow | 11.0.0 | Image processing | Core |
| opencv-python-headless | 4.10.0.84 | Computer vision | Core |
| numpy | 2.1.3 | Numerical computing | Core |
| scikit-learn | 1.6.0 | Machine learning utilities | Core |
| matplotlib | 3.9.3 | Visualization | Required for reports |
| rasterio | 1.4.3 | Geospatial processing | Core |
| geopandas | 1.0.1 | Geospatial data | Core |
| shapely | 2.0.6 | Geometric operations | Core |
| pyproj | 3.7.0 | Coordinate transformations | Core |
| pydantic | 2.10.3 | Data validation | Core |
| pydantic-settings | 2.7.0 | Configuration management | Core |
| python-dotenv | 1.0.1 | Environment variables | Core |
| reportlab | 4.2.5 | PDF generation | Required for reports |
| fpdf2 | 2.8.1 | PDF generation | Required for reports |
| tqdm | 4.67.1 | Progress bars | Used in model loading |
| scipy | 1.14.1 | Scientific computing | Used by sklearn |
| schedule | 1.2.2 | Task scheduling | Used by cleanup service |
| python-magic | 0.4.27 | File type detection | Security validation |

#### ⚠️ Questionable for Production
| Package | Version | Purpose | Issue | Recommendation |
|---------|---------|---------|-------|----------------|
| datasets | 3.2.0 | HuggingFace datasets | Only used for training | **REMOVE from production** |
| huggingface-hub | 0.27.0 | Model downloading | Could be dev-only | Keep (needed for model downloads) |
| einops | 0.8.0 | Tensor operations | May not be used | Investigate usage |
| timm | 1.0.12 | Vision models | May not be used | Investigate usage |

### Development Dependencies Analysis

#### ✅ Correctly in requirements-dev.txt
| Package | Version | Purpose | Status |
|---------|---------|---------|--------|
| pytest | 8.3.4 | Testing | Correct |
| pytest-asyncio | 0.24.0 | Async testing | Correct |
| pytest-cov | 5.0.0 | Test coverage | Correct |
| pytest-mock | 3.14.0 | Mocking | Correct |
| httpx | 0.28.1 | HTTP client | Correct |
| pytest-httpx | 0.31.0 | HTTP testing | Correct |
| psutil | 6.2.0 | System monitoring | Correct |
| black | 24.10.0 | Code formatting | Correct |
| flake8 | 7.1.1 | Linting | Correct |
| mypy | 1.13.0 | Type checking | Correct |
| pre-commit | 4.0.1 | Git hooks | Correct |
| ipython | 8.29.0 | Interactive shell | Correct |
| jupyter | 1.1.1 | Notebooks | Correct |
| memory-profiler | 0.61.0 | Memory profiling | Correct |
| sphinx | 7.3.7 | Documentation | Correct |
| sphinx-rtd-theme | 2.0.0 | Documentation theme | Correct |

### Missing Dependencies

#### Training/Evaluation Dependencies (Not Separated)
**Current Issue:** Training code mixed with production code
**Packages needed for training only:**
- `datasets` (already identified)
- Training-specific optimizers
- Evaluation metrics
- Notebook dependencies

## Import Analysis

Let me check actual imports in the codebase to verify dependencies:

<分析待执行>

## Proposed Dependency Structure

### 1. requirements.txt (Production Runtime Only)
```
# ── Web Framework ──────────────────────────────
fastapi==0.115.0
uvicorn[standard]==0.32.0
python-multipart==0.0.12
aiofiles==24.1.0

# ── ML Core ────────────────────────────────────
torch==2.6.0
torchvision==0.21.0
transformers==4.47.0
sentence-transformers==3.3.1
accelerate==1.2.1
huggingface-hub==0.27.0
open-clip-torch==2.29.0

# ── Computer Vision ────────────────────────────
Pillow==11.0.0
opencv-python-headless==4.10.0.84
numpy==2.1.3
scikit-learn==1.6.0
matplotlib==3.9.3

# ── Geospatial ─────────────────────────────────
rasterio==1.4.3
geopandas==1.0.1
shapely==2.0.6
pyproj==3.7.0

# ── Data / Config ──────────────────────────────
pydantic==2.10.3
pydantic-settings==2.7.0
python-dotenv==1.0.1

# ── PDF Generation ─────────────────────────────
reportlab==4.2.5
fpdf2==2.8.1

# ── Utilities ──────────────────────────────────
tqdm==4.67.1
scipy==1.14.1
schedule==1.2.2
python-magic==0.4.27
```

### 2. requirements-dev.txt (Development)
```
# Production dependencies
-r requirements.txt

# Testing
pytest==8.3.4
pytest-asyncio==0.24.0
pytest-cov==5.0.0
pytest-mock==3.14.0
httpx==0.28.1
pytest-httpx==0.31.0
psutil==6.2.0

# Development tools
black==24.10.0
flake8==7.1.1
mypy==1.13.0
pre-commit==4.0.1

# Debugging and profiling
ipython==8.29.0
jupyter==1.1.1
memory-profiler==0.61.0

# Documentation
sphinx==7.3.7
sphinx-rtd-theme==2.0.0
```

### 3. requirements-training.txt (Training - NEW)
```
# Base dependencies
torch==2.6.0
torchvision==0.21.0
transformers==4.47.0
accelerate==1.2.1
huggingface-hub==0.27.0

# Training-specific
datasets==3.2.0
einops==0.8.0
timm==1.0.12

# Visualization and monitoring
tensorboard==2.17.0
wandb==0.17.0

# Optimization
torch-optimizer==0.4.0
```

### 4. requirements-evaluation.txt (Evaluation - NEW)
```
# Base dependencies
torch==2.6.0
torchvision==0.21.0
transformers==4.47.0

# Evaluation metrics
scikit-learn==1.6.0
scipy==1.14.1
numpy==2.1.3

# Visualization
matplotlib==3.9.3
seaborn==0.13.2

# Data handling
pandas==2.2.2
```

## Action Plan

1. **Remove unnecessary packages from production:**
   - `datasets` (moved to training)
   - `einops` (investigate if used)
   - `timm` (investigate if used)

2. **Create separate requirements files:**
   - `requirements-training.txt`
   - `requirements-evaluation.txt`

3. **Verify imports:**
   - Check if `einops` and `timm` are actually used
   - Ensure no broken imports after separation

4. **Update documentation:**
   - Document which requirements file to use for each purpose
   - Update Dockerfile to use production requirements only

5. **Test:**
   - Verify production install works without training packages
   - Verify training environment can be set up separately