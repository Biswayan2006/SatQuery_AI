"""
Structured agentic-planning tests for :class:`AgenticController`.

These exercise the Task-3 upgrade (explicit planning + structured intent +
user-safe execution tracing) without relying on real model weights — the
model registry returns mock fallbacks, so what we assert here is the
*planning / tracing / integration wiring*, never model quality:

  * intent extraction for the three product-spec examples,
  * ``plan.to_dict()`` shape for every TaskType,
  * deterministic-tool step injection (only when the raster supports the tool),
  * execution-trace metadata shape — and the guarantee that it carries NO
    hidden chain-of-thought (only tool/model name, status, duration, summary),
  * keyword fallback when no semantic encoder is available,
  * a failed optional tool never crashes the pipeline,
  * every mandatory SIH workflow runs end-to-end through ``analyze()``.
"""
import asyncio
import os
import sys

import numpy as np
import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from agent.controller import AgenticController, ExecutionPlan  # noqa: E402
from agent.query_intent import IntentExtractor, QueryIntent  # noqa: E402
from agent.task_classifier import TaskType  # noqa: E402
from models.registry import ModelRegistry  # noqa: E402


# ── Fixtures / helpers ─────────────────────────────────────────────────────────

UTM_CRS = "EPSG:32633"
UTM_TRANSFORM = [10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0]


def _optical_rgbn(nir=0.8, red=0.2, h=32, w=32, geo=False):
    """A 4-band optical image_data dict (RED, GREEN, BLUE, NIR)."""
    arr = np.zeros((h, w, 4), dtype=np.float64)
    arr[..., 0] = red
    arr[..., 1] = 0.3
    arr[..., 2] = 0.1
    arr[..., 3] = nir
    meta = {"sensor": "rgbn"}
    if geo:
        meta["crs"] = UTM_CRS
        meta["transform"] = UTM_TRANSFORM
    return {"numpy_array": arr, "modality": "optical", "shape": [h, w, 4],
            "bands": 4, "metadata": meta}


def _s2(nir=0.4, swir1=0.3, red=0.2, h=32, w=32, geo=True):
    """A 13-band Sentinel-2 image_data dict (red=3, nir=7, swir1=11)."""
    arr = np.zeros((h, w, 13), dtype=np.float64)
    arr[..., 3] = red
    arr[..., 7] = nir
    arr[..., 11] = swir1
    meta = {"sensor": "sentinel-2"}
    if geo:
        meta["crs"] = UTM_CRS
        meta["transform"] = UTM_TRANSFORM
    return {"numpy_array": arr, "modality": "multispectral", "shape": [h, w, 13],
            "bands": 13, "metadata": meta}


def _rgb(h=32, w=32):
    """A plain 3-band RGB image_data dict (no NIR → no spectral indices)."""
    arr = np.zeros((h, w, 3), dtype=np.float64)
    return {"numpy_array": arr, "modality": "optical", "shape": [h, w, 3],
            "bands": 3, "metadata": {"sensor": "rgb"}}


def _sar(h=32, w=32):
    """A 2-band Sentinel-1 SAR image_data dict (vv, vh)."""
    arr = np.zeros((h, w, 2), dtype=np.float64)
    arr[..., 0] = 100.0
    arr[..., 1] = 25.0
    return {"numpy_array": arr, "modality": "sar", "shape": [h, w, 2],
            "bands": 2, "metadata": {"sensor": "sentinel-1"}}


@pytest.fixture()
def controller():
    return AgenticController(ModelRegistry())


def _run(coro):
    return asyncio.run(coro)


# A trace record must never leak reasoning — only these keys are permitted.
_ALLOWED_TRACE_KEYS = {
    "step", "status", "kind", "duration_ms",
    "output_summary", "parameters", "reason",
}
# Keys that would indicate a chain-of-thought / prompt leak.
_FORBIDDEN_TRACE_KEYS = {
    "prompt", "reasoning", "thought", "thoughts", "chain_of_thought",
    "cot", "rationale", "logits", "token_logprobs", "hidden",
}


# ── 1. Structured query intent (the three product-spec examples) ────────────────

def test_intent_builtup_increase_requires_comparison():
    intent = IntentExtractor().extract(
        "Has the built-up area increased?", num_images=2,
        modalities=["optical", "optical"],
    ).to_dict()
    assert intent["concept"] == "built_up"
    assert intent["operation"] == "increase"
    assert intent["requires_comparison"] is True
    assert "requires_fusion" not in intent


def test_intent_water_localize_requires_grounding():
    intent = IntentExtractor().extract(
        "Where are the water bodies?", num_images=1, modalities=["optical"],
    ).to_dict()
    assert intent["concept"] == "water"
    assert intent["operation"] == "localize"
    assert intent["requires_grounding"] is True


def test_intent_fusion_from_dual_modality():
    intent = IntentExtractor().extract(
        "Use the optical and SAR images together to identify built-up regions.",
        num_images=2, modalities=["optical", "sar"],
    ).to_dict()
    assert intent["concept"] == "built_up"
    assert intent["requires_fusion"] is True


def test_intent_scene_default_when_no_concept():
    """A query with no concept keyword falls back to the generic scene concept."""
    intent = IntentExtractor().extract("Tell me about this.", num_images=1)
    assert intent.concept == "scene"


# ── 2. Keyword fallback when no semantic encoder is available ──────────────────

def test_intent_keyword_fallback_without_encoder():
    """With encoder=None the extractor must still classify via keywords only."""
    ie = IntentExtractor(encoder=None)
    intent = ie.extract("Has vegetation decreased over time?", num_images=2)
    assert intent.used_semantic is False
    assert intent.concept == "vegetation"
    assert intent.operation == "decrease"
    assert intent.requires_comparison is True


# ── 3. plan.to_dict() shape for every TaskType ─────────────────────────────────

@pytest.mark.parametrize("task_type,images,query", [
    (TaskType.SINGLE_VQA, [_optical_rgbn()], "What land cover is visible?"),
    (TaskType.CAPTIONING, [_optical_rgbn()], "Describe this scene."),
    (TaskType.GROUNDING, [_optical_rgbn(geo=True)], "Where are the water bodies?"),
    (TaskType.CHANGE_VQA, [_s2(), _s2()], "Has the built-up area increased?"),
    (TaskType.CHANGE_DESCRIPTION, [_s2(), _s2()], "Describe the changes."),
    (TaskType.SAR_OPTICAL_FUSION, [_optical_rgbn(), _sar()],
     "Combine optical and SAR to find built-up regions."),
])
def test_plan_to_dict_shape(controller, task_type, images, query):
    intent = controller.intent_extractor.extract(query, len(images),
                                                  [i["modality"] for i in images])
    plan = controller._plan(task_type, 0.9, images, query, intent)
    d = plan.to_dict()

    assert d["task"] == task_type.value
    assert isinstance(d["intent"], str) and d["intent"]        # human phrase
    assert isinstance(d["intent_detail"], dict)                # structured flags
    assert isinstance(d["models"], list) and d["models"]
    assert isinstance(d["steps"], list) and d["steps"]
    assert isinstance(d["parameters"], dict)
    # The plan always begins with a validation step and ends with the answer.
    assert d["steps"][0].startswith("validate")
    assert d["steps"][-1] in ("format_answer",)


def test_plan_is_produced_before_execution(controller):
    """The plan object exists and is fully populated prior to any model call."""
    images = [_s2(), _s2()]
    intent = controller.intent_extractor.extract(
        "Where has construction expanded?", 2, ["multispectral", "multispectral"])
    plan = controller._plan(TaskType.CHANGE_VQA, 0.9, images,
                            "Where has construction expanded?", intent)
    assert isinstance(plan, ExecutionPlan)
    assert plan.intent is intent
    assert "detect_changes" in plan.steps


# ── 4. Deterministic-tool step injection (honest: only when supported) ─────────

def test_plan_injects_ndbi_for_builtup_change_on_s2(controller):
    """'Has built-up increased?' on Sentinel-2 injects NDBI + compare + change tools."""
    images = [_s2(), _s2()]
    intent = controller.intent_extractor.extract(
        "Has the built-up area increased?", 2, ["multispectral", "multispectral"])
    plan = controller._plan(TaskType.CHANGE_VQA, 0.9, images,
                            "Has the built-up area increased?", intent)
    steps = plan.steps
    assert "compute_ndbi" in steps
    assert "compare_ndbi" in steps
    assert "compute_change_area" in steps
    # Injected evidence steps must precede the final answer step.
    assert steps.index("compute_ndbi") < steps.index("run_change_vqa")


def test_plan_skips_spectral_index_when_bands_absent(controller):
    """A vegetation query on a plain RGB image must NOT inject an NDVI step."""
    images = [_rgb()]
    intent = controller.intent_extractor.extract(
        "How much vegetation is there?", 1, ["optical"])
    plan = controller._plan(TaskType.SINGLE_VQA, 0.9, images,
                            "How much vegetation is there?", intent)
    assert "compute_ndvi" not in plan.steps


def test_plan_injects_geographic_conversion_when_geo_available(controller):
    """A localize query on a geo-referenced image injects convert_to_geographic."""
    images = [_optical_rgbn(geo=True)]
    intent = controller.intent_extractor.extract(
        "Where are the water bodies?", 1, ["optical"])
    plan = controller._plan(TaskType.GROUNDING, 0.9, images,
                            "Where are the water bodies?", intent)
    assert "convert_to_geographic" in plan.steps


def test_plan_injects_sar_statistics_for_fusion(controller):
    images = [_optical_rgbn(), _sar()]
    intent = controller.intent_extractor.extract(
        "Combine optical and SAR to find built-up regions.", 2, ["optical", "sar"])
    plan = controller._plan(TaskType.SAR_OPTICAL_FUSION, 0.9, images,
                            "Combine optical and SAR to find built-up regions.", intent)
    assert "compute_sar_statistics" in plan.steps


# ── 5. Execution-trace metadata shape (NO chain-of-thought) ────────────────────

def _assert_trace_is_user_safe(step_trace):
    assert isinstance(step_trace, list) and step_trace
    for rec in step_trace:
        assert set(rec.keys()).issubset(_ALLOWED_TRACE_KEYS), rec
        assert not (set(rec.keys()) & _FORBIDDEN_TRACE_KEYS), rec
        assert isinstance(rec["step"], str) and rec["step"]
        assert rec["status"] in ("success", "unsupported", "skipped", "error")
        assert rec["kind"] in ("model", "tool", "stage")
        assert isinstance(rec["duration_ms"], (int, float))


def test_execution_trace_is_present_and_safe(controller):
    resp = _run(controller.analyze([_s2(), _s2()], "Has vegetation decreased?"))
    trace = resp.execution_summary.step_trace
    _assert_trace_is_user_safe(trace)
    # The model/handler stage is always recorded first.
    assert trace[0]["kind"] == "stage"


# ── 6. Failed optional tool must never crash the pipeline ──────────────────────

def test_failed_tool_layer_does_not_crash(controller, monkeypatch):
    """If the deterministic tool layer raises, analyze() still returns a response."""
    def _boom(*a, **k):
        raise RuntimeError("synthetic tool failure")

    monkeypatch.setattr(controller.tool_planner, "run_for_task", _boom)
    resp = _run(controller.analyze([_optical_rgbn()], "What is in this image?"))
    assert resp.answer                       # a response is still produced
    assert resp.tool_evidence is None        # evidence degraded to None
    # The trace still records the model stage and the tool-failure marker.
    trace = resp.execution_summary.step_trace
    assert any(r["step"] == "deterministic_tools" and r["status"] == "error"
               for r in trace)


def test_missing_tool_planner_is_tolerated(controller):
    """With no tool planner at all, analyze() still completes."""
    controller.tool_planner = None
    resp = _run(controller.analyze([_optical_rgbn()], "Describe this scene."))
    assert resp.answer
    assert resp.tool_evidence is None


# ── 7. Every mandatory SIH workflow runs end-to-end ────────────────────────────

@pytest.mark.parametrize("task_hint,images,query", [
    ("SINGLE_VQA", [_optical_rgbn()], "What land cover types are visible?"),
    ("CAPTIONING", [_optical_rgbn()], "Describe this satellite scene."),
    ("GROUNDING", [_optical_rgbn(geo=True)], "Where are the water bodies?"),
    ("CHANGE_VQA", [_s2(nir=0.8, swir1=0.2), _s2(nir=0.3, swir1=0.6)],
     "Has the built-up area increased?"),
    ("CHANGE_DESCRIPTION", [_s2(), _s2()], "Describe the changes between the images."),
    ("SAR_OPTICAL_FUSION", [_optical_rgbn(), _sar()],
     "Use optical and SAR together to identify built-up regions."),
])
def test_sih_workflow_end_to_end(controller, task_hint, images, query):
    resp = _run(controller.analyze(images, query, task_hint=task_hint))

    # Core response contract.
    assert resp.session_id
    assert resp.task == task_hint
    assert isinstance(resp.answer, str) and resp.answer
    assert 0.0 <= resp.confidence <= 1.0

    es = resp.execution_summary
    assert es.selected_task == task_hint
    assert es.models_used
    assert es.steps

    # Structured planning surfaced to the client.
    assert es.plan is not None and es.plan["task"] == task_hint
    assert es.intent is not None                       # structured understanding
    _assert_trace_is_user_safe(es.step_trace)          # user-safe tracing

    # These are mock-model runs, so results should be flagged as degraded and
    # requiring verification — never presented as trustworthy.
    assert resp.is_degraded is True


def test_change_vqa_produces_spectral_tool_evidence(controller):
    """The vegetation-change workflow yields deterministic NDVI + change evidence."""
    before = _s2(nir=0.8, swir1=0.2)   # high NIR → high NDVI
    after = _s2(nir=0.3, swir1=0.6)    # low NIR → low NDVI
    resp = _run(controller.analyze([before, after], "Has vegetation decreased?",
                                   task_hint="CHANGE_VQA"))
    assert resp.tool_evidence is not None
    tools = [e.get("tool") for e in resp.tool_evidence]
    # NDVI computed on both images (vegetation concept resolves nir/red on S2).
    assert tools.count("NDVI") == 2
    ndvis = [e for e in resp.tool_evidence if e.get("tool") == "NDVI"]
    assert ndvis[1]["mean"] < ndvis[0]["mean"]         # NDVI decreased
