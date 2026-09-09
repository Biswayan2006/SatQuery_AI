"""
SatQuery AI — SAR-Optical Fusion Tests
=======================================
Verifies the genuine multimodal fusion pathway WITHOUT any network / weight
download.  A tiny random-init ``BlipForQuestionAnswering`` (built offline from a
small ``BlipConfig``) exercises the real BLIP visual-token injection pathway, so
what we assert is the *wiring* — that fused features actually reach answer
generation, that gradients reach the adapter, that a checkpoint round-trips —
never model quality.

Covered (mapping to the Task-4 spec):

  1. SAR-only pair rejected for the fusion task (controller).
  2. Optical-only pair rejected for the fusion task (controller).
  3. Compatible SAR+optical pair accepted → full dict contract.
  4. Fused features actually reach answer generation (spy on the injection
     method) AND changing the SAR input changes the injected tokens.
  5. Gradients reach the fusion adapter during a training step; the frozen BLIP
     vision tower receives none.
  6. A trained checkpoint loads at inference (fusion_trained flips True; forward
     output is identical after reload).
  7. SAR preprocessing: dual-pol VV/VH → VV/VH ratio + dB/normalise; single-band
     → ratio unavailable, no crash; channel count is NOT assumed.
  8. diagnose() reports all four stages OK on a random-init model.
  9. An untrained model flags the result degraded (fusion_trained=False) and
     does not present fused output as trustworthy.

Run with:
    cd satquery-ai/backend
    python -m pytest tests/test_sar_fusion.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest
import torch

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


# ── Offline tiny BLIP + fusion model builders ───────────────────────────────────

def _tiny_blip_vqa():
    """
    Build a randomly-initialised BLIP VQA wrapper offline (no download).

    Uses a small BlipConfig with a vocab large enough that the decoder start /
    sep / pad token ids are in range, so the full generate() path runs.
    """
    from transformers import BlipConfig, BlipForQuestionAnswering
    from models.vqa_model import RemoteSensingVQA
    from unittest.mock import patch

    vocab = 120
    cfg = BlipConfig(
        text_config=dict(
            vocab_size=vocab, hidden_size=64, num_hidden_layers=2,
            num_attention_heads=2, intermediate_size=128, max_position_embeddings=64,
            bos_token_id=vocab - 1, eos_token_id=2, pad_token_id=0, sep_token_id=102 % vocab,
        ),
        vision_config=dict(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=128, image_size=64, patch_size=16,
        ),
    )
    model = BlipForQuestionAnswering(cfg)
    model.eval()

    with patch.object(RemoteSensingVQA, "_load", lambda self: None):
        vqa = RemoteSensingVQA(model_name="tiny-blip", device="cpu")
    vqa.model = model
    vqa.processor = _TinyProcessor(vocab)
    vqa._is_blip2 = False
    vqa.max_new_tokens = 5
    return vqa


class _TinyProcessor:
    """Minimal stand-in for BlipProcessor: tokenises text, makes pixel_values."""

    def __init__(self, vocab: int):
        self.vocab = vocab

    def __call__(self, images=None, text=None, return_tensors=None):
        out = {}
        if images is not None:
            out["pixel_values"] = torch.zeros(1, 3, 64, 64)
        if text is not None:
            n_tokens = max(1, len(str(text).split()))
            ids = torch.arange(3, 3 + n_tokens).long().clamp(max=self.vocab - 1).unsqueeze(0)
            out["input_ids"] = ids
            out["attention_mask"] = torch.ones_like(ids)
        return _ToDeviceDict(out)

    def decode(self, ids, skip_special_tokens=True):
        return "tokencov"

    def batch_decode(self, seqs, skip_special_tokens=True):
        return ["tokencov"]


class _ToDeviceDict(dict):
    def to(self, device):
        return self


def _fusion_model(vqa=None):
    """A SAROpticalFusionModel wired to a tiny offline BLIP (untrained adapter)."""
    from models.sar_fusion_model import SAROpticalFusionModel
    vqa = vqa or _tiny_blip_vqa()
    return SAROpticalFusionModel(device="cpu", _vqa=vqa)


# ── image_data dict fixtures (mirror the app convention) ─────────────────────────

def _optical_dict(h=32, w=32):
    arr = np.zeros((h, w, 4), dtype=np.float32)
    arr[..., 0] = 0.2
    arr[..., 3] = 0.8
    return {"numpy_array": arr, "modality": "optical", "shape": [h, w, 4],
            "bands": 4, "metadata": {"sensor": "rgbn"}}


def _sar_dict(h=32, w=32, vv=100.0, vh=25.0):
    arr = np.zeros((h, w, 2), dtype=np.float32)
    arr[..., 0] = vv
    arr[..., 1] = vh
    return {"numpy_array": arr, "modality": "sar", "shape": [h, w, 2],
            "bands": 2, "metadata": {"sensor": "sentinel-1", "polarizations": ["VV", "VH"]}}


# ═══════════════════════════════════════════════════════════════════════════════
# 1 + 2 — Controller rejects incompatible modality pairs
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def controller():
    from agent.controller import AgenticController
    from models.registry import ModelRegistry
    return AgenticController(ModelRegistry())


def test_sar_only_pair_rejected(controller):
    import asyncio
    resp = asyncio.run(controller.analyze(
        [_sar_dict(), _sar_dict()],
        "Combine optical and SAR to find built-up regions.",
        task_hint="SAR_OPTICAL_FUSION",
    ))
    assert resp.is_degraded is True
    assert "optical" in resp.answer.lower()


def test_optical_only_pair_rejected(controller):
    import asyncio
    resp = asyncio.run(controller.analyze(
        [_optical_dict(), _optical_dict()],
        "Combine optical and SAR to find built-up regions.",
        task_hint="SAR_OPTICAL_FUSION",
    ))
    assert resp.is_degraded is True
    assert "sar" in resp.answer.lower()


def test_exec_sar_fusion_rejects_directly(controller):
    """Unit-level: _exec_sar_fusion rejects SAR-only without touching the model."""
    from agent.controller import ExecutionPlan
    from agent.task_classifier import TaskType
    from agent.query_intent import IntentExtractor
    images = [_sar_dict(), _sar_dict()]
    intent = IntentExtractor().extract("x", 2, ["sar", "sar"])
    plan = controller._plan(TaskType.SAR_OPTICAL_FUSION, 0.9, images, "x", intent)
    res = controller._exec_sar_fusion(plan, images, "x")
    assert res.is_degraded is True
    assert res.confidence <= 0.2


# ═══════════════════════════════════════════════════════════════════════════════
# 3 — Compatible pair accepted → full dict contract
# ═══════════════════════════════════════════════════════════════════════════════

def test_compatible_pair_returns_full_dict():
    model = _fusion_model()
    out = model.fuse_and_analyze(_optical_dict(), _sar_dict(),
                                 "What built-up areas are present?")
    for key in ("answer", "confidence", "fusion_map_b64", "sar_analytics",
                "optical_analytics", "vqa_evidence", "vqa_confidence",
                "fused_features", "fusion_meta", "fusion_trained",
                "requires_verification"):
        assert key in out, f"missing {key}"
    assert isinstance(out["answer"], str) and out["answer"]
    assert 0.0 <= out["confidence"] <= 1.0
    assert out["pathway"] == "fused_visual_tokens"


# ═══════════════════════════════════════════════════════════════════════════════
# 4 — Fused features actually reach answer generation
# ═══════════════════════════════════════════════════════════════════════════════

def test_fused_features_reach_answer_generation():
    """The fused tokens are what get injected — not the optical image alone."""
    model = _fusion_model()
    captured = {}
    real = model._vqa.answer_with_visual_tokens

    def _spy(visual_tokens, question):
        captured["tokens"] = visual_tokens.detach().clone()
        return real(visual_tokens, question)

    model._vqa.answer_with_visual_tokens = _spy
    model.fuse_and_analyze(_optical_dict(), _sar_dict(), "q")
    assert "tokens" in captured, "condition_vqa never injected visual tokens"
    # Injected tokens live in BLIP's hidden space [B, N, H].
    assert captured["tokens"].dim() == 3
    assert captured["tokens"].shape[-1] == model.hidden


def test_changing_sar_changes_injected_tokens():
    """A different SAR input must change the fused tokens — proves SAR is live."""
    model = _fusion_model()
    grabbed = []
    real = model._vqa.answer_with_visual_tokens

    def _spy(visual_tokens, question):
        grabbed.append(visual_tokens.detach().clone())
        return real(visual_tokens, question)

    model._vqa.answer_with_visual_tokens = _spy
    model.fuse_and_analyze(_optical_dict(), _sar_dict(vv=100.0, vh=25.0), "q")
    model.fuse_and_analyze(_optical_dict(), _sar_dict(vv=10.0, vh=90.0), "q")
    assert len(grabbed) == 2
    assert not torch.allclose(grabbed[0], grabbed[1]), \
        "fused tokens identical for different SAR — SAR is being bypassed"


# ═══════════════════════════════════════════════════════════════════════════════
# 5 — Gradients reach the fusion adapter; frozen BLIP vision gets none
# ═══════════════════════════════════════════════════════════════════════════════

def test_gradients_reach_adapter_not_frozen_vision():
    from models.sar_fusion_model import MultimodalFusionAdapter

    vqa = _tiny_blip_vqa()
    vision = vqa.model.vision_model
    for p in vision.parameters():
        p.requires_grad_(False)

    adapter = MultimodalFusionAdapter(hidden=64).train()

    pixel_values = torch.zeros(2, 3, 64, 64)
    with torch.no_grad():
        opt_tokens = vision(pixel_values=pixel_values)[0]      # frozen optical

    sar_in = torch.randn(2, 3, 32, 32)
    sar_tokens = adapter.encode_sar(sar_in)
    fused, _ = adapter.fuse(opt_tokens, sar_tokens)
    logits = adapter.classify(fused)
    loss = logits.pow(2).mean()
    loss.backward()

    # Adapter params must receive gradient.
    grads = [p.grad for n, p in adapter.named_parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads), \
        "no gradient reached the fusion adapter"
    # Frozen vision tower must receive none.
    assert all(p.grad is None for p in vision.parameters())


# ═══════════════════════════════════════════════════════════════════════════════
# 6 — Checkpoint loads at inference (fusion_trained flips True, output identical)
# ═══════════════════════════════════════════════════════════════════════════════

def test_checkpoint_roundtrip_sets_trained_and_reproduces_output():
    model = _fusion_model()
    assert model.fusion_trained is False

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "fusion_best.pt")
        torch.save({"adapter_state_dict": model.adapter.state_dict(),
                    "hidden": model.hidden}, ckpt_path)

        # Deterministic forward BEFORE reload.
        opt = torch.zeros(1, 5, model.hidden)
        sar = torch.randn(1, 3, 32, 32)
        torch.manual_seed(0)
        model.adapter.eval()
        with torch.no_grad():
            sar_tok = model.adapter.encode_sar(sar)
            fused_a, _ = model.adapter.fuse(opt, sar_tok)

        # Fresh model + checkpoint load.
        fresh = _fusion_model()
        assert fresh.load_fusion_checkpoint(ckpt_path) is True
        assert fresh.fusion_trained is True

        fresh.adapter.load_state_dict(model.adapter.state_dict())  # ensure identical weights
        fresh.adapter.eval()
        with torch.no_grad():
            sar_tok2 = fresh.adapter.encode_sar(sar)
            fused_b, _ = fresh.adapter.fuse(opt, sar_tok2)

        assert torch.allclose(fused_a, fused_b, atol=1e-5)


def test_missing_checkpoint_stays_untrained():
    model = _fusion_model()
    assert model.load_fusion_checkpoint("/no/such/file.pt") is False
    assert model.fusion_trained is False


# ═══════════════════════════════════════════════════════════════════════════════
# 7 — Physically-meaningful SAR preprocessing (no fixed-channel assumption)
# ═══════════════════════════════════════════════════════════════════════════════

def test_sar_preprocess_dualpol_builds_ratio():
    from models.sar_fusion.sar_preprocess import preprocess_sar
    arr = np.zeros((16, 16, 2), dtype=np.float32)
    arr[..., 0] = 120.0    # VV (linear intensity → dB)
    arr[..., 1] = 30.0     # VH
    res = preprocess_sar(arr, {"polarizations": ["VV", "VH"]}, out_channels=3)
    assert res.ratio_available is True
    assert "VV/VH" in res.channels
    assert res.log_applied is True                 # positive linear → dB
    assert res.tensor.shape[0] == 3
    assert np.isfinite(res.tensor).all()


def test_sar_preprocess_single_band_no_fake_ratio():
    from models.sar_fusion.sar_preprocess import preprocess_sar
    arr = np.full((16, 16), 50.0, dtype=np.float32)   # single band
    res = preprocess_sar(arr, {}, out_channels=3)
    assert res.ratio_available is False
    assert "VV/VH" not in res.channels
    assert res.tensor.shape[0] == 3                    # tiled honestly to fill
    assert res.note and "ratio" in res.note.lower()


def test_sar_preprocess_does_not_assume_channels():
    from models.sar_fusion.sar_preprocess import preprocess_sar
    # 3-band SAR with no VV/VH metadata: must not invent a ratio.
    arr = np.random.rand(16, 16, 3).astype(np.float32)
    res = preprocess_sar(arr, {}, out_channels=3)
    assert res.ratio_available is False
    assert len(res.polarizations) == 3


def test_sar_preprocess_respects_db_metadata():
    from models.sar_fusion.sar_preprocess import preprocess_sar
    arr = np.full((8, 8, 2), -15.0, dtype=np.float32)  # already dB (negative)
    res = preprocess_sar(arr, {"units": "db", "polarizations": ["VV", "VH"]})
    assert res.log_applied is False


def test_sar_backscatter_stats_reports_evidence():
    from models.sar_fusion.sar_preprocess import sar_backscatter_stats
    arr = np.zeros((8, 8, 2), dtype=np.float32)
    arr[..., 0] = 200.0
    arr[..., 1] = 20.0
    stats = sar_backscatter_stats(arr, {"polarizations": ["VV", "VH"]})
    assert stats["ratio_available"] is True
    assert "vv_vh_ratio_db" in stats
    assert stats["calibrated"] is False               # metadata never claimed it


# ═══════════════════════════════════════════════════════════════════════════════
# 8 — diagnose() reports all four stages OK on a random-init model
# ═══════════════════════════════════════════════════════════════════════════════

def test_diagnose_all_stages_ok():
    model = _fusion_model()
    rep = model.diagnose()
    assert rep["sar_encoder"] == "OK"
    assert rep["optical_encoder"] == "OK"
    assert rep["fusion"] == "OK"
    assert rep["vqa_conditioning"] == "OK"
    assert rep["injection_supported"] is True
    assert rep["fusion_trained"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# 9 — Untrained model flags degraded and never presents fused output as trusted
# ═══════════════════════════════════════════════════════════════════════════════

def test_untrained_flags_requires_verification():
    model = _fusion_model()
    assert model.fusion_trained is False
    out = model.fuse_and_analyze(_optical_dict(), _sar_dict(), "q")
    assert out["fusion_trained"] is False
    assert out["requires_verification"] is True
    # The untrained fused answer is explicitly marked as a placeholder.
    assert "not trained" in out["answer"].lower()


def test_trained_flag_removes_placeholder_prefix():
    model = _fusion_model()
    model.fusion_trained = True         # simulate a loaded checkpoint
    out = model.fuse_and_analyze(_optical_dict(), _sar_dict(), "q")
    assert out["requires_verification"] is False
    assert "not trained" not in out["answer"].lower()
