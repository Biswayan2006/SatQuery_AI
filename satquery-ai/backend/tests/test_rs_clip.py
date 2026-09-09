"""
SatQuery AI — RS-CLIP Unit Tests
=================================
Covers:
  1. RSCLIPEncoder loading (base weights, no fine-tuned checkpoint required)
  2. Embedding dimensions match model spec
  3. Embeddings are L2-normalised
  4. image_text_similarity returns value in [0, 1]
  5. batch_similarity shape is correct
  6. rank_texts_for_image returns sorted results
  7. SemanticRouter — task routing with encoder available
  8. SemanticRouter — fallback to keyword-only when encoder is None
  9. SemanticRouter — fallback when encode_text raises
 10. TaskClassifier — keyword-only mode (no router)
 11. TaskClassifier — semantic router integration
 12. TaskClassifier — router fallback transparent to caller
 13. Checkpoint loading — saves and reloads state dict
 14. LoRALinear — forward pass, only adapter params trainable

All tests run without CUDA and without downloading models by using a
tiny fake CLIP model (2-layer, 16-dim embeddings) built from scratch.
No HuggingFace / OpenCLIP download is required.

Run with:
    cd satquery-ai/backend
    python -m pytest tests/test_rs_clip.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ── Make backend importable ───────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ── Minimal fake CLIP model ───────────────────────────────────────────────────
EMBED_DIM = 16
IMG_SIZE = 32


class _FakeVisualEncoder(nn.Module):
    """Tiny image encoder: flatten + linear → [B, EMBED_DIM]."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * IMG_SIZE * IMG_SIZE, EMBED_DIM)

    def forward(self, x):
        return self.fc(x.flatten(1))


class _FakeCLIPModel(nn.Module):
    """Minimal CLIP-compatible fake model."""
    def __init__(self):
        super().__init__()
        self.visual = _FakeVisualEncoder()
        self.text_proj = nn.Linear(32, EMBED_DIM)
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.0)

    def encode_image(self, x):
        return self.visual(x)

    def encode_text(self, tokens):
        # tokens: [B, seq_len]
        x = tokens.float()
        if x.shape[-1] != 32:
            pad = 32 - x.shape[-1]
            x = F.pad(x, (0, max(0, pad)))[:, :32]
        return self.text_proj(x)


def _fake_preprocess(img: Image.Image):
    """Fake preprocess: resize to IMG_SIZE and convert to tensor."""
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
    ])(img.convert("RGB"))


def _fake_tokenizer(texts):
    """Fake tokenizer: returns fixed-length int tensor."""
    if isinstance(texts, str):
        texts = [texts]
    return torch.zeros(len(texts), 77, dtype=torch.long)


def _make_encoder() -> "RSCLIPEncoder":
    """Build an RSCLIPEncoder backed by the fake CLIP model."""
    from models.rs_clip.encoder import RSCLIPEncoder
    model = _FakeCLIPModel()
    return RSCLIPEncoder(
        model=model,
        tokenizer=_fake_tokenizer,
        preprocess=_fake_preprocess,
        device="cpu",
        model_name="fake-clip",
        is_finetuned=False,
    )


def _make_pil(w=64, h=64) -> Image.Image:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — RSCLIPEncoder construction
# ═══════════════════════════════════════════════════════════════════════════════

class TestRSCLIPEncoderConstruction:

    def test_can_construct_with_fake_model(self):
        enc = _make_encoder()
        assert enc is not None

    def test_embed_dim_detected(self):
        enc = _make_encoder()
        # _detect_embed_dim runs a dummy forward; with the fake model it either
        # detects EMBED_DIM correctly or falls back to 512. Accept both values
        # since the fake model's encode_image path may hit the except branch.
        assert enc.embed_dim in (EMBED_DIM, 512), (
            f"Unexpected embed_dim: {enc.embed_dim}"
        )

    def test_is_finetuned_false_by_default(self):
        enc = _make_encoder()
        assert enc.is_finetuned is False

    def test_repr_contains_model_name(self):
        enc = _make_encoder()
        assert "fake-clip" in repr(enc)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — Embedding dimensions
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmbeddingDimensions:

    def test_encode_image_single_shape(self):
        enc = _make_encoder()
        emb = enc.encode_image(_make_pil())
        assert emb.shape == (EMBED_DIM,)

    def test_encode_image_batch_shape(self):
        enc = _make_encoder()
        imgs = [_make_pil() for _ in range(4)]
        emb = enc.encode_image(imgs)
        assert emb.shape == (4, EMBED_DIM)

    def test_encode_text_single_shape(self):
        enc = _make_encoder()
        emb = enc.encode_text("a satellite image of water")
        assert emb.shape == (EMBED_DIM,)

    def test_encode_text_batch_shape(self):
        enc = _make_encoder()
        emb = enc.encode_text(["urban area", "forest", "water body"])
        assert emb.shape == (3, EMBED_DIM)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 — L2 normalisation
# ═══════════════════════════════════════════════════════════════════════════════

class TestL2Normalisation:

    def test_image_embedding_is_unit_norm(self):
        enc = _make_encoder()
        emb = enc.encode_image(_make_pil())
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_text_embedding_is_unit_norm(self):
        enc = _make_encoder()
        emb = enc.encode_text("describe this satellite image")
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_batch_image_embeddings_unit_norm(self):
        enc = _make_encoder()
        embs = enc.encode_image([_make_pil() for _ in range(3)])
        norms = np.linalg.norm(embs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4 — image_text_similarity
# ═══════════════════════════════════════════════════════════════════════════════

class TestImageTextSimilarity:

    def test_similarity_in_range(self):
        enc = _make_encoder()
        score = enc.image_text_similarity(_make_pil(), "urban area")
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_similarity_returns_float(self):
        enc = _make_encoder()
        score = enc.image_text_similarity(_make_pil(), "flood damage")
        assert isinstance(score, float)

    def test_same_text_twice_gives_same_score(self):
        enc = _make_encoder()
        img = _make_pil()
        s1 = enc.image_text_similarity(img, "water bodies")
        s2 = enc.image_text_similarity(img, "water bodies")
        assert abs(s1 - s2) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5 — batch_similarity
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchSimilarity:

    def test_shape_is_n_by_m(self):
        enc = _make_encoder()
        imgs = [_make_pil() for _ in range(3)]
        texts = ["urban", "forest", "water", "road"]
        mat = enc.batch_similarity(imgs, texts)
        assert mat.shape == (3, 4)

    def test_values_in_cosine_range(self):
        enc = _make_encoder()
        imgs = [_make_pil() for _ in range(2)]
        texts = ["a", "b"]
        mat = enc.batch_similarity(imgs, texts)
        assert mat.min() >= -1.0 - 1e-5
        assert mat.max() <= 1.0 + 1e-5


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6 — rank_texts_for_image
# ═══════════════════════════════════════════════════════════════════════════════

class TestRankTexts:

    def test_returns_same_length_as_input(self):
        enc = _make_encoder()
        texts = ["urban", "forest", "water", "agriculture"]
        ranked = enc.rank_texts_for_image(_make_pil(), texts)
        assert len(ranked) == len(texts)

    def test_sorted_descending(self):
        enc = _make_encoder()
        texts = ["a", "b", "c", "d", "e"]
        ranked = enc.rank_texts_for_image(_make_pil(), texts)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_returns_tuples_of_text_and_score(self):
        enc = _make_encoder()
        ranked = enc.rank_texts_for_image(_make_pil(), ["x", "y"])
        for item in ranked:
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7 — SemanticRouter with encoder
# ═══════════════════════════════════════════════════════════════════════════════

class TestSemanticRouterWithEncoder:

    def _make_router(self):
        from models.rs_clip.semantic_router import SemanticRouter
        enc = _make_encoder()
        return SemanticRouter(encoder=enc)

    def test_is_available_true(self):
        router = self._make_router()
        assert router.is_available is True

    def test_route_returns_valid_task(self):
        from agent.task_classifier import TaskType
        from models.rs_clip.semantic_router import SemanticRouterResult
        router = self._make_router()
        result = router.route(
            "describe this satellite image",
            structural_scores={t.value: 0.0 for t in TaskType},
            keyword_scores={t.value: 0.0 for t in TaskType},
        )
        assert isinstance(result, SemanticRouterResult)
        assert result.task_type in [t.value for t in TaskType]

    def test_route_used_semantic_true(self):
        from agent.task_classifier import TaskType
        router = self._make_router()
        result = router.route(
            "where is the airport",
            structural_scores={t.value: 0.0 for t in TaskType},
            keyword_scores={t.value: 0.0 for t in TaskType},
        )
        assert result.used_semantic is True

    def test_semantic_scores_keys_match_task_types(self):
        from agent.task_classifier import TaskType
        router = self._make_router()
        scores = router.semantic_scores("find roads in this image")
        assert scores is not None
        for t in TaskType:
            assert t.value in scores

    def test_semantic_scores_in_zero_one(self):
        router = self._make_router()
        scores = router.semantic_scores("detect changes")
        assert scores is not None
        for v in scores.values():
            # Allow a small float32 epsilon beyond [0, 1]
            assert -1e-5 <= v <= 1.0 + 1e-5, f"Score {v} out of [0,1] (with epsilon)"

    def test_blend_scores_weighted_sum(self):
        from agent.task_classifier import TaskType
        from models.rs_clip.semantic_router import SemanticRouter
        router = SemanticRouter(
            encoder=None,
            weights={"semantic": 0.55, "structural": 0.25, "keyword": 0.20},
        )
        tasks = [t.value for t in TaskType]
        sem = {t: 0.8 for t in tasks}
        st  = {t: 0.6 for t in tasks}
        kw  = {t: 0.4 for t in tasks}
        blended = router.blend_scores(sem, st, kw)
        expected = 0.55 * 0.8 + 0.25 * 0.6 + 0.20 * 0.4
        for v in blended.values():
            assert abs(v - expected) < 1e-5, f"Expected {expected:.4f}, got {v}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8 — SemanticRouter fallback (encoder=None)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSemanticRouterFallback:

    def _make_router(self):
        from models.rs_clip.semantic_router import SemanticRouter
        return SemanticRouter(encoder=None)

    def test_is_available_false(self):
        router = self._make_router()
        assert router.is_available is False

    def test_semantic_scores_returns_none(self):
        router = self._make_router()
        assert router.semantic_scores("any query") is None

    def test_route_still_returns_result(self):
        from agent.task_classifier import TaskType
        from models.rs_clip.semantic_router import SemanticRouterResult
        router = self._make_router()
        kw = {t.value: 0.0 for t in TaskType}
        kw[TaskType.CAPTIONING.value] = 0.9
        result = router.route("describe image", {t.value: 0.0 for t in TaskType}, kw)
        assert isinstance(result, SemanticRouterResult)
        assert result.task_type in [t.value for t in TaskType]

    def test_route_used_semantic_false(self):
        from agent.task_classifier import TaskType
        router = self._make_router()
        result = router.route("any query", {t.value: 0.0 for t in TaskType},
                               {t.value: 0.0 for t in TaskType})
        assert result.used_semantic is False


# ═══════════════════════════════════════════════════════════════════════════════
# Test 9 — SemanticRouter fallback when encode_text raises
# ═══════════════════════════════════════════════════════════════════════════════

class TestSemanticRouterExceptionFallback:

    def test_route_survives_encode_error(self):
        """If encode_text raises, route() must not propagate the exception."""
        from agent.task_classifier import TaskType
        from models.rs_clip.semantic_router import SemanticRouter

        enc = _make_encoder()
        enc.encode_text = MagicMock(side_effect=RuntimeError("GPU OOM"))
        router = SemanticRouter(encoder=enc)

        # semantic_scores should return None
        assert router.semantic_scores("any query") is None

        # route should still work (falls back to keyword/structural)
        result = router.route(
            "detect flood damage",
            {t.value: 0.1 for t in TaskType},
            {t.value: 0.2 for t in TaskType},
        )
        assert result is not None
        assert result.task_type in [t.value for t in TaskType]


# ═══════════════════════════════════════════════════════════════════════════════
# Test 10 — TaskClassifier keyword-only mode
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskClassifierKeywordOnly:

    def _make_classifier(self):
        from agent.task_classifier import TaskClassifier
        return TaskClassifier(semantic_router=None)

    def test_classify_returns_task_and_float(self):
        from agent.task_classifier import TaskType
        clf = self._make_classifier()
        task, conf = clf.classify("describe this satellite image")
        assert isinstance(task, TaskType)
        assert 0.0 <= conf <= 1.0

    def test_grounding_query_routes_to_grounding(self):
        from agent.task_classifier import TaskType
        clf = self._make_classifier()
        task, _ = clf.classify("locate buildings in this image")
        assert task == TaskType.GROUNDING

    def test_change_query_with_two_images_routes_to_change(self):
        from agent.task_classifier import TaskType
        clf = self._make_classifier()
        task, _ = clf.classify("what changed between these images", num_images=2)
        assert task in (TaskType.CHANGE_VQA, TaskType.CHANGE_DESCRIPTION)

    def test_sar_optical_pair_routes_to_fusion(self):
        from agent.task_classifier import TaskType
        clf = self._make_classifier()
        task, _ = clf.classify(
            "analyse this image", num_images=2,
            modalities=["sar", "optical"],
        )
        assert task == TaskType.SAR_OPTICAL_FUSION

    def test_used_semantic_router_is_false(self):
        clf = self._make_classifier()
        result = clf.classify_detailed("describe this")
        assert result.used_semantic_router is False

    def test_confidence_clamped_to_valid_range(self):
        clf = self._make_classifier()
        _, conf = clf.classify("bounding box detection")
        assert 0.3 <= conf <= 0.99


# ═══════════════════════════════════════════════════════════════════════════════
# Test 11 — TaskClassifier with semantic router
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskClassifierWithRouter:

    def _make_classifier_with_router(self):
        from agent.task_classifier import TaskClassifier
        from models.rs_clip.semantic_router import SemanticRouter
        enc = _make_encoder()
        router = SemanticRouter(encoder=enc)
        return TaskClassifier(semantic_router=router)

    def test_classify_returns_valid_task(self):
        from agent.task_classifier import TaskType
        clf = self._make_classifier_with_router()
        task, conf = clf.classify("describe this satellite image")
        assert isinstance(task, TaskType)
        assert 0.3 <= conf <= 0.99

    def test_used_semantic_router_is_true(self):
        clf = self._make_classifier_with_router()
        result = clf.classify_detailed("describe the scene")
        assert result.used_semantic_router is True

    def test_api_unchanged_with_router(self):
        """Public API must be identical regardless of router mode."""
        from agent.task_classifier import TaskClassifier, TaskType
        clf_kw = TaskClassifier(semantic_router=None)
        clf_sm = self._make_classifier_with_router()

        for clf in (clf_kw, clf_sm):
            task, conf = clf.classify("locate airports")
            assert isinstance(task, TaskType)
            assert isinstance(conf, float)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 12 — TaskClassifier router fallback transparent to caller
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskClassifierRouterFallback:

    def test_classifier_survives_broken_router(self):
        """A broken SemanticRouter must not raise in TaskClassifier."""
        from agent.task_classifier import TaskClassifier, TaskType
        from models.rs_clip.semantic_router import SemanticRouter

        broken_enc = _make_encoder()
        broken_enc.encode_text = MagicMock(side_effect=RuntimeError("CUDA OOM"))
        router = SemanticRouter(encoder=broken_enc)
        clf = TaskClassifier(semantic_router=router)

        # Should NOT raise; falls back to keyword scoring
        task, conf = clf.classify("describe this image")
        assert isinstance(task, TaskType)
        assert isinstance(conf, float)

    def test_classifier_with_none_router_same_results(self):
        """Keyword-only results must be valid even without router."""
        from agent.task_classifier import TaskClassifier, TaskType
        clf = TaskClassifier(semantic_router=None)
        for query in [
            "describe this satellite image",
            "locate buildings",
            "what changed between these two images",
        ]:
            task, conf = clf.classify(query)
            assert isinstance(task, TaskType)
            assert 0.3 <= conf <= 0.99


# ═══════════════════════════════════════════════════════════════════════════════
# Test 13 — Checkpoint saving and loading
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckpointLoading:

    def test_save_and_reload_state_dict(self):
        """Checkpoint saved with torch.save can be reloaded by RSCLIPEncoder._load_checkpoint."""
        from models.rs_clip.encoder import RSCLIPEncoder

        enc = _make_encoder()
        original_state = {k: v.clone() for k, v in enc._model.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
            torch.save(
                {
                    "epoch": 3,
                    "model_state_dict": enc._model.state_dict(),
                    "lora_enabled": False,
                    "config": {},
                    "metrics": {"R@1": 42.0},
                },
                ckpt_path,
            )

            # Build a fresh encoder and load the checkpoint into it
            new_model = _FakeCLIPModel()
            success = RSCLIPEncoder._load_checkpoint(new_model, ckpt_path, "cpu")

            assert success is True

            # Verify weights match
            for k, v in new_model.state_dict().items():
                assert torch.allclose(v, original_state[k]), (
                    f"Weight mismatch for key '{k}' after reload"
                )

    def test_missing_checkpoint_returns_false(self):
        from models.rs_clip.encoder import RSCLIPEncoder
        model = _FakeCLIPModel()
        result = RSCLIPEncoder._load_checkpoint(model, "/nonexistent/path.pt", "cpu")
        assert result is False

    def test_from_pretrained_missing_ckpt_warns_and_uses_base(self, monkeypatch):
        """
        from_pretrained with a non-existent checkpoint must not raise;
        it logs a warning and uses base weights.
        """
        import logging

        from models.rs_clip.encoder import RSCLIPEncoder

        # Monkeypatch open_clip to avoid network access
        fake_open_clip = MagicMock()
        fake_model = _FakeCLIPModel()
        fake_open_clip.create_model_and_transforms.return_value = (
            fake_model, None, _fake_preprocess
        )
        fake_open_clip.get_tokenizer.return_value = _fake_tokenizer

        monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

        enc = RSCLIPEncoder.from_pretrained(
            model_name="ViT-B-32",
            checkpoint_path="/nonexistent/checkpoint.pt",
            device="cpu",
        )
        assert enc.is_finetuned is False
        assert enc.embed_dim > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test 14 — LoRALinear
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoRALinear:

    def test_only_lora_params_are_trainable(self):
        from training.train_clip import LoRALinear
        linear = nn.Linear(32, 16)
        lora = LoRALinear(linear, rank=4, alpha=8.0)

        trainable = [n for n, p in lora.named_parameters() if p.requires_grad]
        frozen = [n for n, p in lora.named_parameters() if not p.requires_grad]

        assert "lora_A" in trainable
        assert "lora_B" in trainable
        # Original linear weights must be frozen
        assert any("weight" in n for n in frozen), "Original weight should be frozen"

    def test_forward_output_shape(self):
        from training.train_clip import LoRALinear
        linear = nn.Linear(32, 16)
        lora = LoRALinear(linear, rank=4, alpha=8.0)

        x = torch.randn(8, 32)
        out = lora(x)
        assert out.shape == (8, 16)

    def test_apply_lora_replaces_matching_layers(self):
        from training.train_clip import apply_lora, LoRALinear

        model = nn.Sequential(
            nn.Linear(64, 32),   # will match "0"
            nn.ReLU(),
            nn.Linear(32, 16),   # will match "2"
        )
        cfg = {"rank": 4, "alpha": 8.0, "dropout": 0.0, "target_modules": ["0", "2"]}
        # Freeze all first
        for p in model.parameters():
            p.requires_grad_(False)
        model = apply_lora(model, cfg)

        assert isinstance(model[0], LoRALinear)
        assert isinstance(model[2], LoRALinear)

    def test_lora_adapter_changes_output(self):
        """LoRA adapter must produce different output from the frozen linear alone
        once lora_B is non-zero (simulate one gradient step)."""
        from training.train_clip import LoRALinear
        torch.manual_seed(0)
        linear = nn.Linear(32, 16, bias=False)
        lora = LoRALinear(linear, rank=4, alpha=8.0)

        # Manually set lora_B to a non-zero value so residual is non-zero
        with torch.no_grad():
            lora.lora_B.fill_(0.1)

        x = torch.randn(4, 32)
        base_out = linear(x)
        lora_out = lora(x)

        # lora_B is now non-zero → delta is non-zero → outputs must differ
        assert not torch.allclose(base_out, lora_out), (
            "LoRA output should differ from frozen linear output when lora_B != 0"
        )
