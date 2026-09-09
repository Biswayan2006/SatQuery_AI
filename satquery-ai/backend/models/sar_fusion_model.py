"""
SatQuery AI — SAR-Optical Fusion Model
======================================
A **genuine multimodal feature pathway** that fuses SAR and optical imagery and
uses the fused representation to condition VQA answer generation.

What changed (and why it matters)
---------------------------------
The previous implementation encoded both modalities, produced a fused vector,
then *threw it away*: the answer came from running the plain VQA model on the
optical image alone, with SAR reduced to a few words appended to the prompt.
The fused features never influenced the answer.

This version builds a real pathway:

    SAR   ─▶ encode_sar   ─▶ SAR visual tokens   [B, N_s, H]
    Optical ▶ encode_optical ▶ optical visual tokens [B, N_o, H]
                                   │
                                   ▼
                        fuse_modalities (cross-attention + gated residual)
                                   │
                                   ▼
                    fused visual tokens [B, N_o, H]
                                   │
    Question ──────────────────────┼─▶ condition_vqa
                                   ▼
                        multimodal answer (BLIP cross-attention)

The fused tokens are injected into BLIP's ``encoder_hidden_states`` slot — the
same place the vision encoder's ``image_embeds`` normally go — so both
modalities condition the whole answer.  The fused vector is **never** flattened
into a text prompt.

Modular interface
-----------------
  * :meth:`encode_optical` — optical → visual tokens (reuses BLIP's vision tower)
  * :meth:`encode_sar`     — SAR → visual tokens (trainable SAR encoder)
  * :meth:`fuse_modalities`— cross-modal fusion → fused tokens (+ gate map)
  * :meth:`condition_vqa`  — fused tokens + question → answer

Honesty
-------
The fusion adapter and SAR encoder are randomly initialised until a checkpoint
is loaded.  Without a checkpoint, :attr:`fusion_trained` is False, the pathway
still runs (so it is testable), but results are flagged as untrained /
requiring verification and are NEVER presented as trustworthy.  See
:meth:`diagnose` for a component-by-component readiness report.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from models.sar_fusion.sar_preprocess import preprocess_sar, sar_backscatter_stats

logger = logging.getLogger("satquery.sar_fusion")

# Number of land-cover classes (BigEarthNet-43) for the optional classifier head.
NUM_LANDCOVER_CLASSES = 43


def _pick_num_heads(hidden: int) -> int:
    """Largest attention-head count that divides ``hidden`` (cap 12)."""
    for h in (12, 8, 6, 4, 2, 1):
        if hidden % h == 0 and h <= max(1, hidden // 8):
            return h
    return 1


# ── SAR encoder ─────────────────────────────────────────────────────────────────

class SAREncoder(nn.Module):
    """
    Lightweight convolutional SAR encoder → visual tokens ``[B, N, H]``.

    Trained from scratch (no ImageNet download); operates on the
    physically-meaningful SAR channels produced by ``preprocess_sar`` (VV, VH,
    VV/VH where available).  A conv stem downsamples to a token grid, then each
    grid cell becomes a token projected to the fusion hidden size ``H``.
    """

    def __init__(self, in_channels: int = 3, hidden: int = 768, grid: int = 12):
        super().__init__()
        self.grid = grid
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.proj = nn.Linear(256, hidden)
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] → feature map → [B, grid*grid, hidden]
        feat = self.pool(self.stem(x))                 # [B, 256, grid, grid]
        b, c, gh, gw = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)       # [B, grid*grid, 256]
        tokens = self.proj(tokens)                     # [B, N, hidden]
        return tokens + self.pos[:, : tokens.size(1)]


# ── Multimodal fusion adapter ───────────────────────────────────────────────────

class MultimodalFusionAdapter(nn.Module):
    """
    Cross-modal fusion of optical + SAR visual tokens.

    Optical tokens are the queries; SAR tokens are the keys/values.  A
    multi-head cross-attention block lets each optical token attend to the SAR
    evidence, and a learned per-token gate controls how much SAR signal is
    injected (a gated residual, so an untrained adapter degrades toward the
    optical tokens rather than corrupting them).  Output keeps the optical token
    layout ``[B, N_o, H]`` so it drops straight into BLIP's cross-attention.

    Auxiliary heads (used only during training) provide the parameter-efficient
    objectives: a projection for contrastive modality alignment and a
    land-cover classifier on the pooled fused tokens.
    """

    def __init__(
        self,
        hidden: int = 768,
        sar_in_channels: int = 3,
        sar_grid: int = 12,
        proj_dim: int = 256,
        num_classes: int = NUM_LANDCOVER_CLASSES,
    ):
        super().__init__()
        self.hidden = hidden
        self.sar_encoder = SAREncoder(sar_in_channels, hidden, sar_grid)

        heads = _pick_num_heads(hidden)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=heads, batch_first=True
        )
        self.norm_q = nn.LayerNorm(hidden)
        self.norm_kv = nn.LayerNorm(hidden)
        self.gate_proj = nn.Linear(hidden, 1)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2), nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )

        # Auxiliary training heads (contrastive alignment + land-cover class).
        self.opt_proj = nn.Linear(hidden, proj_dim)
        self.sar_proj = nn.Linear(hidden, proj_dim)
        self.classifier = nn.Linear(hidden, num_classes)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))  # ln(1/0.07)

        # Bias the gate slightly closed so a fresh adapter leans on optical.
        nn.init.constant_(self.gate_proj.bias, -1.0)

    def encode_sar(self, sar_tensor: torch.Tensor) -> torch.Tensor:
        return self.sar_encoder(sar_tensor)

    def fuse(
        self,
        optical_tokens: torch.Tensor,
        sar_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        q = self.norm_q(optical_tokens)
        kv = self.norm_kv(sar_tokens)
        attn_out, attn_weights = self.cross_attn(q, kv, kv, need_weights=True)
        gate = torch.sigmoid(self.gate_proj(attn_out))         # [B, N_o, 1]
        fused = optical_tokens + gate * attn_out               # gated residual
        fused = fused + self.ffn(fused)

        gate_map = gate.squeeze(-1)                            # [B, N_o]
        meta = {
            "gate_mean": float(gate.mean().item()),
            "gate_map": gate_map.detach(),
            "attn_entropy": float(
                (-(attn_weights.clamp_min(1e-9) *
                   attn_weights.clamp_min(1e-9).log()).sum(-1)).mean().item()
            ),
        }
        return fused, meta

    # Auxiliary objectives (training only) ------------------------------------
    def contrastive_features(
        self, optical_tokens: torch.Tensor, sar_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        opt = F.normalize(self.opt_proj(optical_tokens.mean(dim=1)), dim=-1)
        sar = F.normalize(self.sar_proj(sar_tokens.mean(dim=1)), dim=-1)
        return opt, sar

    def classify(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        return self.classifier(fused_tokens.mean(dim=1))


# ── Fusion model (inference orchestration) ──────────────────────────────────────

class SAROpticalFusionModel:
    """
    Orchestrates the SAR-optical fusion pathway and answers queries.

    Parameters
    ----------
    device : str
    cache_dir : str, optional
    vqa_model_name : str
        BLIP VQA model used both as the optical vision tower and the
        answer-generation decoder.  BLIP (not BLIP-2) is required for
        visual-token injection; other backends fall back to a deterministic,
        clearly-flagged answer.
    fusion_checkpoint : str, optional
        Path to a trained adapter checkpoint (produced by
        ``training/train_fusion.py``).  Absent → ``fusion_trained=False``.
    """

    def __init__(
        self,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        vqa_model_name: str = "Salesforce/blip-vqa-base",
        fusion_checkpoint: Optional[str] = None,
        _vqa: Any = None,
        _adapter: Optional[MultimodalFusionAdapter] = None,
    ):
        self.device = device
        self.cache_dir = cache_dir
        self.vqa_model_name = vqa_model_name
        self.fusion_checkpoint = fusion_checkpoint or None
        self._vqa = _vqa
        self.adapter: Optional[MultimodalFusionAdapter] = _adapter
        self.fusion_trained = False
        self.hidden = 768
        self._injection_supported = False

        if _vqa is None:
            self._load_vqa()
        self._init_adapter()
        if self.fusion_checkpoint:
            self.load_fusion_checkpoint(self.fusion_checkpoint)

    # ── Loading ────────────────────────────────────────────────────────────────

    def _load_vqa(self) -> None:
        try:
            from models.vqa_model import RemoteSensingVQA
            self._vqa = RemoteSensingVQA(
                model_name=self.vqa_model_name,
                device=self.device,
                cache_dir=self.cache_dir,
            )
        except Exception as exc:
            logger.warning("VQA sub-model for SAR fusion unavailable: %s", exc)
            self._vqa = None

    def _vqa_vision_hidden(self) -> int:
        """Discover the BLIP vision hidden size, defaulting to 768."""
        try:
            return int(self._vqa.model.config.vision_config.hidden_size)
        except Exception:
            return 768

    def _init_adapter(self) -> None:
        self._injection_supported = self._detect_injection_support()
        if self._injection_supported:
            self.hidden = self._vqa_vision_hidden()
        if self.adapter is None:
            self.adapter = MultimodalFusionAdapter(hidden=self.hidden)
        self.adapter = self.adapter.to(self.device).eval()

    def _detect_injection_support(self) -> bool:
        """True only for a real BLIP model exposing the injection pathway."""
        vqa = self._vqa
        if vqa is None or getattr(vqa, "_is_blip2", False):
            return False
        model = getattr(vqa, "model", None)
        return bool(
            model is not None
            and hasattr(model, "vision_model")
            and hasattr(model, "text_encoder")
            and hasattr(model, "text_decoder")
            and hasattr(vqa, "answer_with_visual_tokens")
        )

    def load_fusion_checkpoint(self, path: str) -> bool:
        """
        Load a trained adapter checkpoint.  Returns True on success and sets
        ``fusion_trained=True``.  Missing / invalid checkpoint → stays untrained.
        """
        if not path or not os.path.exists(path):
            logger.info("No fusion checkpoint at %s — adapter stays untrained.", path)
            self.fusion_trained = False
            return False
        try:
            ckpt = torch.load(path, map_location=self.device)
            state = ckpt.get("adapter_state_dict", ckpt)
            hidden = ckpt.get("hidden", self.hidden)
            if hidden != self.hidden:
                self.hidden = hidden
                self.adapter = MultimodalFusionAdapter(hidden=hidden)
            self.adapter.load_state_dict(state, strict=False)
            self.adapter = self.adapter.to(self.device).eval()
            self.fusion_trained = True
            logger.info("Loaded fusion adapter checkpoint from %s", path)
            return True
        except Exception as exc:
            logger.error("Fusion checkpoint load failed: %s", exc)
            self.fusion_trained = False
            return False

    # ── Modular interface ──────────────────────────────────────────────────────

    def encode_optical(self, optical: Union[Image.Image, "np.ndarray", torch.Tensor]) -> torch.Tensor:
        """Optical image → BLIP visual tokens ``[B, N, H]``."""
        if not self._injection_supported:
            raise NotImplementedError("Optical token encoding requires a BLIP VQA model.")
        pixel_values = self._optical_pixel_values(optical)
        with torch.no_grad():
            vout = self._vqa.model.vision_model(pixel_values=pixel_values)
        return vout[0]

    def encode_sar(self, sar: Union["np.ndarray", torch.Tensor, Image.Image],
                   metadata: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Raw SAR array → SAR visual tokens ``[B, N, H]``."""
        sar_tensor = self._sar_tensor(sar, metadata)
        return self.adapter.encode_sar(sar_tensor)

    def fuse_modalities(
        self, optical_tokens: torch.Tensor, sar_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Cross-modal fusion → fused visual tokens ``[B, N_o, H]`` + gate meta."""
        return self.adapter.fuse(optical_tokens, sar_tokens)

    def condition_vqa(self, fused_tokens: torch.Tensor, question: str) -> Dict[str, Any]:
        """Answer ``question`` conditioned on the fused visual tokens."""
        if not self._injection_supported:
            raise NotImplementedError("VQA conditioning requires a BLIP VQA model.")
        return self._vqa.answer_with_visual_tokens(fused_tokens, question)

    # ── Public API (orchestration) ───────────────────────────────────────────────

    def fuse_and_analyze(
        self,
        optical: Any,
        sar: Any,
        query: str,
    ) -> Dict[str, Any]:
        """
        Fuse optical + SAR and answer ``query`` through the genuine multimodal
        pathway.  ``optical`` / ``sar`` may be image_data dicts (preferred — the
        raw SAR array + metadata enable physically-meaningful preprocessing) or
        PIL images / arrays.

        Returns the established dict contract (answer, confidence, fusion_map_b64,
        sar_analytics, optical_analytics, vqa_evidence, vqa_confidence) plus
        ``fused_features``, ``fusion_meta``, ``fusion_trained`` and
        ``requires_verification``.
        """
        opt_pil = self._as_optical_pil(optical)
        sar_arr, sar_meta = self._as_sar_array(sar)

        # Deterministic evidence (never fabricated; independent of the network).
        sar_analytics = self._analyze_sar(sar_arr, sar_meta)
        opt_analytics = self._analyze_optical(opt_pil)

        fused_features = None
        fusion_meta: Dict[str, Any] = {}
        answer = None
        conf = None
        vqa_evidence = None
        vqa_confidence = None
        pathway = "deterministic"

        if self._injection_supported:
            try:
                opt_tokens = self.encode_optical(opt_pil)
                sar_tokens = self.encode_sar(sar_arr, sar_meta)
                fused, fusion_meta = self.fuse_modalities(opt_tokens, sar_tokens)
                fused_features = fused
                vqa_out = self.condition_vqa(fused, query)
                answer = vqa_out.get("answer")
                conf = float(vqa_out.get("confidence", 0.5))
                vqa_evidence = vqa_out.get("evidence")
                vqa_confidence = vqa_out.get("confidence")
                pathway = "fused_visual_tokens"
            except Exception as exc:
                logger.warning("Fusion pathway failed (%s) — deterministic fallback.", exc)

        if answer is None:
            # Honest fallback: no learned fusion available; report the
            # deterministic multimodal evidence rather than a fake answer.
            answer = self._deterministic_answer(query, sar_analytics, opt_analytics)
            conf = conf if conf is not None else 0.4

        # Untrained adapter → the fused answer is not trustworthy.
        requires_verification = (
            pathway == "fused_visual_tokens" and not self.fusion_trained
        )
        if requires_verification:
            answer = (
                "[Fusion adapter not trained — treat as a placeholder] " + answer
            )

        fusion_b64 = self._create_fusion_viz(opt_pil, sar_arr, fusion_meta)

        return {
            "answer": answer,
            "confidence": round(float(conf), 4),
            "fusion_map_b64": fusion_b64,
            "sar_analytics": sar_analytics,
            "optical_analytics": opt_analytics,
            "vqa_evidence": vqa_evidence,
            "vqa_confidence": vqa_confidence,
            # New structured outputs.
            "fused_features": (
                fused_features.detach().cpu().numpy().tolist()
                if isinstance(fused_features, torch.Tensor) and fused_features.numel() < 20000
                else None
            ),
            "fusion_meta": {k: v for k, v in fusion_meta.items() if k != "gate_map"},
            "fusion_trained": self.fusion_trained,
            "pathway": pathway,
            "requires_verification": requires_verification,
        }

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def diagnose(self) -> Dict[str, Any]:
        """
        Component-by-component readiness report.  Runs a dummy pair through each
        stage; each stage is isolated so one failure is reported, not raised.
        Reports pathway wiring — NOT model quality.
        """
        report: Dict[str, Any] = {
            "sar_encoder": "not_run",
            "optical_encoder": "not_run",
            "fusion": "not_run",
            "vqa_conditioning": "not_run",
            "injection_supported": self._injection_supported,
            "fusion_trained": self.fusion_trained,
        }

        # SAR encoder.
        sar_tokens = None
        try:
            dummy_sar = np.random.rand(2, 32, 32).astype(np.float32) * 0.1
            sar_tokens = self.encode_sar(dummy_sar, {"polarizations": ["VV", "VH"]})
            report["sar_encoder"] = "OK"
        except Exception as exc:
            report["sar_encoder"] = f"error: {exc}"

        # Optical encoder.
        opt_tokens = None
        try:
            opt_tokens = self.encode_optical(Image.new("RGB", (64, 64), (120, 120, 120)))
            report["optical_encoder"] = "OK"
        except NotImplementedError as exc:
            report["optical_encoder"] = f"unsupported: {exc}"
        except Exception as exc:
            report["optical_encoder"] = f"error: {exc}"

        # Fusion.
        fused = None
        try:
            if sar_tokens is not None and opt_tokens is not None:
                fused, _ = self.fuse_modalities(opt_tokens, sar_tokens)
                report["fusion"] = "OK"
            elif sar_tokens is not None:
                # Exercise fusion against self-shaped optical tokens so the
                # fusion block is still verified when the vision tower is absent.
                fake_opt = torch.zeros(1, 4, self.hidden)
                fused, _ = self.fuse_modalities(fake_opt, sar_tokens)
                report["fusion"] = "OK (optical tower unavailable — used placeholder queries)"
        except Exception as exc:
            report["fusion"] = f"error: {exc}"

        # VQA conditioning.
        try:
            if self._injection_supported and fused is not None:
                out = self.condition_vqa(fused, "What is shown?")
                report["vqa_conditioning"] = "OK" if out.get("answer") is not None else "error: no answer"
            else:
                report["vqa_conditioning"] = "unsupported: BLIP injection pathway unavailable"
        except Exception as exc:
            report["vqa_conditioning"] = f"error: {exc}"

        return report

    # ── Input coercion ───────────────────────────────────────────────────────────

    def _optical_pixel_values(self, optical) -> torch.Tensor:
        if isinstance(optical, torch.Tensor):
            t = optical if optical.dim() == 4 else optical.unsqueeze(0)
            return t.to(self.device)
        pil = self._as_optical_pil(optical)
        proc = self._vqa.processor(images=pil, return_tensors="pt")
        return proc["pixel_values"].to(self.device)

    def _sar_tensor(self, sar, metadata) -> torch.Tensor:
        if isinstance(sar, torch.Tensor):
            t = sar if sar.dim() == 4 else sar.unsqueeze(0)
            return t.float().to(self.device)
        arr, meta = self._as_sar_array(sar) if not isinstance(sar, np.ndarray) else (sar, metadata or {})
        pre = preprocess_sar(arr, meta, out_channels=3)
        return torch.from_numpy(pre.tensor).unsqueeze(0).float().to(self.device)

    @staticmethod
    def _as_optical_pil(optical) -> Image.Image:
        if isinstance(optical, Image.Image):
            return optical.convert("RGB") if optical.mode != "RGB" else optical
        if isinstance(optical, dict):
            pil = optical.get("pil_image")
            if pil is not None:
                return pil.convert("RGB") if pil.mode != "RGB" else pil
            arr = optical.get("numpy_array")
            if arr is not None:
                from utils.image_utils import normalize_to_rgb
                return normalize_to_rgb(arr, optical.get("bands", arr.shape[-1] if arr.ndim == 3 else 1))
        if isinstance(optical, np.ndarray):
            from utils.image_utils import normalize_to_rgb
            return normalize_to_rgb(optical, optical.shape[-1] if optical.ndim == 3 else 1)
        return Image.new("RGB", (224, 224), (128, 128, 128))

    @staticmethod
    def _as_sar_array(sar) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract a raw SAR array + metadata for physically-meaningful preprocessing."""
        if isinstance(sar, dict):
            arr = sar.get("numpy_array")
            meta = dict(sar.get("metadata") or {})
            if arr is not None:
                return np.asarray(arr, dtype=np.float32), meta
            pil = sar.get("pil_image")
            if pil is not None:
                return np.asarray(pil.convert("L"), dtype=np.float32), meta
        if isinstance(sar, np.ndarray):
            return np.asarray(sar, dtype=np.float32), {}
        if isinstance(sar, Image.Image):
            return np.asarray(sar.convert("L"), dtype=np.float32), {}
        return np.zeros((32, 32), dtype=np.float32), {}

    # ── Deterministic analytics (evidence the model must not fabricate) ──────────

    @staticmethod
    def _analyze_sar(sar_arr: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Physically-grounded SAR statistics from real polarisation channels."""
        return sar_backscatter_stats(sar_arr, metadata)

    @staticmethod
    def _analyze_optical(opt_image: Image.Image) -> Dict[str, Any]:
        arr = np.array(opt_image.convert("RGB"), dtype=np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        ndvi_proxy = (g.mean() - r.mean()) / (g.mean() + r.mean() + 1e-6)
        water_proxy = float((b > r + 20).mean() * 100)
        return {
            "mean_rgb": [round(float(c.mean()), 2) for c in (r, g, b)],
            "ndvi_proxy": round(float(ndvi_proxy), 4),
            "water_pct_estimate": round(water_proxy, 2),
            "dominant_feature": (
                "water" if water_proxy > 30
                else "vegetation" if ndvi_proxy > 0.1
                else "urban / bare soil"
            ),
        }

    @staticmethod
    def _deterministic_answer(query: str, sar: Dict, opt: Dict) -> str:
        """Honest fallback answer built purely from deterministic evidence."""
        scattering = sar.get("dominant_scattering", "an unresolved SAR surface type")
        ratio = (
            f", VV/VH ratio {sar['vv_vh_ratio_db']:.1f} dB"
            if sar.get("ratio_available") else ""
        )
        return (
            f"SAR-optical evidence: optical suggests {opt['dominant_feature']} "
            f"(NDVI proxy {opt['ndvi_proxy']:.3f}); SAR backscatter indicates "
            f"{scattering}{ratio}."
        )

    def _create_fusion_viz(self, optical: Image.Image, sar_arr: np.ndarray,
                           fusion_meta: Dict[str, Any]) -> Optional[str]:
        """Fusion visualisation using the learned cross-modal gate map."""
        try:
            from utils.visualization import create_fusion_visualization
            from utils.image_utils import image_to_base64

            sar_pil = self._sar_to_pil(sar_arr)
            gate_map = fusion_meta.get("gate_map")
            heat = gate_map.float().cpu().numpy() if isinstance(gate_map, torch.Tensor) else None
            viz = create_fusion_visualization(optical, sar_pil, heat)
            return image_to_base64(viz)
        except Exception as exc:
            logger.warning("Fusion visualization failed: %s", exc)
            return None

    @staticmethod
    def _sar_to_pil(sar_arr: np.ndarray) -> Image.Image:
        arr = np.asarray(sar_arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        u8 = (((arr - lo) / (hi - lo) * 255) if hi > lo else np.zeros_like(arr)).astype(np.uint8)
        return Image.fromarray(u8, mode="L").convert("RGB")
