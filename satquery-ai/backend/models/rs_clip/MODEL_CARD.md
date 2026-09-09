# SatQuery AI — RS-CLIP Model Card

## Model Overview

| Field | Value |
|---|---|
| **Model name** | SatQuery RS-CLIP |
| **Version** | 1.0 |
| **Base model** | OpenCLIP ViT-B/32 (`openai` pretrained weights) |
| **Fine-tuning method** | LoRA (Low-Rank Adaptation, rank=8, alpha=16) on attention projection layers |
| **Framework** | [open-clip-torch](https://github.com/mlfoundations/open_clip) |
| **Task** | Remote-sensing image–text representation learning |
| **Embedding dimension** | 512 (ViT-B/32) |
| **Input image size** | 224 × 224 px (RGB) |

---

## Intended Use

RS-CLIP is a contrastively fine-tuned CLIP model adapted for remote-sensing imagery.

### Supported inference use cases

1. **Semantic task routing** — embedding user queries and task descriptions to select the best analysis pipeline
2. **Answer verification** — computing similarity between a generated caption and the original query
3. **Remote-sensing semantic similarity** — comparing two images or an image and a text at the semantic level
4. **Caption reranking** — ranking multiple candidate captions by relevance to a query or image
5. **Future retrieval** — image-to-image or text-to-image search over a corpus

### Out-of-scope uses

- Direct visual question answering (use the BLIP/BLIP-2 VQA model)
- Pixel-level segmentation or object detection
- SAR-to-optical translation
- Medical, forensic, or surveillance applications

---

## Training Datasets

| Dataset | Split | Samples | License |
|---|---|---|---|
| [BigEarthNet-S2](https://bigearth.net/) | train + val | ~269 000 | [Community Data License Agreement – Permissive, Version 1.0](https://cdla.dev/permissive-1-0/) |
| [VRSBench](https://arxiv.org/abs/2406.12681) | train + val | ~29 614 (image–caption pairs) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

> **Note:** Datasets are NOT bundled with this repository. Configure paths in
> `training/configs/rs_clip_base.yaml` before training.

### Training data limitations

- BigEarthNet covers European land-cover predominantly (Sentinel-2, 10–60 m GSD).
- VRSBench includes diverse RS scenes but skews toward optical nadir imagery.
- Neither dataset contains significant SAR, hyperspectral, or nighttime imagery.
- Text descriptions are label-derived (BigEarthNet) or human-annotated (VRSBench);
  they may not cover all remote-sensing terminology.

---

## Training Method

### Contrastive learning

Symmetric InfoNCE loss (CLIP-style):

```
L = (CE(I→T) + CE(T→I)) / 2
```

where image and text features are L2-normalised before the dot product.

### Parameter-efficient fine-tuning (LoRA)

By default only LoRA adapter weights are trained (~0.5 % of total parameters).
Full fine-tuning is available but not recommended without at least 16 GB VRAM.

| Hyperparameter | Default |
|---|---|
| Base model | ViT-B/32 (openai) |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA target layers | `attn.in_proj`, `attn.out_proj` |
| Learning rate | 5 × 10⁻⁶ |
| Batch size (effective) | 256 (64 × 4 grad accumulation) |
| Epochs | 15 |
| Mixed precision | fp16 |
| LR scheduler | Cosine annealing |
| Warmup steps | 200 |
| Weight decay | 0.1 |
| Early stopping patience | 4 epochs |
| Optimizer | AdamW |

### Training command

```bash
cd satquery-ai/backend
python training/train_clip.py --config training/configs/rs_clip_base.yaml
```

---

## Evaluation Metrics

Metrics below are illustrative targets; actual values depend on the fine-tuning
run and available data.

### BigEarthNet image-to-text retrieval (test split)

| Metric | Base ViT-B/32 (no fine-tuning) | RS-CLIP fine-tuned (target) |
|---|---|---|
| R@1 | ~15 % | ~40 % |
| R@5 | ~35 % | ~65 % |
| R@10 | ~48 % | ~78 % |

### VRSBench image-to-text retrieval (test split)

| Metric | Base ViT-B/32 | RS-CLIP fine-tuned (target) |
|---|---|---|
| R@1 | ~22 % | ~52 % |
| R@5 | ~50 % | ~78 % |
| R@10 | ~63 % | ~87 % |

> These are approximate targets based on related work.
> Run `training/evaluate_clip.py` to obtain numbers for your checkpoint.

### Zero-shot classification (BigEarthNet-43 classes)

| Metric | Base ViT-B/32 | RS-CLIP fine-tuned (target) |
|---|---|---|
| Top-1 accuracy | ~28 % | ~45 % |

---

## Known Limitations

1. **Not fine-tuned for SAR imagery.** The model was trained on optical RGB composites. SAR backscatter patterns are not well-represented in the embedding space.
2. **European geographic bias.** BigEarthNet is derived from Sentinel-2 scenes over Europe; equatorial, arid, and polar landscapes may be under-represented.
3. **LoRA adapters are small.** With rank=8, capacity for domain shift is limited. Consider rank=16 or full fine-tuning for very specialised domains.
4. **Text descriptions are label-derived.** BigEarthNet text is generated from land-cover class names. Natural-language diversity is lower than human-annotated captions.
5. **No temporal / multi-temporal understanding.** The model encodes individual images; it has no explicit representation of change over time.
6. **224 × 224 input.** Large-scale satellite tiles are downsampled; fine spatial detail is lost.
7. **Base CLIP confidence is not calibrated.** The `image_text_similarity` score is a rescaled cosine, not a probability.

---

## Checkpoint Location

Default checkpoint path (configured in `training/configs/rs_clip_base.yaml`):

```
satquery-ai/backend/checkpoints/rs_clip/rs_clip_best.pt
```

Set `rs_clip_checkpoint` in the application `.env` or override at load time:

```python
from models.rs_clip import RSCLIPEncoder
encoder = RSCLIPEncoder.from_pretrained(
    checkpoint_path="./checkpoints/rs_clip/rs_clip_best.pt"
)
```

---

## License

- **Base model weights** (OpenCLIP ViT-B/32, openai): Subject to [OpenAI CLIP model license](https://github.com/openai/CLIP/blob/main/LICENSE).
- **LoRA adapter weights trained on BigEarthNet**: [Community Data License Agreement – Permissive 1.0](https://cdla.dev/permissive-1-0/).
- **LoRA adapter weights trained on VRSBench**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Training and inference code (SatQuery AI)**: See root `LICENSE` file.

---

*Generated automatically by SatQuery AI build pipeline.*
*Last updated: 2026-09-05*
