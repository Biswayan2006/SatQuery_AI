"""
SatQuery AI — RS-CLIP inference package.

Public API::

    from models.rs_clip import RSCLIPEncoder

    encoder = RSCLIPEncoder.from_pretrained("ViT-B-32", checkpoint_path="...")
    img_emb  = encoder.encode_image(pil_image)          # np.ndarray [D]
    txt_emb  = encoder.encode_text("urban area")        # np.ndarray [D]
    score    = encoder.image_text_similarity(pil, text) # float ∈ [0, 1]
"""
from models.rs_clip.encoder import RSCLIPEncoder  # noqa: F401

__all__ = ["RSCLIPEncoder"]
