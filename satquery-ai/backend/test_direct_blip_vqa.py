"""
SatQuery AI — Direct BLIP VQA Diagnostic Test

Bypasses TaskClassifier, AgenticController, ExecutionPlanner, ResultIntegration.
Tests the BLIP VQA model directly on an actual image.

Purpose: distinguish PIPELINE FAILURE from MODEL CAPABILITY LIMITATION.
"""
from __future__ import annotations

import os
import sys
import time

# Ensure we can import from the backend directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_direct_blip_vqa():
    """Run BLIP VQA directly on a test image and report raw outputs."""
    from PIL import Image

    # ── 1. Find a test image ─────────────────────────────────────────────────
    fixtures_dir = os.path.join(os.path.dirname(__file__), "_audit_fixtures")
    test_image_path = None
    for candidate in ["opt_a.png", "opt_b.png"]:
        p = os.path.join(fixtures_dir, candidate)
        if os.path.exists(p):
            test_image_path = p
            break

    if test_image_path is None:
        # Generate a synthetic image with a visible road-like feature
        print("No test fixture found, generating synthetic road image...")
        img = Image.new("RGB", (512, 512), (80, 120, 60))  # green background
        pixels = img.load()
        # Draw a grey road across the image
        for x in range(512):
            for y in range(200, 320):
                pixels[x, y] = (128, 128, 128)
        # Draw some building-like rectangles
        for bx in range(50, 150):
            for by in range(50, 120):
                pixels[bx, by] = (180, 160, 140)
        test_image_path = os.path.join(fixtures_dir, "_synthetic_road.png")
        img.save(test_image_path)
    else:
        img = Image.open(test_image_path)

    print(f"Test image: {test_image_path}")
    print(f"Image size: {img.size}, mode: {img.mode}")

    # ── 2. Load BLIP VQA directly ────────────────────────────────────────────
    from config import get_settings
    settings = get_settings()

    from models.vqa_model import RemoteSensingVQA
    print(f"\nLoading BLIP VQA model ({settings.vqa_model_name})...")
    t0 = time.perf_counter()
    vqa = RemoteSensingVQA(
        model_name=settings.vqa_model_name,
        device=settings.resolved_device,
        cache_dir=settings.model_cache_dir,
    )
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f}s on {vqa.device}")

    # ── 3. Run diagnostic questions ──────────────────────────────────────────
    questions = [
        "Is there a road in this image?",
        "What is the main road like?",
        "Describe the road.",
        "What is visible in the image?",
        "Is there a building?",
        "Is there water?",
    ]

    print("\n" + "=" * 72)
    print("DIRECT BLIP VQA TEST RESULTS")
    print("=" * 72)

    results = []
    for q in questions:
        prompt = RemoteSensingVQA._build_prompt(q)
        t0 = time.perf_counter()
        result = vqa.answer(img, q)
        infer_time = time.perf_counter() - t0

        entry = {
            "question": q,
            "prompt_sent": prompt,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "confidence_is_calibrated": result.get("confidence_is_calibrated", False),
            "evidence": result.get("evidence", {}),
            "inference_time_s": round(infer_time, 3),
        }
        results.append(entry)

        print(f"\nQ: {q}")
        print(f"  Prompt sent:  {prompt}")
        print(f"  Answer:       {result['answer']}")
        print(f"  Confidence:   {result['confidence']:.4f} (calibrated: {result.get('confidence_is_calibrated', False)})")
        print(f"  Inference:    {infer_time:.3f}s")
        if result.get("evidence", {}).get("token_logprobs"):
            print(f"  Token logprobs: {result['evidence']['token_logprobs']}")

    # ── 4. Summary assessment ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ASSESSMENT")
    print("=" * 72)

    # Check if road-related questions got reasonable answers
    road_answers = [r for r in results if "road" in r["question"].lower()]
    yes_answers = sum(1 for r in road_answers if "yes" in r["answer"].lower())
    total_road = len(road_answers)

    print(f"Road-related questions: {total_road}")
    print(f"Positive road answers:  {yes_answers}/{total_road}")

    if yes_answers >= total_road * 0.5:
        print("VERDICT: BLIP VQA can partially understand roads in this image.")
    elif yes_answers > 0:
        print("VERDICT: BLIP VQA shows limited road recognition — model capability is marginal.")
    else:
        print("VERDICT: BLIP VQA fails to recognize roads — MODEL DOMAIN LIMITATION.")

    print("\nNOTE: This test isolates MODEL CAPABILITY from PIPELINE ROUTING.")
    print("If routing is correct but answers are wrong, the issue is model-domain gap,")
    print("not software bugs.")

    return results


if __name__ == "__main__":
    results = test_direct_blip_vqa()
