"""
SatQuery AI — Task Classifier Routing Tests

Verifies that natural-language queries are correctly routed to the
appropriate task type based on semantic intent, not just keyword matching.

Covers:
  - Generic scene description → CAPTIONING
  - Object/entity-specific description → SINGLE_VQA
  - Yes/no existence questions → SINGLE_VQA
  - Spatial localization questions → GROUNDING
  - Counting questions → GROUNDING
  - Change detection questions → CHANGE_VQA / CHANGE_DESCRIPTION
"""
from __future__ import annotations

import pytest

from agent.task_classifier import TaskClassifier, TaskType


@pytest.fixture
def classifier():
    """Default classifier (keyword-only mode, no semantic router)."""
    return TaskClassifier(semantic_router=None)


# ── CAPTIONING (generic scene description) ────────────────────────────────────

CAPTIONING_CASES = [
    "Describe the image.",
    "Describe the scene.",
    "Give an overview of the image.",
    "Summarize the scene.",
    "What does this image show?",
    "Tell me about this image.",
]


@pytest.mark.parametrize("query", CAPTIONING_CASES, ids=lambda q: q[:40])
class TestCaptioningRouting:
    def test_routes_to_captioning(self, classifier, query):
        task, conf = classifier.classify(query, num_images=1, modalities=["optical"])
        assert task == TaskType.CAPTIONING, (
            f"Expected CAPTIONING for '{query}', got {task.value}"
        )

    def test_confidence_positive(self, classifier, query):
        _, conf = classifier.classify(query, num_images=1, modalities=["optical"])
        assert conf > 0.3


# ── SINGLE_VQA (object/entity-specific description) ──────────────────────────

VQA_DESCRIBE_CASES = [
    "Describe the road.",
    "Describe the building.",
    "Describe the river.",
    "Describe the vegetation.",
    "Describe the highway.",
    "Describe the bridge.",
    "Describe the forest.",
    "Describe the water body.",
    "Describe the parking lot.",
    "Describe the airport.",
]


@pytest.mark.parametrize("query", VQA_DESCRIBE_CASES, ids=lambda q: q[:40])
class TestVQADescribeRouting:
    def test_routes_to_vqa(self, classifier, query):
        task, conf = classifier.classify(query, num_images=1, modalities=["optical"])
        assert task == TaskType.SINGLE_VQA, (
            f"Expected SINGLE_VQA for '{query}', got {task.value}"
        )

    def test_confidence_strong(self, classifier, query):
        _, conf = classifier.classify(query, num_images=1, modalities=["optical"])
        assert conf >= 0.5


# ── SINGLE_VQA (yes/no and existence questions) ──────────────────────────────

VQA_EXISTENCE_CASES = [
    "Is there a road?",
    "Is there a building?",
    "Is there water?",
    "Is there a bridge?",
    "Is there vegetation?",
    "Are there vehicles?",
    "What type of road is visible?",
]


@pytest.mark.parametrize("query", VQA_EXISTENCE_CASES, ids=lambda q: q[:40])
class TestVQAExistenceRouting:
    def test_routes_to_vqa(self, classifier, query):
        task, conf = classifier.classify(query, num_images=1, modalities=["optical"])
        assert task == TaskType.SINGLE_VQA, (
            f"Expected SINGLE_VQA for '{query}', got {task.value}"
        )


# ── GROUNDING (spatial localization) ─────────────────────────────────────────

GROUNDING_LOCALIZATION_CASES = [
    "Where is the road?",
    "Locate the building.",
    "Where is the bridge?",
    "Show me the water body.",
    "Identify the vehicles.",
    "Find the river.",
    "Detect the structures.",
]


@pytest.mark.parametrize("query", GROUNDING_LOCALIZATION_CASES, ids=lambda q: q[:40])
class TestGroundingLocalizationRouting:
    def test_routes_to_grounding(self, classifier, query):
        task, conf = classifier.classify(query, num_images=1, modalities=["optical"])
        assert task == TaskType.GROUNDING, (
            f"Expected GROUNDING for '{query}', got {task.value}"
        )


# ── GROUNDING (counting) ────────────────────────────────────────────────────

GROUNDING_COUNTING_CASES = [
    "How many buildings are there?",
    "How many vehicles are visible?",
    "Count the trees.",
    "Number of structures in the image.",
]


@pytest.mark.parametrize("query", GROUNDING_COUNTING_CASES, ids=lambda q: q[:40])
class TestGroundingCountingRouting:
    def test_routes_to_grounding(self, classifier, query):
        task, conf = classifier.classify(query, num_images=1, modalities=["optical"])
        assert task == TaskType.GROUNDING, (
            f"Expected GROUNDING for '{query}', got {task.value}"
        )


# ── CHANGE tasks ─────────────────────────────────────────────────────────────

CHANGE_VQA_CASES = [
    "What changed between these images?",
    "How much has changed?",
    "Is there deforestation?",
]

CHANGE_DESC_CASES = [
    "Describe the changes.",
    "What happened to the landscape?",
    "Summarize changes.",
]


@pytest.mark.parametrize("query", CHANGE_VQA_CASES, ids=lambda q: q[:40])
class TestChangeVQARouting:
    def test_routes_to_change(self, classifier, query):
        task, conf = classifier.classify(
            query, num_images=2, modalities=["optical", "optical"]
        )
        assert task in (TaskType.CHANGE_VQA, TaskType.CHANGE_DESCRIPTION), (
            f"Expected CHANGE task for '{query}', got {task.value}"
        )


@pytest.mark.parametrize("query", CHANGE_DESC_CASES, ids=lambda q: q[:40])
class TestChangeDescriptionRouting:
    def test_routes_to_change(self, classifier, query):
        task, conf = classifier.classify(
            query, num_images=2, modalities=["optical", "optical"]
        )
        assert task in (TaskType.CHANGE_VQA, TaskType.CHANGE_DESCRIPTION), (
            f"Expected CHANGE task for '{query}', got {task.value}"
        )


# ── EDGE CASES ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_query_defaults_to_vqa(self, classifier):
        task, _ = classifier.classify("", num_images=1, modalities=["optical"])
        assert task == TaskType.SINGLE_VQA

    def test_two_images_no_query_defaults_to_change(self, classifier):
        task, _ = classifier.classify(
            "", num_images=2, modalities=["optical", "optical"]
        )
        assert task in (TaskType.CHANGE_VQA, TaskType.CHANGE_DESCRIPTION)

    def test_sar_single_image_prefers_vqa(self, classifier):
        task, _ = classifier.classify(
            "What is visible?", num_images=1, modalities=["sar"]
        )
        # SAR + generic query should not route to captioning
        assert task != TaskType.CAPTIONING

    def test_fusion_routing_with_sar_optical_pair(self, classifier):
        task, _ = classifier.classify(
            "Compare these images",
            num_images=2,
            modalities=["sar", "optical"],
        )
        assert task == TaskType.SAR_OPTICAL_FUSION
