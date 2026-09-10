"""
Tests for the geographic claim guard.

Validates that:
  - BLIP-style geographic hallucinations are detected and stripped
  - Ordinary visual captions pass through unchanged
  - CRS metadata is recognized as independent evidence
  - No visual features (urban, road, vegetation) are stripped
  - The original caption is preserved internally for debugging
  - Edge cases are handled gracefully
"""
import os
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from models.geographic_guard import (
    GeographicGuardResult,
    apply_geographic_guard,
    detect_geographic_claims,
    sanitize_caption_for_display,
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _geo_image(crs="EPSG:4326", transform=None):
    """Image data dict with CRS metadata (simulates a GeoTIFF)."""
    return {
        "is_geotiff": True,
        "metadata": {
            "crs": crs,
            "transform": transform or [0.0, 1.0, 0.0, 0.0, -1.0, 0.0],
        },
    }


def _no_geo_image():
    """Image data dict without CRS metadata (simulates a plain PNG)."""
    return {
        "is_geotiff": False,
        "metadata": {},
    }


# ── London hallucination ─────────────────────────────────────────────────────

class TestLondonHallucination:
    def test_bare_of_london(self):
        """BLIP classic: 'an aerial view of london'"""
        result = apply_geographic_guard(
            "a satellite image showing an aerial view of london"
        )
        assert result.has_geographic_claims
        assert len(result.stripped_claims) > 0
        assert "london" not in result.sanitized_caption.lower()
        assert result.sanitized_caption  # not empty

    def test_in_london(self):
        result = apply_geographic_guard(
            "a satellite image showing buildings in London"
        )
        assert result.has_geographic_claims
        assert "london" not in result.sanitized_caption.lower()

    def test_near_london(self):
        result = apply_geographic_guard(
            "a satellite image showing fields near London"
        )
        assert result.has_geographic_claims
        assert "london" not in result.sanitized_caption.lower()

    def test_city_of_london(self):
        result = apply_geographic_guard(
            "a satellite image showing the city of London"
        )
        assert result.has_geographic_claims
        assert "london" not in result.sanitized_caption.lower()


# ── Kolkata hallucination ─────────────────────────────────────────────────────

class TestKolkataHallucination:
    def test_of_kolkata(self):
        result = apply_geographic_guard(
            "a satellite image showing an aerial view of Kolkata"
        )
        assert result.has_geographic_claims
        assert "kolkata" not in result.sanitized_caption.lower()

    def test_in_kolkata_india(self):
        result = apply_geographic_guard(
            "a satellite image showing an urban area in Kolkata, India"
        )
        assert result.has_geographic_claims
        assert "kolkata" not in result.sanitized_caption.lower()
        assert "india" not in result.sanitized_caption.lower()


# ── New York hallucination ────────────────────────────────────────────────────

class TestNewYorkHallucination:
    def test_of_new_york(self):
        result = apply_geographic_guard(
            "a satellite image showing an aerial view of New York"
        )
        assert result.has_geographic_claims
        assert "new york" not in result.sanitized_caption.lower()

    def test_in_new_york_city(self):
        result = apply_geographic_guard(
            "a satellite image showing dense buildings in New York City"
        )
        assert result.has_geographic_claims
        assert "new york" not in result.sanitized_caption.lower()


# ── India hallucination ───────────────────────────────────────────────────────

class TestIndiaHallucination:
    def test_of_india(self):
        result = apply_geographic_guard(
            "a satellite image showing a region of India"
        )
        assert result.has_geographic_claims
        assert "india" not in result.sanitized_caption.lower()

    def test_in_india(self):
        result = apply_geographic_guard(
            "a satellite image showing farmland in India"
        )
        assert result.has_geographic_claims
        assert "india" not in result.sanitized_caption.lower()


# ── Bidhannagar / specific Indian cities ──────────────────────────────────────

class TestSpecificIndianCities:
    def test_bidhannagar(self):
        result = apply_geographic_guard(
            "a satellite image showing the city of Bidhannagar"
        )
        assert result.has_geographic_claims
        assert "bidhannagar" not in result.sanitized_caption.lower()

    def test_kolkata_west_bengal(self):
        result = apply_geographic_guard(
            "a satellite image showing urban area in Kolkata, West Bengal"
        )
        assert result.has_geographic_claims
        assert "kolkata" not in result.sanitized_caption.lower()
        assert "west bengal" not in result.sanitized_caption.lower()


# ── No geographic claim (should pass through unchanged) ───────────────────────

class TestNoGeographicClaim:
    def test_urban_caption(self):
        caption = "a satellite image showing an aerial view of an urban area"
        result = apply_geographic_guard(caption)
        assert not result.has_geographic_claims
        assert result.sanitized_caption == caption
        assert len(result.stripped_claims) == 0

    def test_vegetation_caption(self):
        caption = "a satellite image showing dense vegetation and a river"
        result = apply_geographic_guard(caption)
        assert not result.has_geographic_claims
        assert result.sanitized_caption == caption

    def test_infrastructure_caption(self):
        caption = "a satellite image showing roads, buildings, and a bridge"
        result = apply_geographic_guard(caption)
        assert not result.has_geographic_claims
        assert result.sanitized_caption == caption

    def test_generic_scene(self):
        caption = "a satellite image showing an aerial view of a city"
        result = apply_geographic_guard(caption)
        assert not result.has_geographic_claims
        assert result.sanitized_caption == caption

    def test_rural_caption(self):
        caption = "a satellite image showing agricultural fields and scattered houses"
        result = apply_geographic_guard(caption)
        assert not result.has_geographic_claims
        assert result.sanitized_caption == caption

    def test_water_caption(self):
        caption = "a satellite image showing a large body of water with coastline"
        result = apply_geographic_guard(caption)
        assert not result.has_geographic_claims
        assert result.sanitized_caption == caption


# ── Verified geographic metadata ──────────────────────────────────────────────

class TestVerifiedGeographicMetadata:
    def test_crs_metadata_recognized(self):
        """CRS metadata is recognized as independent evidence."""
        caption = "a satellite image showing an aerial view of london"
        image_data = _geo_image()
        result = apply_geographic_guard(caption, image_data)
        assert result.has_independent_evidence
        assert "CRS" in result.verification_reason

    def test_no_crs_not_verified(self):
        """Without CRS metadata, geographic claims are not verified."""
        caption = "a satellite image showing an aerial view of london"
        result = apply_geographic_guard(caption, _no_geo_image())
        assert not result.has_independent_evidence
        assert "no independent" in result.verification_reason.lower()

    def test_none_image_data(self):
        """None image_data is handled gracefully."""
        caption = "a satellite image showing an aerial view of london"
        result = apply_geographic_guard(caption, None)
        assert not result.has_independent_evidence

    def test_crs_but_no_transform(self):
        """CRS without transform is not considered verified."""
        image_data = {
            "is_geotiff": True,
            "metadata": {"crs": "EPSG:4326", "transform": None},
        }
        result = apply_geographic_guard(
            "a satellite image showing an aerial view of london",
            image_data,
        )
        assert not result.has_independent_evidence


# ── Visual features are never stripped ────────────────────────────────────────

class TestVisualFeaturesPreserved:
    def test_urban_not_stripped(self):
        caption = "a satellite image showing an urban area"
        result = apply_geographic_guard(caption)
        assert "urban" in result.sanitized_caption.lower()

    def test_road_not_stripped(self):
        caption = "a satellite image showing a road"
        result = apply_geographic_guard(caption)
        assert "road" in result.sanitized_caption.lower()

    def test_vegetation_not_stripped(self):
        caption = "a satellite image showing dense vegetation"
        result = apply_geographic_guard(caption)
        assert "vegetation" in result.sanitized_caption.lower()

    def test_water_not_stripped(self):
        caption = "a satellite image showing a river"
        result = apply_geographic_guard(caption)
        assert "river" in result.sanitized_caption.lower()

    def test_building_not_stripped(self):
        caption = "a satellite image showing buildings"
        result = apply_geographic_guard(caption)
        assert "buildings" in result.sanitized_caption.lower()

    def test_forest_not_stripped(self):
        caption = "a satellite image showing a forest"
        result = apply_geographic_guard(caption)
        assert "forest" in result.sanitized_caption.lower()


# ── Original caption preserved for debugging ──────────────────────────────────

class TestOriginalPreserved:
    def test_original_in_result(self):
        original = "a satellite image showing an aerial view of london"
        result = apply_geographic_guard(original)
        assert result.original_caption == original

    def test_original_unchanged(self):
        original = "a satellite image showing an aerial view of london"
        result = apply_geographic_guard(original)
        assert result.original_caption != result.sanitized_caption


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_caption(self):
        result = apply_geographic_guard("")
        assert not result.has_geographic_claims
        assert result.sanitized_caption == ""

    def test_no_geographic_claims_in_prefix(self):
        """'A satellite image showing' should not be treated as a location."""
        caption = "A satellite image showing an aerial view of an urban area"
        result = apply_geographic_guard(caption)
        assert not result.has_geographic_claims

    def test_mixed_visual_and_geographic(self):
        caption = "a satellite image showing roads and buildings in Tokyo"
        result = apply_geographic_guard(caption)
        assert result.has_geographic_claims
        assert "tokyo" not in result.sanitized_caption.lower()
        assert "roads" in result.sanitized_caption.lower()
        assert "buildings" in result.sanitized_caption.lower()

    def test_multiple_geographic_claims(self):
        caption = "a satellite image showing an aerial view of Mumbai, India"
        result = apply_geographic_guard(caption)
        assert result.has_geographic_claims
        assert len(result.stripped_claims) >= 2
        assert "mumbai" not in result.sanitized_caption.lower()
        assert "india" not in result.sanitized_caption.lower()

    def test_lowercase_place_not_stripped(self):
        """Lowercase words should not be treated as geographic claims."""
        caption = "a satellite image showing an aerial view of a city"
        result = apply_geographic_guard(caption)
        assert not result.has_geographic_claims
        assert result.sanitized_caption == caption


# ── Sanitize convenience function ─────────────────────────────────────────────

class TestSanitizeConvenience:
    def test_returns_tuple(self):
        safe_caption, guard_result = sanitize_caption_for_display(
            "a satellite image showing an aerial view of london"
        )
        assert isinstance(guard_result, GeographicGuardResult)
        assert "london" not in safe_caption.lower()

    def test_with_image_data(self):
        safe_caption, guard_result = sanitize_caption_for_display(
            "a satellite image showing an aerial view of london",
            _geo_image(),
        )
        assert guard_result.has_independent_evidence
        assert "london" not in safe_caption.lower()


# ── Detection patterns ────────────────────────────────────────────────────────

class TestDetectionPatterns:
    def test_detect_preposition_in(self):
        claims = detect_geographic_claims("buildings in London")
        assert len(claims) >= 1
        assert any("London" in c.text for c in claims)

    def test_detect_preposition_near(self):
        claims = detect_geographic_claims("fields near Paris")
        assert len(claims) >= 1
        assert any("Paris" in c.text for c in claims)

    def test_detect_city_of(self):
        claims = detect_geographic_claims("the city of Tokyo")
        assert len(claims) >= 1
        assert any("Tokyo" in c.text for c in claims)

    def test_detect_end_location(self):
        claims = detect_geographic_claims("an aerial view of London")
        assert len(claims) >= 1
        assert any("London" in c.text for c in claims)

    def test_no_visual_features(self):
        claims = detect_geographic_claims("an urban area with roads")
        assert len(claims) == 0

    def test_compound_place_name(self):
        claims = detect_geographic_claims("buildings in New York")
        assert len(claims) >= 1
        assert any("New York" in c.text for c in claims)


# ── Regression: existing functionality ────────────────────────────────────────

class TestRegression:
    def test_vqa_not_affected(self):
        """VQA output should not be processed by geographic guard."""
        from models.geographic_guard import apply_geographic_guard
        # VQA answers like "urban" or "yes" should pass through
        result = apply_geographic_guard("urban")
        assert result.sanitized_caption == "urban"
        assert not result.has_geographic_claims

    def test_rs_clip_not_affected(self):
        """RS-CLIP output should not be processed by geographic guard."""
        result = apply_geographic_guard(
            "dense urban built-up area (RS-CLIP score: 0.63)"
        )
        assert result.sanitized_caption == "dense urban built-up area (RS-CLIP score: 0.63)"
        assert not result.has_geographic_claims

    def test_ndvi_evidence_not_affected(self):
        """NDVI tool evidence is a dict, not a caption string."""
        # This test ensures the guard only processes caption strings
        result = apply_geographic_guard("NDVI mean: 0.45")
        assert not result.has_geographic_claims
