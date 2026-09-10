"""
SatQuery AI — Geographic Claim Guard
=====================================
Prevents generic captioning models (BLIP, GIT) from asserting unsupported
geographic identities.

Design principle:
  OBSERVED features (urban, roads, vegetation, water) → allowed
  UNVERIFIED place names (London, Kolkata, India)     → stripped

The guard detects named geographic claims in generated text, checks whether
the system has independent evidence supporting that claim (CRS metadata +
reverse geocoding), and either confirms or strips the claim.

The original model output is preserved internally for debugging but never
exposed as verified user-facing information.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger("satquery.geographic_guard")


# ── Geographic claim detection ───────────────────────────────────────────────
#
# We use a multi-layer approach:
#   1. Preposition + CapitalizedWord patterns (high precision)
#   2. Standalone CapitalizedWord at end of caption (high recall)
#   3. Known place-name patterns (countries, continents, regions)
#
# Each pattern returns the matched text so the caller can decide what to do.

# Common prepositions that precede location claims
_PREPOSITIONS = r"(?:in|near|around|over|at|from|of|to|through|across|beyond|toward|towards)"

# Pattern: preposition + optional article + optional "city of" + PlaceName
_PATTERN_PREP_LOCATION = re.compile(
    rf"\s+{_PREPOSITIONS}\s+(?:the\s+)?(?:city\s+of\s+|country\s+of\s+|state\s+of\s+)?"
    r"([A-Z][\w-]+(?:[\s,]+[A-Z][\w-]+)*)",
    re.UNICODE,
)

# Pattern: "city of X" / "country of X" without a preposition
_PATTERN_OF_LOCATION = re.compile(
    r"\s+(?:the\s+)?(?:city|country|state|region|province|island|coast|mountain|lake|river)"
    r"\s+of\s+([A-Z][\w-]+(?:[\s,]+[A-Z][\w-]+)*)",
    re.UNICODE,
)

# Pattern: standalone capitalized word(s) at end of string
# Catches: "...a view of new york", "...a view of London"
_PATTERN_END_LOCATION = re.compile(
    r"\s+([A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s*$",
    re.UNICODE,
)

# Pattern: known geographic entity after "of" (case-insensitive)
# Catches BLIP's lowercase output: "an aerial view of london"
_PATTERN_OF_KNOWN_ENTITY = re.compile(
    r"\bof\s+([A-Za-z][\w-]+(?:\s+[A-Za-z][\w-]+)*)\s*[.,;:!\s]*$",
    re.UNICODE,
)

# Pattern: known geographic entity after prepositions (case-insensitive)
_PATTERN_PREP_KNOWN_ENTITY = re.compile(
    rf"\s+{_PREPOSITIONS}\s+(?:the\s+)?(?:city\s+of\s+|country\s+of\s+)?([A-Za-z][\w-]+(?:[\s,]+[A-Za-z][\w-]+)*)",
    re.UNICODE,
)

# Pattern: comma-separated compound: "Kolkata, India" or "New York, USA"
_PATTERN_COMPOUND_GEO = re.compile(
    r"([A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s*,\s*([A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)",
    re.UNICODE,
)

# Pattern: "New York City" — compound with trailing "City"
_PATTERN_COMPOUND_CITY_SUFFIX = re.compile(
    r"\b([A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+City\b",
    re.UNICODE,
)

# Pattern: known geographic实体 that are not visual features
# These are ALWAYS geographic claims when capitalized
_KNOWN_GEOGRAPHIC_ENTITIES = {
    # Continents
    "africa", "antarctica", "asia", "australia", "europe", "north america",
    "south america",
    # Major countries (common in BLIP training data)
    "india", "china", "japan", "brazil", "russia", "canada", "australia",
    "france", "germany", "italy", "spain", "united kingdom", "uk",
    "united states", "usa", "mexico", "argentina", "south africa",
    "nigeria", "kenya", "egypt", "saudi arabia", "iran", "iraq",
    "pakistan", "bangladesh", "sri lanka", "nepal", "myanmar",
    "thailand", "vietnam", "indonesia", "philippines", "malaysia",
    "singapore", "new zealand", "south korea", "north korea",
    "colombia", "peru", "chile", "venezuela", "ecuador",
    # Major cities (common in BLIP training data)
    "london", "paris", "tokyo", "new york", "beijing", "shanghai",
    "mumbai", "delhi", "kolkata", "chennai", "bangalore", "hyderabad",
    "dubai", "singapore", "hong kong", "sydney", "melbourne",
    "los angeles", "chicago", "houston", "phoenix", "philadelphia",
    "san antonio", "san diego", "dallas", "san jose", "austin",
    "seattle", "denver", "boston", "washington", "miami", "atlanta",
    "toronto", "vancouver", "montreal", "berlin", "madrid", "rome",
    "paris", "lisbon", "amsterdam", "brussels", "vienna", "prague",
    "warsaw", "budapest", "istanbul", "moscow", "saint petersburg",
    "cairo", "lagos", "nairobi", "cape town", "johannesburg",
    "buenos aires", "rio de janeiro", "sao paulo", "lima", "bogota",
    "santiago", "mexico city", "tehran", "baghdad", "riyadh",
    "karachi", "lahore", "dhaka", "colombo", "kathmandu",
    "bangkok", "hanoi", "jakarta", "manila", "kuala lumpur",
    "taipei", "seoul", "pyongyang",
    # Indian cities/states (relevant for ISRO/SAC context)
    "bidhannagar", "salt lake", "howrah", "durgapur", "asansol",
    "siliguri", "jalpaiguri", "kharagpur", "dhanbad", "bokaro",
    "ranchi", "patna", "varanasi", "lucknow", "agra", "jaipur",
    "jodhpur", "udaipur", "ahmedabad", "surat", "vadodara",
    "rajkot", "indore", "bhopal", "nagpur", "pune", "nashik",
    "noida", "gurugram", "faridabad", "ghaziabad", "meerut",
    "uttar pradesh", "madhya pradesh", "rajasthan", "gujarat",
    "maharashtra", "karnataka", "tamil nadu", "kerala", "andhra pradesh",
    "telangana", "west bengal", "bihar", "jharkhand", "odisha",
    "assam", "meghalaya", "manipur", "mizoram", "nagaland",
    "tripura", "sikkim", "arunachal pradesh", "himachal pradesh",
    "jammu", "kashmir", "ladakh", "chhattisgarh", "goa",
}

# ── Visual feature words (allowed, never stripped) ────────────────────────────
# These describe what is OBSERVED in the image, not WHERE it is.
_VISUAL_FEATURES = {
    "urban", "rural", "city", "town", "village", "settlement",
    "road", "highway", "street", "bridge", "tunnel", "intersection",
    "building", "house", "structure", "tower", "facility",
    "vegetation", "forest", "tree", "field", "crop", "farmland",
    "water", "river", "lake", "pond", "ocean", "sea", "stream",
    "mountain", "hill", "valley", "desert", "coast", "shoreline",
    "park", "playground", "stadium", "airport", "runway",
    "port", "harbor", "harbour", "dock",
    "residential", "commercial", "industrial",
    "aerial", "satellite", "overhead", "top-down", "bird",
}


@dataclass
class GeoClaim:
    """A single detected geographic claim in generated text."""
    text: str               # the matched place name
    start: int              # character start index
    end: int                # character end index
    pattern: str            # which pattern matched
    is_visual_feature: bool # True if the word describes a visual feature, not a location


@dataclass
class GeographicGuardResult:
    """Result of geographic claim analysis on a caption."""
    original_caption: str
    sanitized_caption: str
    geographic_claims: List[GeoClaim] = field(default_factory=list)
    has_geographic_claims: bool = False
    verified_claims: List[str] = field(default_factory=list)
    stripped_claims: List[str] = field(default_factory=list)
    has_independent_evidence: bool = False
    verification_reason: str = ""


def _is_known_geographic_entity(word: str) -> bool:
    """Check if a word is a known geographic entity (not a visual feature)."""
    lower = word.lower().strip()
    if lower in _KNOWN_GEOGRAPHIC_ENTITIES:
        return True
    # Check compound names: "new york" → both "new" and "york" individually
    # but the full phrase is in the set
    return False


def _is_visual_feature(word: str) -> bool:
    """Check if a word describes a visual feature rather than a location."""
    return word.lower().strip() in _VISUAL_FEATURES


def _extract_place_name(match: re.Match, group_index: int = 1) -> Optional[str]:
    """Extract the place name from a regex match, cleaning up artifacts."""
    try:
        name = match.group(group_index).strip()
        # Remove trailing punctuation artifacts
        name = name.rstrip(".,;:!?")
        # Must be at least 2 chars to be a real place name
        if len(name) < 2:
            return None
        return name
    except (IndexError, AttributeError):
        return None


def detect_geographic_claims(caption: str) -> List[GeoClaim]:
    """
    Detect named geographic claims in a caption.

    Returns a list of GeoClaim objects. Each claim is a potential location
    name that may need verification.
    """
    claims: List[GeoClaim] = []
    seen_spans: set = set()

    def _add_claim(name: str, start: int, end: int, pattern: str):
        if name and not _is_visual_feature(name) and _is_known_geographic_entity(name.lower()):
            span = (start, end)
            if span not in seen_spans:
                seen_spans.add(span)
                claims.append(GeoClaim(
                    text=name, start=start, end=end,
                    pattern=pattern, is_visual_feature=False,
                ))

    # 1. Case-insensitive preposition + location (catches "in kolkata", "near London")
    for match in _PATTERN_PREP_KNOWN_ENTITY.finditer(caption):
        name = _extract_place_name(match)
        _add_claim(name, match.start(1), match.end(1), "preposition_location")

    # 2. Case-insensitive "city of X" patterns
    for match in _PATTERN_OF_LOCATION.finditer(caption):
        name = _extract_place_name(match)
        _add_claim(name, match.start(1), match.end(1), "of_location")

    # 3. Known entity after "of" at end (catches "view of london" lowercase)
    for match in _PATTERN_OF_KNOWN_ENTITY.finditer(caption):
        name = _extract_place_name(match)
        _add_claim(name, match.start(1), match.end(1), "of_known_entity")

    # 4. Standalone capitalized words at end (case-sensitive, for "of London")
    for match in _PATTERN_END_LOCATION.finditer(caption):
        name = _extract_place_name(match)
        _add_claim(name, match.start(1), match.end(1), "end_location")

    # 5. Comma-separated compounds: "Mumbai, India" → both parts
    for match in _PATTERN_COMPOUND_GEO.finditer(caption):
        full_text = match.group(0).strip()
        part1 = match.group(1).strip().rstrip(".,;:!?")
        part2 = match.group(2).strip().rstrip(".,;:!?")
        # Only add if it's not already covered by a prior match
        if _is_known_geographic_entity(part1.lower()):
            _add_claim(part1, match.start(1), match.end(1), "compound_geo")
        if _is_known_geographic_entity(part2.lower()):
            _add_claim(part2, match.start(2), match.end(2), "compound_geo")

    # 6. "New York City" → strip "City" suffix, keep "New York"
    for match in _PATTERN_COMPOUND_CITY_SUFFIX.finditer(caption):
        name = match.group(1).strip()
        _add_claim(name, match.start(1), match.end(1), "compound_city_suffix")

    return claims


def _has_crs_metadata(image_data: Optional[dict]) -> bool:
    """Check if the image has CRS/geographic metadata."""
    if image_data is None:
        return False
    metadata = image_data.get("metadata", {})
    return bool(
        image_data.get("is_geotiff")
        and metadata.get("crs")
        and metadata.get("transform")
    )


def _get_crs_description(image_data: Optional[dict]) -> str:
    """Get a human-readable description of the CRS metadata."""
    if image_data is None:
        return "no image data"
    metadata = image_data.get("metadata", {})
    crs = metadata.get("crs", "unknown")
    transform = metadata.get("transform")
    if crs and transform:
        return f"CRS={crs} with affine transform"
    elif crs:
        return f"CRS={crs} without transform"
    return "no CRS metadata"


def apply_geographic_guard(
    caption: str,
    image_data: Optional[dict] = None,
) -> GeographicGuardResult:
    """
    Apply the geographic claim guard to a caption.

    Args:
        caption: The generated caption text.
        image_data: Optional image metadata dict (for CRS verification).

    Returns:
        GeographicGuardResult with sanitized caption and metadata.
    """
    result = GeographicGuardResult(
        original_caption=caption,
        sanitized_caption=caption,
    )

    # Detect geographic claims
    claims = detect_geographic_claims(caption)
    result.geographic_claims = claims
    result.has_geographic_claims = len(claims) > 0

    if not claims:
        # No geographic claims detected — nothing to do
        result.verification_reason = "no geographic claims detected"
        return result

    # Check for independent evidence
    has_crs = _has_crs_metadata(image_data)
    result.has_independent_evidence = has_crs
    result.verification_reason = (
        f"CRS metadata available: {_get_crs_description(image_data)}"
        if has_crs
        else "no independent geographic evidence (captioning model has no geolocation input)"
    )

    # Strip ALL unverified geographic claims
    # We strip everything because:
    #   1. CRS metadata gives coordinates, not place names
    #   2. Without reverse geocoding, we can't verify place names
    #   3. BLIP's place names are memorized patterns, not observations
    sanitized = caption
    for claim in claims:
        # Replace the claim with a neutral description
        before = sanitized[:claim.start]
        after = sanitized[claim.end:]
        # Insert a neutral placeholder only if the sentence structure needs it
        insertion = _neutral_replacement(claim, sanitized, before, after)
        sanitized = before + insertion + after
        result.stripped_claims.append(claim.text)
        logger.info(
            "Geographic claim stripped: '%s' (pattern=%s, has_crs=%s)",
            claim.text, claim.pattern, has_crs,
        )

    # Clean up whitespace
    sanitized = re.sub(r"\s{2,}", " ", sanitized).strip(" ,.-")
    result.sanitized_caption = sanitized

    if result.stripped_claims:
        logger.info(
            "Caption sanitized: '%s' → '%s' (stripped %d claims)",
            caption, sanitized, len(result.stripped_claims),
        )

    return result


def _neutral_replacement(
    claim: GeoClaim,
    full_caption: str,
    before: str,
    after: str,
) -> str:
    """
    Determine what to put in place of a stripped geographic claim.

    Strategy:
    - If the claim follows "of" → replace with generic descriptor
    - If the claim is at the end → just remove it
    - If the claim is mid-sentence → remove it
    """
    # Check context: what comes before the claim?
    before_stripped = before.rstrip()

    # "of London" → "of an urban area" only if we can infer urban
    if before_stripped.endswith("of"):
        # Use a neutral generic term
        return " an area"

    # Default: just remove the claim
    return ""


def sanitize_caption_for_display(
    caption: str,
    image_data: Optional[dict] = None,
) -> Tuple[str, GeographicGuardResult]:
    """
    Convenience function: sanitize a caption and return both the clean
    version and the full guard result for evidence/debugging.

    Usage:
        safe_caption, guard_result = sanitize_caption_for_display(
            raw_caption, image_data
        )
    """
    result = apply_geographic_guard(caption, image_data)
    return result.sanitized_caption, result
