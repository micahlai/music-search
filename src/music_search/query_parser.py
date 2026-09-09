"""Future structured natural-language query parsing boundary."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ParsedQuery:
    """Target contract for a later LLM-backed query parser."""

    semantic_text: str
    moods: tuple[str, ...] = field(default_factory=tuple)
    instruments_required: tuple[str, ...] = field(default_factory=tuple)
    instruments_excluded: tuple[str, ...] = field(default_factory=tuple)
    vocals: str | None = None
    event_type: str | None = None
    event_time_range_seconds: tuple[float, float] | None = None


def parse_query(prompt: str) -> ParsedQuery:
    """Return the V1 pass-through query until structured LLM parsing is implemented.

    TODO: Add a schema-constrained LLM parser, validation, and deterministic fallbacks.
    """

    normalized = prompt.strip()
    if not normalized:
        raise ValueError("prompt cannot be empty")
    return ParsedQuery(semantic_text=normalized)
