"""Higher-level analytics over scored segments (tone timeline, section breaks)."""

from swissfin.analysis.tone_timeline import build_tone_timeline, detect_section_break

__all__ = ["build_tone_timeline", "detect_section_break"]
