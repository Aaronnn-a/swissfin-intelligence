"""SwissFin Intelligence — privacy-first AI for Swiss financial documents.

Phase 1 covers earnings-call intelligence: download, transcribe (Whisper), and
score sentiment of Swiss earnings calls fully locally.
"""

from swissfin.config import settings

__all__ = ["settings", "__version__"]
__version__ = "0.1.0"
