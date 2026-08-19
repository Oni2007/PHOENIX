"""Put the compiled extension on sys.path.

The build tree lives outside the repository (see CLAUDE.md): the working tree is on
a cloud-synced Windows drive, and Linux build artefacts have no business being
uploaded to it. PHOENIX_BUILD_DIR points at wherever it actually is; the in-tree
`build/` remains the default for anyone building conventionally.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_build = Path(os.environ.get("PHOENIX_BUILD_DIR", Path(__file__).resolve().parents[1] / "build"))
if _build.is_dir() and str(_build) not in sys.path:
    sys.path.insert(0, str(_build))
