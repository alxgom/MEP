from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(eq=False)
class EnvView:
    """Lightweight graph view used by the interactive routing algorithms."""

    nodes: Any
    adj: dict[int, list[tuple[int, float, Any]]]
