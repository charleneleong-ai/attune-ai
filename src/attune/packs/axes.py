from __future__ import annotations

from enum import StrEnum


class Axis(StrEnum):
    # Signal axes are a pack-authoring vocabulary. The engine treats an axis as an opaque
    # string label, so a new condition adds an axis here without touching engine code.
    METABOLIC = "metabolic"
    CYCLE = "cycle"
    DERMATOLOGICAL = "dermatological"
    PSYCHOLOGICAL = "psychological"
    PHYSIOLOGICAL = "physiological"
    BEHAVIORAL = "behavioral"
    PAIN = "pain"
    COGNITIVE = "cognitive"
