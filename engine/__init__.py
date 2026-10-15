"""Decision engine. `A(...)` reads values from engine/assumptions.yaml."""
from functools import lru_cache
from pathlib import Path

import yaml

ASSUMPTIONS_PATH = Path(__file__).with_name("assumptions.yaml")


@lru_cache(maxsize=1)
def load_assumptions():
    return yaml.safe_load(ASSUMPTIONS_PATH.read_text(encoding="utf-8"))


def A(section, key):
    """Value of an assumption (unwraps {value: ..., source: ...})."""
    v = load_assumptions()[section][key]
    return v["value"] if isinstance(v, dict) and "value" in v else v


def confirm_items():
    """All assumptions flagged `confirm: true` (pending chemist confirmation)."""
    out = []
    for sec, items in load_assumptions().items():
        if not isinstance(items, dict):
            continue
        for k, v in items.items():
            if isinstance(v, dict) and v.get("confirm"):
                out.append({"item": f"{sec}.{k}", "value": v.get("value"), "source": v.get("source")})
    return out
