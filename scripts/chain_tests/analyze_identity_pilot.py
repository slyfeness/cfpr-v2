"""Frozen analysis helpers for the exploratory identity-injection pilot."""
from __future__ import annotations

import math


CRITERIA = {
    "donor_rise": 0.15,
    "own_drop": 0.30,
    "sharpness_ratio": 0.80,
    "outside_own_abs_delta": 0.02,
}


def _finite(value):
    return value is not None and math.isfinite(float(value))


def is_candidate(per_person):
    """Apply all four pre-registered criteria to both pilot people."""
    for row in per_person.values():
        values = (row.get("donor_rise"), row.get("own_drop"), row.get("sharpness_ratio"), row.get("outside_own_delta"))
        if not all(_finite(value) for value in values):
            return False
        if float(row["donor_rise"]) < CRITERIA["donor_rise"]:
            return False
        if float(row["own_drop"]) < CRITERIA["own_drop"]:
            return False
        if float(row["sharpness_ratio"]) < CRITERIA["sharpness_ratio"]:
            return False
        if abs(float(row["outside_own_delta"])) > CRITERIA["outside_own_abs_delta"]:
            return False
    return True
