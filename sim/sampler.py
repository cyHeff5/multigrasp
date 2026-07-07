from __future__ import annotations

import math
import numpy as np


def sample_object_spec(sampler_cfg: dict, rng: np.random.Generator) -> dict:
    # Zieht eine zufällige Form, sampelt Dimensionen und Reibung gleichverteilt
    # (Feix et al. 2014). Masse wird dichtebasiert aus dem Volumen berechnet.
    shape = str(rng.choice(sampler_cfg["shapes"]))
    dims  = sampler_cfg[shape]

    spec: dict = {
        "shape":            shape,
        "lateral_friction": _uniform(sampler_cfg["lateral_friction"], rng),
        "yaw_rad":          float(rng.uniform(0.0, 2.0 * math.pi)),  # zufällige Rotation um Z
    }
    for key, bounds in dims.items():
        spec[key] = _uniform(bounds, rng)

    if shape == "rect_cylinder":
        # thickness ist per Konvention immer die kürzere Seite, width die längere.
        t, w = spec["thickness_cm"], spec["width_cm"]
        spec["thickness_cm"] = min(t, w)
        spec["width_cm"]     = max(t, w)

    spec["mass_kg"] = sample_mass_kg(spec, sampler_cfg, rng)
    return spec


def object_volume_cm3(spec: dict) -> float:
    # Volumen der Primitivform in cm³ (Dimensionen liegen in cm vor).
    shape = spec["shape"]
    if shape == "sphere":
        r = spec["size_cm"] / 2.0
        return 4.0 / 3.0 * math.pi * r ** 3
    if shape == "cube":
        return spec["size_cm"] ** 3
    if shape == "cylinder":
        r = spec["thickness_cm"] / 2.0
        return math.pi * r ** 2 * spec["height_cm"]
    if shape == "rect_cylinder":
        return spec["thickness_cm"] * spec["width_cm"] * spec["height_cm"]
    # URDF/unbekannt: grobe Würfel-Schätzung aus size_cm (AABB-Kantenlänge).
    return float(spec.get("size_cm", 5.0)) ** 3


def sample_mass_kg(spec: dict, sampler_cfg: dict, rng: np.random.Generator) -> float:
    # Realistische Masse = Dichte × Volumen. Verhindert absurde Dichten, die eine feste
    # kg-Range bei kleinen/dünnen Objekten erzeugt (dünner Zylinder -> 37 g/cm³).
    # Fallback auf feste kg-Range, falls density_g_cm3 nicht in der Config steht.
    if "density_g_cm3" in sampler_cfg:
        density = _uniform(sampler_cfg["density_g_cm3"], rng)     # g/cm³
        return density * object_volume_cm3(spec) / 1000.0         # g -> kg
    return _uniform(sampler_cfg["mass_kg"], rng)


def _uniform(bounds: dict, rng: np.random.Generator) -> float:
    return float(rng.uniform(bounds["min"], bounds["max"]))
