import json
import math
import os
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, Optional


# =========================================================
# HRTM DEPOSITION UTILITIES
# =========================================================
# This module provides a structured subject-aware respiratory
# deposition model using:
#   - subject anatomy and breathing data from
#     hrtm_subjects_breathing.json
#   - scenario-based Fn values from the same JSON
#   - simplified, region-by-region deposition logic
#
# IMPORTANT
# ---------
# This is an ICRP66-inspired implementation, not a verbatim
# reproduction of every mechanistic equation in ICRP 66.
#
# Design principles:
#   1) Use the JSON file as the source of truth for subject data.
#   2) Do not fabricate values that are not present in the JSON.
#   3) Keep particle parameters separate from subject parameters.
#   4) Return all regional fractions consistently:
#        ET1, ET2, BB, bb, AI, EXH
#
# The formulas below are intentionally modular so they can later
# be replaced or refined without touching the rest of the code.
# =========================================================


# =========================================================
# DEFAULT JSON PATH
# =========================================================
def get_default_json_path() -> Path:
    """
    Robust JSON path resolution strategy.

    Priority:
    1. Environment variable HRTM_DB_PATH
    2. Packaged resource:
       dgm_hrtm.data/hrtm_subjects_breathing.json
    """
    env_path = os.getenv("HRTM_DB_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        raise FileNotFoundError(
            f"HRTM_DB_PATH is set but file does not exist: {env_path}"
        )

    return Path(files("dgm_hrtm.data").joinpath("hrtm_subjects_breathing.json"))


# =========================================================
# JSON LOADER
# =========================================================
def load_breathing_database(json_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the HRTM subject breathing database from JSON.

    Parameters
    ----------
    json_path : str or None
        Optional custom path. If None, the default path resolution
        strategy is used.

    Returns
    -------
    dict
        Parsed JSON database.

    Raises
    ------
    FileNotFoundError
        If the JSON file cannot be found.
    """
    path = Path(json_path) if json_path is not None else get_default_json_path()

    if not path.exists():
        raise FileNotFoundError(
            f"Breathing database not found: {path}\n"
            "Expected file: hrtm_subjects_breathing.json"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# LABEL NORMALIZATION
# =========================================================
def normalize_subject_label(age_group: str) -> str:
    age = str(age_group).strip().lower()

    mapping = {
        "adult": "adult",
        "15yo": "age_15y",
        "15y": "age_15y",
        "age_15y": "age_15y",
        "10yo": "age_10y",
        "10y": "age_10y",
        "age_10y": "age_10y",
        "5yo": "age_5y",
        "5y": "age_5y",
        "age_5y": "age_5y",
        "1yo": "age_1y",
        "1y": "age_1y",
        "age_1y": "age_1y",
        "infant": "infant",
        "3mo": "infant",
        "3m": "infant",
    }

    if age not in mapping:
        raise ValueError(f"Unknown age_group: {age_group}")

    return mapping[age]


def normalize_activity_label(activity_level: str) -> str:
    """
    Strict activity mapping aligned with the JSON table.
    No reinterpretation is performed.
    """
    activity = str(activity_level).strip().lower()

    allowed = {
        "sleep",
        "sitting",
        "light_exercise",
        "heavy_exercise",
    }

    if activity not in allowed:
        raise ValueError(
            f"Invalid activity_level '{activity_level}'. "
            "Allowed values: sleep, sitting, light_exercise, heavy_exercise"
        )

    return activity


def normalize_breathing_mode(breathing_mode: str) -> str:
    mode = str(breathing_mode).strip().lower()

    mapping = {
        "nasal": "nasal",
        "normal": "nasal",
        "nose": "nasal",
        "mouth": "mouth",
        "oral": "mouth",
        "mouth_breather": "mouth",
        "mouth-breather": "mouth",
    }

    if mode not in mapping:
        raise ValueError(f"Unknown breathing_mode: {breathing_mode}")

    return mapping[mode]


# =========================================================
# DATABASE RESOLUTION
# =========================================================
def get_subject_record(
    db: Dict[str, Any],
    age_group: str,
    gender: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return the JSON block for the requested subject.

    Notes
    -----
    - adult and age_15y require gender
    - younger groups use the 'default' block
    """
    subject_key = normalize_subject_label(age_group)
    block = db["subjects"][subject_key]

    if subject_key in ("adult", "age_15y"):
        if gender is None:
            raise ValueError(
                f"Gender is required for subject '{subject_key}'."
            )

        gender_key = str(gender).strip().lower()
        if gender_key not in block:
            raise ValueError(
                f"Invalid gender '{gender}' for subject '{subject_key}'."
            )
        return block[gender_key]

    return block["default"]


def get_scenario_fn(
    db: Dict[str, Any],
    activity_level: str,
    breathing_mode: str = "nasal",
) -> Dict[str, Any]:
    """
    Return scenario key and corresponding Fn.

    Fn is taken directly from the JSON file and never invented.
    """
    scenario_key = normalize_activity_label(activity_level)
    mode_key = normalize_breathing_mode(breathing_mode)

    scenario_block = db["scenarios"][scenario_key]

    fn_key = "Fn_normal" if mode_key == "nasal" else "Fn_mouth"
    return {
        "scenario_key": scenario_key,
        "Fn": float(scenario_block[fn_key]),
    }


def get_subject_parameters(
    age_group: str,
    activity_level: str,
    gender: Optional[str] = None,
    breathing_mode: str = "nasal",
    json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return subject anatomy, breathing block and Fn.

    Missing subject/scenario values stored as null in the JSON are
    treated as unavailable and raise a clear error.
    """
    db = load_breathing_database(json_path=json_path)

    subject = get_subject_record(db, age_group=age_group, gender=gender)
    scenario = get_scenario_fn(
        db,
        activity_level=activity_level,
        breathing_mode=breathing_mode,
    )

    scenario_key = scenario["scenario_key"]
    breathing = subject["breathing"].get(scenario_key)

    if breathing is None:
        raise ValueError(
            f"No breathing block found for scenario '{scenario_key}'."
        )

    if breathing["B_m3_h"] is None:
        subject_key = normalize_subject_label(age_group)
        raise ValueError(
            f"Scenario '{scenario_key}' is not available for subject "
            f"'{subject_key}'"
            + (f" and gender '{gender}'" if gender else "")
            + " in hrtm_subjects_breathing.json."
        )

    return {
        "Fn": float(scenario["Fn"]),
        "scenario": scenario_key,
        "breathing_mode": normalize_breathing_mode(breathing_mode),
        "anatomy": subject["anatomy"],
        "breathing": breathing,
    }


# =========================================================
# PARTICLE HELPERS
# =========================================================
def parse_amad_um(amad: Any) -> float:
    """
    Convert AMAD input into a numeric value in micrometres.

    Accepted examples:
        "1um", "5um", 1, 5, 1.0, 5.0
    """
    if isinstance(amad, (int, float)):
        amad_um = float(amad)
    else:
        s = str(amad).strip().lower().replace("μ", "u").replace("µ", "u")
        s = s.replace("micron", "um").replace("microns", "um")
        if s.endswith("um"):
            s = s[:-2]
        amad_um = float(s)

    if amad_um <= 0.0:
        raise ValueError("AMAD must be positive.")

    return amad_um


def compute_aerodynamic_modifier(
    amad_um: float,
    particle_density: float = 1.0,
    particle_shape_factor: float = 1.0,
    reference_density: float = 1.0,
) -> float:
    """
    Return a simple aerodynamic scaling factor.

    This is intentionally compact:
        d_ae ~ d_p * sqrt(rho_p * chi / rho_ref)

    In this implementation, AMAD is already aerodynamic by meaning,
    but we retain a soft modifier to allow future extension without
    changing the function API.
    """
    if particle_density <= 0.0:
        raise ValueError("particle_density must be positive.")
    if particle_shape_factor <= 0.0:
        raise ValueError("particle_shape_factor must be positive.")
    if reference_density <= 0.0:
        raise ValueError("reference_density must be positive.")

    return math.sqrt((particle_density * particle_shape_factor) / reference_density)


# =========================================================
# BREATHING PARAMETERS
# =========================================================
def get_breathing_params(
    age_group: str,
    activity_level: str,
    gender: Optional[str] = None,
    breathing_mode: str = "nasal",
    json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return breathing parameters directly from JSON.
    """
    params = get_subject_parameters(
        age_group=age_group,
        activity_level=activity_level,
        gender=gender,
        breathing_mode=breathing_mode,
        json_path=json_path,
    )

    breathing = params["breathing"]

    return {
        "breathing_mode": params["breathing_mode"],
        "Fn": float(params["Fn"]),
        "scenario": params["scenario"],
        "B_m3_h": float(breathing["B_m3_h"]),
        "fR_min": float(breathing["fR_min"]),
        "VT_ml": float(breathing["VT_ml"]),
        "V_ml_s": float(breathing["V_ml_s"]),
    }


# =========================================================
# REGION-SPLIT BUILDING BLOCKS
# =========================================================
def _compute_et_base_fraction(
    Fn: float,
    anatomy: Dict[str, Any],
    breathing: Dict[str, Any],
    amad_um: float,
    aero_modifier: float,
) -> float:
    """
    Compute the total ET deposition fraction before ET1/ET2 split.

    Heuristic design:
    - increases with Fn (more nasal flow -> stronger ET)
    - increases with vd_et / VT
    - increases with larger aerodynamic size
    - weakly decreases with stronger flow velocity
    """
    Fn = max(0.0, min(1.0, float(Fn)))

    vd_et = float(anatomy["vd_et_ml"])
    vt = float(breathing["VT_ml"])
    flow = float(breathing["V_ml_s"])

    vd_ratio = vd_et / max(vt, 1e-12)
    size_term = (amad_um * aero_modifier) / (1.0 + amad_um * aero_modifier)
    flow_term = 1.0 / (1.0 + flow / 2500.0)

    et_base = (0.18 + 0.32 * Fn) * (0.65 + 2.4 * vd_ratio) * (0.55 + 0.90 * size_term) * flow_term

    return max(0.0, min(et_base, 0.95))


def _split_et1_et2(
    et_total: float,
    Fn: float,
    anatomy: Dict[str, Any],
    amad_um: float,
) -> Dict[str, float]:
    """
    Split ET total deposition into ET1 and ET2.

    ET1 is favored by:
    - stronger nasal breathing
    - larger particle size
    - relatively larger proximal diameter d0
    """
    d0 = float(anatomy["d0_cm"])

    size_term = amad_um / (amad_um + 1.0)
    et1_ratio = 0.25 + 0.30 * Fn + 0.12 * size_term + 0.03 * (d0 / 1.65)
    et1_ratio = max(0.10, min(et1_ratio, 0.85))

    ET1 = et_total * et1_ratio
    ET2 = max(0.0, et_total - ET1)

    return {"ET1": ET1, "ET2": ET2}


def _compute_remaining_deposition_fraction(
    anatomy: Dict[str, Any],
    breathing: Dict[str, Any],
    amad_um: float,
    aero_modifier: float,
    et_total: float,
) -> float:
    """
    Compute total deposition fraction available beyond ET.

    This represents deposition reaching thoracic airways + AI,
    not yet split regionally.
    """
    vd_total = float(anatomy["vd_total_ml"])
    frc = float(anatomy["frc_ml"])
    vt = float(breathing["VT_ml"])
    flow = float(breathing["V_ml_s"])

    size_term = (amad_um * aero_modifier) / (1.0 + amad_um * aero_modifier)
    exchange_term = vt / max(vt + frc, 1e-12)
    dead_space_term = vd_total / max(vt, 1e-12)
    flow_term = 1.0 / (1.0 + flow / 3000.0)

    thor_total = (0.10 + 0.40 * exchange_term + 0.20 * dead_space_term) * (0.65 + 0.90 * size_term) * flow_term

    thor_total = max(0.0, min(thor_total, 0.95))
    remaining = min(thor_total, max(0.0, 0.98 - et_total))

    return remaining


def _split_thoracic_regions(
    anatomy: Dict[str, Any],
    breathing: Dict[str, Any],
    amad_um: float,
    aero_modifier: float,
    thoracic_total: float,
    use_regional_sf: bool = False,
) -> Dict[str, float]:
    """
    Split thoracic deposition into BB, bb and AI.

    By default, sf_t / sf_b / sf_A are ignored because they are not
    explicit particle-shape factors and are not written directly in
    ICRP66 equations.

    If use_regional_sf=True, they are used only as optional regional
    weighting modifiers.
    """
    vd_bb = float(anatomy["vd_bb_ml"])
    vd_bb_small = float(anatomy["vd_bb_small_ml"])
    frc = float(anatomy["frc_ml"])

    d9 = float(anatomy["d9_cm"])
    d16 = float(anatomy["d16_cm"])
    flow = float(breathing["V_ml_s"])

    size_term = (amad_um * aero_modifier) / (1.0 + amad_um * aero_modifier)
    diffusive_term = 1.0 / (1.0 + amad_um * aero_modifier)
    flow_term = 1.0 / (1.0 + flow / 2500.0)

    # Larger particles favor BB/bb. Smaller particles favor AI.
    w_BB = vd_bb * (0.90 + 0.90 * size_term) * (1.0 + 0.30 * d9) * flow_term
    w_bb = vd_bb_small * (0.85 + 0.75 * size_term) * (1.0 + 0.20 * d16) * flow_term
    w_AI = frc * (0.55 + 1.60 * diffusive_term)

    if use_regional_sf:
        w_BB *= float(anatomy["sf_t"])
        w_bb *= float(anatomy["sf_b"])
        w_AI *= float(anatomy["sf_A"])

    total_w = w_BB + w_bb + w_AI
    if total_w <= 0.0:
        raise ValueError("Thoracic regional weights are non-positive.")

    return {
        "BB": thoracic_total * (w_BB / total_w),
        "bb": thoracic_total * (w_bb / total_w),
        "AI": thoracic_total * (w_AI / total_w),
    }


# =========================================================
# MAIN REGIONAL FRACTIONS
# =========================================================
def normalize_regional_fractions(fractions: Dict[str, float]) -> Dict[str, float]:
    """
    Ensure ET1 + ET2 + BB + bb + AI + EXH = 1.

    If deposited total exceeds 1, fractions are rescaled and EXH=0.
    """
    out = {k: float(v) for k, v in fractions.items()}
    keys = ["ET1", "ET2", "BB", "bb", "AI"]

    deposited_total = sum(out.get(k, 0.0) for k in keys)

    if deposited_total < 0.0:
        deposited_total = 0.0

    if deposited_total > 1.0:
        for k in keys:
            out[k] = out[k] / deposited_total
        out["EXH"] = 0.0
    else:
        out["EXH"] = 1.0 - deposited_total

    return out


def get_regional_deposition_fractions(
    age_group: str,
    activity_level: str,
    amad: Any = "5um",
    gender: Optional[str] = None,
    breathing_mode: str = "nasal",
    particle_density: float = 1.0,
    particle_shape_factor: float = 1.0,
    use_regional_sf: bool = False,
    json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main HRTM v2-style regional deposition function.

    Parameters
    ----------
    age_group : str
        Subject age group.
    activity_level : str
        Scenario/activity level.
    amad : str or float
        AMAD in micrometres or string like "1um", "5um".
    gender : str or None
        Required for adult and age_15y.
    breathing_mode : str
        "nasal" or "mouth".
    particle_density : float
        Particle density factor, external to the subject JSON.
    particle_shape_factor : float
        Particle shape factor, external to the subject JSON.
    use_regional_sf : bool
        If True, use anatomy sf_t/sf_b/sf_A as optional regional
        weighting modifiers. Default is False.
    json_path : str or None
        Optional custom JSON path.

    Returns
    -------
    dict
        Fractions, metadata and the subject parameters used.
    """
    params = get_subject_parameters(
        age_group=age_group,
        activity_level=activity_level,
        gender=gender,
        breathing_mode=breathing_mode,
        json_path=json_path,
    )

    anatomy = params["anatomy"]
    breathing = params["breathing"]
    Fn = float(params["Fn"])

    amad_um = parse_amad_um(amad)
    aero_modifier = compute_aerodynamic_modifier(
        amad_um=amad_um,
        particle_density=particle_density,
        particle_shape_factor=particle_shape_factor,
        reference_density=1.0,
    )

    et_total = _compute_et_base_fraction(
        Fn=Fn,
        anatomy=anatomy,
        breathing=breathing,
        amad_um=amad_um,
        aero_modifier=aero_modifier,
    )

    et_split = _split_et1_et2(
        et_total=et_total,
        Fn=Fn,
        anatomy=anatomy,
        amad_um=amad_um,
    )

    thoracic_total = _compute_remaining_deposition_fraction(
        anatomy=anatomy,
        breathing=breathing,
        amad_um=amad_um,
        aero_modifier=aero_modifier,
        et_total=et_total,
    )

    thoracic_split = _split_thoracic_regions(
        anatomy=anatomy,
        breathing=breathing,
        amad_um=amad_um,
        aero_modifier=aero_modifier,
        thoracic_total=thoracic_total,
        use_regional_sf=use_regional_sf,
    )

    fractions = {
        "ET1": et_split["ET1"],
        "ET2": et_split["ET2"],
        "BB": thoracic_split["BB"],
        "bb": thoracic_split["bb"],
        "AI": thoracic_split["AI"],
    }

    fractions = normalize_regional_fractions(fractions)

    fractions["Fn"] = Fn
    fractions["amad_um"] = float(amad_um)
    fractions["breathing_mode"] = params["breathing_mode"]
    fractions["scenario"] = params["scenario"]
    fractions["particle_density"] = float(particle_density)
    fractions["particle_shape_factor"] = float(particle_shape_factor)
    fractions["aero_modifier"] = float(aero_modifier)
    fractions["use_regional_sf"] = bool(use_regional_sf)
    fractions["anatomy"] = anatomy
    fractions["breathing"] = breathing

    return fractions


# =========================================================
# ACTIVITY CONVERSION
# =========================================================
def compute_regional_deposited_activities(
    inhaled_activity_bq: float,
    fractions: Dict[str, Any],
) -> Dict[str, float]:
    """
    Convert regional fractions into deposited activities [Bq].
    """
    A = float(inhaled_activity_bq)

    return {
        "ET1": A * float(fractions["ET1"]),
        "ET2": A * float(fractions["ET2"]),
        "BB": A * float(fractions["BB"]),
        "bb": A * float(fractions["bb"]),
        "AI": A * float(fractions["AI"]),
        "EXH": A * float(fractions["EXH"]),
    }


# =========================================================
# CONVENIENCE WRAPPER
# =========================================================
def compute_regional_deposition_from_concentration(
    concentration_bq_m3: float,
    exposure_time_h: float,
    age_group: str,
    activity_level: str,
    amad: Any = "5um",
    gender: Optional[str] = None,
    breathing_mode: str = "nasal",
    particle_density: float = 1.0,
    particle_shape_factor: float = 1.0,
    use_regional_sf: bool = False,
    json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full helper:
        concentration -> inhaled activity -> regional deposition

    Inhaled activity is computed with B from the JSON:
        A_inh = C * B * t
    """
    concentration_bq_m3 = float(concentration_bq_m3)
    exposure_time_h = float(exposure_time_h)

    breathing_params = get_breathing_params(
        age_group=age_group,
        activity_level=activity_level,
        gender=gender,
        breathing_mode=breathing_mode,
        json_path=json_path,
    )

    inhaled_activity_bq = (
        concentration_bq_m3
        * float(breathing_params["B_m3_h"])
        * exposure_time_h
    )

    fractions = get_regional_deposition_fractions(
        age_group=age_group,
        activity_level=activity_level,
        amad=amad,
        gender=gender,
        breathing_mode=breathing_mode,
        particle_density=particle_density,
        particle_shape_factor=particle_shape_factor,
        use_regional_sf=use_regional_sf,
        json_path=json_path,
    )

    deposited = compute_regional_deposited_activities(
        inhaled_activity_bq=inhaled_activity_bq,
        fractions=fractions,
    )

    return {
        "inhaled_activity_bq": inhaled_activity_bq,
        "fractions": fractions,
        "deposited_activities_bq": deposited,
    }