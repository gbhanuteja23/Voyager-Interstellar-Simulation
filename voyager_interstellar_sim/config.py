"""Central astronomical constants, piecewise logarithmic scaling math, planetary
ephemeris tables, and mission timeline definitions for Voyager 1 & Voyager 2.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Filesystem & Omniverse Kit Paths
# ---------------------------------------------------------------------------
PACKAGE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = PACKAGE_DIR.parent
ASSETS_RAW_DIR: Path = PACKAGE_DIR / "assets_raw"
ASSETS_USD_DIR: Path = PACKAGE_DIR / "assets_usd"
TEXTURES_DIR: Path = ASSETS_USD_DIR / "textures"


def resolve_isaac_sim_root() -> Path:
    """Locate the Isaac Sim installation directory across Windows and WSL."""
    env_isaac = os.environ.get("ISAAC_PATH")
    if env_isaac and Path(env_isaac).exists():
        return Path(env_isaac).resolve()
    win_path = Path("D:/0-NVIDIA-Omniverse/isaac-sim")
    if win_path.exists():
        return win_path
    wsl_path = Path("/mnt/d/0-NVIDIA-Omniverse/isaac-sim")
    if wsl_path.exists():
        return wsl_path
    return win_path


ISAAC_SIM_ROOT: Path = resolve_isaac_sim_root()
KIT_MDL_BASE_DIR: Path = ISAAC_SIM_ROOT / "kit" / "mdl" / "core" / "Base"
SIMREADY_CENTRAL_DIR: Path = (
    Path("D:/0-NVIDIA-Omniverse/1-Isaac-sim-assets/simready-assets-hf/simready-central")
    if os.name == "nt"
    else Path("/mnt/d/0-NVIDIA-Omniverse/1-Isaac-sim-assets/simready-assets-hf/simready-central")
)

# ---------------------------------------------------------------------------
# Physical & Astronomical Constants
# ---------------------------------------------------------------------------
AU_KM: float = 149_597_870.7  # 1 Astronomical Unit in km
AU_BILLION_KM: float = AU_KM / 1.0e9  # ~0.14959787 billion km
SPEED_OF_LIGHT_KM_S: float = 299_792.458
CurrentEpochYear: float = 2026.865  # November 2026

# Incoming Very Local Interstellar Medium (VLISM) wind unit vector (heliocentric ecliptic)
# Nose of the Heliosphere points roughly toward ecliptic lon ~255.4 deg, lat ~+5.2 deg
_NOSE_LON = math.radians(255.4)
_NOSE_LAT = math.radians(5.2)
HELIO_NOSE_DIR: Tuple[float, float, float] = (
    math.cos(_NOSE_LAT) * math.cos(_NOSE_LON),
    math.cos(_NOSE_LAT) * math.sin(_NOSE_LON),
    math.sin(_NOSE_LAT),
)
ISM_INFLOW_DIR: Tuple[float, float, float] = (
    -HELIO_NOSE_DIR[0],
    -HELIO_NOSE_DIR[1],
    -HELIO_NOSE_DIR[2],
)

# ---------------------------------------------------------------------------
# Piecewise Logarithmic Radial Distance Scaling Math (Grand Planetary Proportions)
# ---------------------------------------------------------------------------
# Expanded 3.2x scene horizon so planets are towering globes (Earth=28m diameter,
# Jupiter=72m diameter, Saturn+Rings=140m) while Rocket (~1.55m) & Voyager (~0.45m dish)
# look realistically small against massive planetary horizons — still within ~3,600 m max
# float32 radius (< 0.3 mm precision jitter!).
R0_EARTH_SIM_M: float = 240.0
ALPHA_INNER: float = 2.2
S2_PLANETARY_M: float = 330.0
S3_HELIOSPHERE_M: float = 520.0
S4_OORT_M: float = 270.0

R1_BOUND_M: float = R0_EARTH_SIM_M * math.log(1.0 + ALPHA_INNER * 2.0) / math.log(1.0 + ALPHA_INNER)
R2_BOUND_M: float = R1_BOUND_M + S2_PLANETARY_M * math.log(55.0 / 2.0)
R3_BOUND_M: float = R2_BOUND_M + S3_HELIOSPHERE_M * math.log(250.0 / 55.0)

SUN_VISUAL_RADIUS_M: float = 34.0
SUN_CORONA_INNER_M: float = 41.5
SUN_CORONA_OUTER_M: float = 54.0

# Sleek aerospace scale relative to towering 28m Earth & 72m Jupiter
SPACECRAFT_VISUAL_SCALE: float = 0.12   # 3.7m dish -> 0.44m in scene (Earth is 63x wider!)
ROCKET_VISUAL_SCALE: float = 0.032      # 48.8m Titan IIIE -> 1.56m in scene (Earth is 18x wider!)


def au_to_sim_radius(r_au: float) -> float:
    """Convert true heliocentric radial distance (AU) to simulation scene radius (m)."""
    r = max(0.0, float(r_au))
    if r <= 2.0:
        return R0_EARTH_SIM_M * math.log(1.0 + ALPHA_INNER * r) / math.log(1.0 + ALPHA_INNER)
    if r <= 55.0:
        return R1_BOUND_M + S2_PLANETARY_M * math.log(r / 2.0)
    if r <= 250.0:
        return R2_BOUND_M + S3_HELIOSPHERE_M * math.log(r / 55.0)
    return R3_BOUND_M + S4_OORT_M * math.log(r / 250.0)


def sim_radius_to_au(r_sim: float) -> float:
    """Exact analytical inverse of au_to_sim_radius (scene meters -> true AU)."""
    rs = max(0.0, float(r_sim))
    if rs <= R1_BOUND_M:
        exponent = rs * math.log(1.0 + ALPHA_INNER) / R0_EARTH_SIM_M
        return (math.exp(exponent) - 1.0) / ALPHA_INNER
    if rs <= R2_BOUND_M:
        return 2.0 * math.exp((rs - R1_BOUND_M) / S2_PLANETARY_M)
    if rs <= R3_BOUND_M:
        return 55.0 * math.exp((rs - R2_BOUND_M) / S3_HELIOSPHERE_M)
    return 250.0 * math.exp((rs - R3_BOUND_M) / S4_OORT_M)


def au_vec_to_sim(vec_au: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Map a 3D heliocentric position vector in AU into simulation world meters."""
    x, y, z = vec_au
    r_au = math.sqrt(x * x + y * y + z * z)
    if r_au < 1.0e-12:
        return (0.0, 0.0, 0.0)
    r_sim = au_to_sim_radius(r_au)
    scale = r_sim / r_au
    return (x * scale, y * scale, z * scale)


# ---------------------------------------------------------------------------
# Planetary Physical & Keplerian Orbital Elements (J2000 Ecliptic)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlanetSpec:
    name: str
    a_au: float               # Semi-major axis (AU)
    ecc: float                # Orbital eccentricity
    inc_deg: float            # Inclination to ecliptic (deg)
    raan_deg: float           # Longitude of ascending node (deg)
    arg_peri_deg: float       # Argument of perihelion (deg)
    mean_lon_1977_deg: float  # Calibrated mean longitude at epoch 1977.65
    period_years: float       # Sidereal orbital period (yr)
    radius_km: float          # Physical equatorial radius (km)
    visual_radius_m: float    # Apparent visual radius in scene (m) — towering against spacecraft!
    tilt_deg: float           # Obliquity / axial tilt (deg)
    rot_period_hours: float   # Sidereal rotation period (hours, negative = retrograde)
    base_color: Tuple[float, float, float]
    roughness: float          # Tuned for realistic solar reflection & specular glint
    specular: float           # Specular reflectivity level [0..1]
    texture_key: str
    has_rings: bool = False
    atmo_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    atmo_opacity: float = 0.0


PLANETS: List[PlanetSpec] = [
    PlanetSpec(
        "Mercury", 0.3871, 0.2056, 7.00, 48.33, 29.12, 212.0, 0.240846,
        2439.7, 6.20, 0.03, 1407.6, (0.62, 0.59, 0.56), 0.78, 0.28, "mercury"
    ),
    PlanetSpec(
        "Venus", 0.7233, 0.0068, 3.39, 76.68, 54.88, 95.0, 0.615198,
        6051.8, 12.80, 177.36, -5832.5, (0.94, 0.82, 0.60), 0.42, 0.65, "venus",
        atmo_color=(1.0, 0.88, 0.62), atmo_opacity=0.10
    ),
    PlanetSpec(
        "Earth", 1.0000, 0.0167, 0.00, 0.00, 102.94, 342.0, 1.000017,
        6371.0, 14.00, 23.44, 23.934, (0.25, 0.52, 0.88), 0.24, 0.90, "earth",
        atmo_color=(0.24, 0.62, 1.0), atmo_opacity=0.13
    ),
    PlanetSpec(
        "Mars", 1.5237, 0.0934, 1.85, 49.56, 286.50, 48.0, 1.880848,
        3389.5, 9.20, 25.19, 24.623, (0.84, 0.40, 0.23), 0.72, 0.35, "mars",
        atmo_color=(0.95, 0.62, 0.42), atmo_opacity=0.05
    ),
    PlanetSpec(
        "Jupiter", 5.2038, 0.0489, 1.30, 100.46, 273.87, 81.5, 11.862615,
        69911.0, 36.00, 3.13, 9.925, (0.90, 0.78, 0.62), 0.36, 0.68, "jupiter",
        atmo_color=(0.95, 0.82, 0.65), atmo_opacity=0.06
    ),
    PlanetSpec(
        "Saturn", 9.5826, 0.0565, 2.49, 113.66, 339.39, 148.0, 29.447498,
        58232.0, 30.00, 26.73, 10.656, (0.94, 0.84, 0.64), 0.38, 0.65, "saturn",
        has_rings=True, atmo_color=(0.96, 0.88, 0.68), atmo_opacity=0.06
    ),
    PlanetSpec(
        "Uranus", 19.2184, 0.0463, 0.77, 74.01, 96.99, 218.5, 84.016846,
        25362.0, 19.50, 97.77, -17.24, (0.62, 0.88, 0.92), 0.34, 0.72, "uranus",
        atmo_color=(0.52, 0.92, 0.98), atmo_opacity=0.08
    ),
    PlanetSpec(
        "Neptune", 30.1104, 0.0095, 1.77, 131.78, 272.85, 259.2, 164.79132,
        24622.0, 19.00, 28.32, 16.11, (0.26, 0.48, 0.95), 0.32, 0.78, "neptune",
        atmo_color=(0.30, 0.58, 1.0), atmo_opacity=0.10
    ),
]

PLANET_BY_NAME: Dict[str, PlanetSpec] = {p.name: p for p in PLANETS}

# ---------------------------------------------------------------------------
# Piecewise Mission Timeline Scrubber Mapping (s in [0.0, 1.0] <-> Mission Year)
# ---------------------------------------------------------------------------
TIMELINE_KNOTS: List[Tuple[float, float]] = [
    (0.00, 1977.6790),
    (0.14, 1977.6850),
    (0.26, 1979.1800),
    (0.38, 1980.8600),
    (0.50, 1989.6500),
    (0.65, 2012.6500),
    (0.71, 2018.8500),
    (0.82, 2026.8650),
    (0.92, 2550.0000),
    (1.00, 4800.0000),
]


@dataclass(frozen=True)
class PhasePreset:
    index: int
    label: str
    short_label: str
    s_voyager1: float
    s_voyager2: float
    recommended_camera: str
    description: str


MISSION_PHASES: List[PhasePreset] = [
    PhasePreset(
        1,
        "1. Launch & Deployment (1977)",
        "1. Launch (1977)",
        0.025,
        0.025,
        "RocketStagingCam",
        "Titan IIIE / Centaur ascent off sunlit Earth, SRB & fairing jettison, and Voyager boom unfolding.",
    ),
    PhasePreset(
        2,
        "2. Jupiter Gravity Assist (1979)",
        "2. Jupiter (1979)",
        0.260,
        0.285,
        "VoyagerChaseCam",
        "Close Jovian gravity assist boosting Voyager above Solar System escape velocity.",
    ),
    PhasePreset(
        3,
        "3. Saturn & Outer Planets (1980-89)",
        "3. Saturn/Outer",
        0.380,
        0.500,
        "VoyagerChaseCam",
        "Saturn ring plane & Titan flyby (V1) / Uranus & Neptune Grand Tour (V2).",
    ),
    PhasePreset(
        4,
        "4. Heliopause Crossing (2012/2018)",
        "4. Heliopause",
        0.650,
        0.710,
        "HeliopauseCrossSectionCam",
        "Crossing the Heliopause (~121 AU) from subsonic Heliosheath into Interstellar Space.",
    ),
    PhasePreset(
        5,
        "5. Oort Cloud (~2,000+ AU)",
        "5. Oort Cloud",
        0.935,
        0.935,
        "EclipticOverviewCam",
        "Entering the vast spherical shell of icy Oort Cloud planetesimals.",
    ),
    PhasePreset(
        6,
        "6. Current Interstellar Position (2026)",
        "6. Now (2026)",
        0.820,
        0.820,
        "VoyagerChaseCam",
        "Present-day VLISM cruise (V1 ~166.4 AU North, V2 ~139.5 AU South).",
    ),
]


def scrub_to_year(s: float) -> float:
    """Convert normalized scrubber [0.0, 1.0] to continuous decimal Mission Year."""
    u = max(0.0, min(1.0, float(s)))
    for i in range(len(TIMELINE_KNOTS) - 1):
        s0, y0 = TIMELINE_KNOTS[i]
        s1, y1 = TIMELINE_KNOTS[i + 1]
        if u <= s1 or i == len(TIMELINE_KNOTS) - 2:
            t = 0.0 if (s1 - s0) < 1e-9 else (u - s0) / (s1 - s0)
            return y0 + t * (y1 - y0)
    return TIMELINE_KNOTS[-1][1]


def year_to_scrub(year: float) -> float:
    """Convert decimal Mission Year back to normalized scrubber [0.0, 1.0]."""
    y = float(year)
    if y <= TIMELINE_KNOTS[0][1]:
        return 0.0
    for i in range(len(TIMELINE_KNOTS) - 1):
        s0, y0 = TIMELINE_KNOTS[i]
        s1, y1 = TIMELINE_KNOTS[i + 1]
        if y <= y1:
            t = 0.0 if (y1 - y0) < 1e-9 else (y - y0) / (y1 - y0)
            return s0 + t * (s1 - s0)
    return 1.0


CAMERA_MODES: List[Tuple[str, str]] = [
    ("VoyagerChaseCam", "Voyager Chase Cam"),
    ("RocketStagingCam", "Rocket Staging Cam"),
    ("EclipticOverviewCam", "Solar System Ecliptic Overview"),
    ("HeliopauseCrossSectionCam", "Heliopause Cross-Section Cam"),
    ("FreeOrbitCam", "Free Orbit"),
]
