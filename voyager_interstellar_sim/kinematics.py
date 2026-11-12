"""Orbital Mechanics, Catmull-Rom Grand Tour Splines, Launch Staging, and Live Telemetry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from voyager_interstellar_sim.config import (
    AU_BILLION_KM,
    AU_KM,
    HELIO_NOSE_DIR,
    PLANETS,
    PLANET_BY_NAME,
    PlanetSpec,
    SPEED_OF_LIGHT_KM_S,
    au_to_sim_radius,
    au_vec_to_sim,
    scrub_to_year,
    sim_radius_to_au,
)


# ---------------------------------------------------------------------------
# Keplerian Propagator
# ---------------------------------------------------------------------------
def solve_kepler(mean_anomaly_rad: float, ecc: float, iterations: int = 7) -> float:
    """Solve E - e*sin(E) = M via Newton-Raphson."""
    m = math.fmod(mean_anomaly_rad, 2.0 * math.pi)
    if m < -math.pi:
        m += 2.0 * math.pi
    elif m > math.pi:
        m -= 2.0 * math.pi
    e_anom = m + (ecc * math.sin(m) if ecc < 0.8 else 0.0)
    for _ in range(iterations):
        f_val = e_anom - ecc * math.sin(e_anom) - m
        f_der = 1.0 - ecc * math.cos(e_anom)
        e_anom -= f_val / max(f_der, 1e-8)
    return e_anom


def planet_position_au(spec: PlanetSpec, year: float) -> Tuple[float, float, float]:
    """Compute 3D heliocentric J2000 ecliptic coordinates (in AU) at decimal year."""
    dt = year - 1977.68
    n_motion = 2.0 * math.pi / spec.period_years
    omega_bar = math.radians(spec.raan_deg + spec.arg_peri_deg)
    m0 = math.radians(spec.mean_lon_1977_deg) - omega_bar
    m_now = m0 + n_motion * dt
    e_anom = solve_kepler(m_now, spec.ecc)

    x_orb = spec.a_au * (math.cos(e_anom) - spec.ecc)
    y_orb = spec.a_au * math.sqrt(max(0.0, 1.0 - spec.ecc * spec.ecc)) * math.sin(e_anom)

    raan = math.radians(spec.raan_deg)
    inc = math.radians(spec.inc_deg)
    argp = math.radians(spec.arg_peri_deg)

    cos_o, sin_o = math.cos(raan), math.sin(raan)
    cos_i, sin_i = math.cos(inc), math.sin(inc)
    cos_w, sin_w = math.cos(argp), math.sin(argp)

    px = (cos_o * cos_w - sin_o * sin_w * cos_i) * x_orb + (-cos_o * sin_w - sin_o * cos_w * cos_i) * y_orb
    py = (sin_o * cos_w + cos_o * sin_w * cos_i) * x_orb + (-sin_o * sin_w + cos_o * cos_w * cos_i) * y_orb
    pz = (sin_w * sin_i) * x_orb + (cos_w * sin_i) * y_orb
    return (px, py, pz)


def planet_position_sim(spec: PlanetSpec, year: float) -> np.ndarray:
    """Compute planet center position in simulation world meters."""
    return np.array(au_vec_to_sim(planet_position_au(spec, year)), dtype=np.float64)


def sample_orbit_curve_sim(spec: PlanetSpec, num_samples: int = 180) -> List[Tuple[float, float, float]]:
    """Sample a closed Keplerian orbit loop in simulation world meters."""
    pts: List[Tuple[float, float, float]] = []
    t0 = 1977.68
    for i in range(num_samples + 1):
        yr = t0 + spec.period_years * (i / num_samples)
        pts.append(au_vec_to_sim(planet_position_au(spec, yr)))
    return pts


def planet_spin_deg(spec: PlanetSpec, year: float) -> float:
    """Compute visual self-rotation angle in degrees (damped slightly so high time-warp stays smooth)."""
    days = (year - 1977.68) * 365.25
    rot_days = spec.rot_period_hours / 24.0
    # Visual spin scaled by 0.04x so 1x-100x time-warp produces smooth majestic rotation
    raw_deg = (days / rot_days) * 360.0 * 0.04
    return math.fmod(raw_deg, 360.0)


# ---------------------------------------------------------------------------
# Heliosphere / Heliopause Analytical Boundary Surface Math
# ---------------------------------------------------------------------------
def heliosphere_boundary_radius_au(direction_unit: np.ndarray, nose_au: float = 121.0, tail_stretch: float = 2.55) -> float:
    """Rankine-Parker teardrop radial distance (AU) as a function of heliocentric unit direction.

    In the nose direction (`cos_theta = +1`), r = `nose_au` (~121 AU for Heliopause, ~90 AU for Termination Shock).
    In the flanks (`cos_theta = 0`), r ~ 1.32 * `nose_au`.
    In the heliotail (`cos_theta = -1`), r stretches into a comet-like tail (`tail_stretch * nose_au`).
    """
    nose = np.asarray(HELIO_NOSE_DIR, dtype=np.float64)
    nose /= np.linalg.norm(nose)
    d = np.asarray(direction_unit, dtype=np.float64)
    dn = np.linalg.norm(d)
    if dn < 1e-9:
        return nose_au
    d = d / dn
    cos_th = float(np.clip(np.dot(d, nose), -1.0, 1.0))
    # Smooth analytical Rankine-like profile:
    # f(+1) = 1.0, f(0) = 1.32, f(-1) = tail_stretch
    u = 0.5 * (1.0 - cos_th)  # 0 at nose, 0.5 at flank, 1.0 at tail
    shape = 1.0 + 0.64 * u + (tail_stretch - 1.64) * (u ** 2.6)
    return nose_au * shape


# ---------------------------------------------------------------------------
# Voyager 1 & Voyager 2 Historical Grand Tour & Interstellar Splines
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryKnot:
    year: float
    pos_sim: np.ndarray
    true_r_au: float
    vel_km_s: float


def _spherical_au_vec(r_au: float, lon_deg: float, lat_deg: float) -> Tuple[float, float, float]:
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    return (
        r_au * math.cos(lat) * math.cos(lon),
        r_au * math.cos(lat) * math.sin(lon),
        r_au * math.sin(lat),
    )


def _flyby_point_sim(planet_name: str, year: float, radial_out: float, z_offset: float) -> Tuple[np.ndarray, float]:
    """Place spacecraft flyby knot right outside a planet's apparent visual sphere for dramatic flybys."""
    spec = PLANET_BY_NAME[planet_name]
    p_au = planet_position_au(spec, year)
    p_sim = np.array(au_vec_to_sim(p_au), dtype=np.float64)
    r_len = np.linalg.norm(p_sim)
    rad_dir = p_sim / max(r_len, 1e-6)
    tang_dir = np.array([-rad_dir[1], rad_dir[0], 0.0], dtype=np.float64)
    tn = np.linalg.norm(tang_dir)
    if tn > 1e-6:
        tang_dir /= tn
    clearance = spec.visual_radius_m * 1.68
    offset = rad_dir * (clearance * radial_out) + tang_dir * (clearance * 0.65) + np.array([0.0, 0.0, clearance * z_offset])
    return p_sim + offset, spec.a_au


def _build_voyager1_knots() -> List[TrajectoryKnot]:
    earth_1977 = planet_position_sim(PLANET_BY_NAME["Earth"], 1977.685)
    e_rad = earth_1977 / max(np.linalg.norm(earth_1977), 1e-6)
    launch_orbit_pos = earth_1977 + e_rad * (PLANET_BY_NAME["Earth"].visual_radius_m * 1.48)

    jup_pos, jup_au = _flyby_point_sim("Jupiter", 1979.180, radial_out=0.85, z_offset=0.08)
    sat_pos, sat_au = _flyby_point_sim("Saturn", 1980.860, radial_out=1.15, z_offset=-0.42)

    # Post-Saturn northward ecliptic exit (~+34.9 deg ecliptic latitude, ~254 deg ecliptic lon -> toward Heliosphere Nose!)
    # Calibrate exact Heliopause crossing radius along V1's asymptotic direction so V1 crosses the Heliopause shell right at 2012.65!
    v1_dir = np.array(_spherical_au_vec(1.0, 254.2, 34.8), dtype=np.float64)
    hp_au = heliosphere_boundary_radius_au(v1_dir, nose_au=116.8, tail_stretch=2.55)  # ~121.6 AU at +34.8 deg!
    ts_au = heliosphere_boundary_radius_au(v1_dir, nose_au=89.8, tail_stretch=2.35)   # ~94.0 AU

    knots = [
        TrajectoryKnot(1977.685, launch_orbit_pos, 1.001, 38.6),
        TrajectoryKnot(1978.350, np.array(au_vec_to_sim(_spherical_au_vec(2.65, 35.0, 0.8))), 2.65, 24.8),
        TrajectoryKnot(1979.180, jup_pos, jup_au, 35.4),     # Jupiter gravity assist boost!
        TrajectoryKnot(1980.020, np.array(au_vec_to_sim(_spherical_au_vec(7.45, 118.0, 1.6))), 7.45, 23.9),
        TrajectoryKnot(1980.860, sat_pos, sat_au, 31.2),     # Saturn & Titan polar deflection!
        TrajectoryKnot(1985.000, np.array(au_vec_to_sim(_spherical_au_vec(22.8, 228.0, 24.5))), 22.8, 18.4),
        TrajectoryKnot(1989.650, np.array(au_vec_to_sim(_spherical_au_vec(39.4, 246.0, 31.8))), 39.4, 17.6),
        TrajectoryKnot(2004.950, np.array(au_vec_to_sim(tuple(v1_dir * ts_au))), ts_au, 17.1),
        TrajectoryKnot(2012.650, np.array(au_vec_to_sim(tuple(v1_dir * hp_au))), hp_au, 17.04),
        TrajectoryKnot(2018.850, np.array(au_vec_to_sim(tuple(v1_dir * 144.2))), 144.2, 16.99),
        TrajectoryKnot(2026.865, np.array(au_vec_to_sim(tuple(v1_dir * 166.4))), 166.4, 16.95),
        TrajectoryKnot(2550.000, np.array(au_vec_to_sim(tuple(v1_dir * 2040.0))), 2040.0, 16.90),
        TrajectoryKnot(4800.000, np.array(au_vec_to_sim(tuple(v1_dir * 10100.0))), 10100.0, 16.88),
    ]
    return knots


def _build_voyager2_knots() -> List[TrajectoryKnot]:
    earth_1977 = planet_position_sim(PLANET_BY_NAME["Earth"], 1977.685)
    e_rad = earth_1977 / max(np.linalg.norm(earth_1977), 1e-6)
    launch_orbit_pos = earth_1977 + e_rad * (PLANET_BY_NAME["Earth"].visual_radius_m * 1.48) + np.array([0.0, 0.8, -0.4])

    jup_pos, jup_au = _flyby_point_sim("Jupiter", 1979.520, radial_out=0.95, z_offset=-0.06)
    sat_pos, sat_au = _flyby_point_sim("Saturn", 1981.650, radial_out=1.22, z_offset=0.12)
    ura_pos, ura_au = _flyby_point_sim("Uranus", 1986.070, radial_out=0.92, z_offset=0.15)
    nep_pos, nep_au = _flyby_point_sim("Neptune", 1989.650, radial_out=0.88, z_offset=0.52)

    # Post-Neptune southward ecliptic exit (~-32.2 to -48 deg lat, ~289 deg ecliptic lon)
    v2_dir_hp = np.array(_spherical_au_vec(1.0, 288.5, -32.2), dtype=np.float64)
    v2_dir_deep = np.array(_spherical_au_vec(1.0, 291.0, -44.0), dtype=np.float64)
    hp2_au = heliosphere_boundary_radius_au(v2_dir_hp, nose_au=116.8, tail_stretch=2.55)
    ts2_au = heliosphere_boundary_radius_au(v2_dir_hp, nose_au=89.8, tail_stretch=2.35)

    knots = [
        TrajectoryKnot(1977.685, launch_orbit_pos, 1.001, 36.8),
        TrajectoryKnot(1978.550, np.array(au_vec_to_sim(_spherical_au_vec(2.90, 42.0, -0.6))), 2.90, 21.5),
        TrajectoryKnot(1979.520, jup_pos, jup_au, 29.8),     # Jupiter assist
        TrajectoryKnot(1981.650, sat_pos, sat_au, 28.4),     # Saturn assist
        TrajectoryKnot(1986.070, ura_pos, ura_au, 24.6),     # Uranus assist
        TrajectoryKnot(1989.650, nep_pos, nep_au, 27.9),     # Neptune polar skim -> south dive!
        TrajectoryKnot(1998.000, np.array(au_vec_to_sim(_spherical_au_vec(55.2, 282.0, -24.5))), 55.2, 15.9),
        TrajectoryKnot(2007.660, np.array(au_vec_to_sim(tuple(v2_dir_hp * ts2_au))), ts2_au, 15.5),
        TrajectoryKnot(2018.850, np.array(au_vec_to_sim(tuple(v2_dir_hp * hp2_au))), hp2_au, 15.37),
        TrajectoryKnot(2026.865, np.array(au_vec_to_sim(tuple(v2_dir_deep * 139.5))), 139.5, 15.28),
        TrajectoryKnot(2550.000, np.array(au_vec_to_sim(tuple(v2_dir_deep * 1810.0))), 1810.0, 15.20),
        TrajectoryKnot(4800.000, np.array(au_vec_to_sim(tuple(v2_dir_deep * 9050.0))), 9050.0, 15.18),
    ]
    return knots


VOYAGER1_KNOTS: List[TrajectoryKnot] = _build_voyager1_knots()
VOYAGER2_KNOTS: List[TrajectoryKnot] = _build_voyager2_knots()


def _catmull_rom_eval(knots: List[TrajectoryKnot], year: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Evaluate C1 centripetal/uniform Catmull-Rom spline in scene space + telemetry at `year`."""
    if year <= knots[0].year:
        tangent = knots[1].pos_sim - knots[0].pos_sim
        tn = np.linalg.norm(tangent)
        return knots[0].pos_sim.copy(), tangent / max(tn, 1e-6), knots[0].true_r_au, knots[0].vel_km_s
    if year >= knots[-1].year:
        tangent = knots[-1].pos_sim - knots[-2].pos_sim
        tn = np.linalg.norm(tangent)
        return knots[-1].pos_sim.copy(), tangent / max(tn, 1e-6), knots[-1].true_r_au, knots[-1].vel_km_s

    idx = 0
    for i in range(len(knots) - 1):
        if year <= knots[i + 1].year:
            idx = i
            break

    k1 = knots[idx]
    k2 = knots[idx + 1]
    k0 = knots[max(0, idx - 1)]
    k3 = knots[min(len(knots) - 1, idx + 2)]

    u = (year - k1.year) / max(k2.year - k1.year, 1e-9)
    u2 = u * u
    u3 = u2 * u

    p0, p1, p2, p3 = k0.pos_sim, k1.pos_sim, k2.pos_sim, k3.pos_sim
    pos = 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )
    deriv = 0.5 * (
        (-p0 + p2)
        + (4.0 * p0 - 10.0 * p1 + 8.0 * p2 - 2.0 * p3) * u
        + (-3.0 * p0 + 9.0 * p1 - 9.0 * p2 + 3.0 * p3) * u2
    )
    dn = np.linalg.norm(deriv)
    vel_dir = deriv / dn if dn > 1e-8 else (p2 - p1) / max(np.linalg.norm(p2 - p1), 1e-6)

    # Smooth telemetry interpolation
    true_au = k1.true_r_au + u * (k2.true_r_au - k1.true_r_au)
    # Cross-check against scene radial inverse so telemetry matches visual boundary crossings cleanly
    sim_au = sim_radius_to_au(float(np.linalg.norm(pos)))
    blended_au = 0.65 * true_au + 0.35 * sim_au
    vel_kms = k1.vel_km_s + (3 * u2 - 2 * u3) * (k2.vel_km_s - k1.vel_km_s)
    return pos, vel_dir, blended_au, vel_kms


def sample_voyager_curve_sim(spacecraft_id: int = 1, num_samples: int = 320) -> List[Tuple[float, float, float]]:
    """Sample smooth trajectory curve points in simulation meters for UsdGeom.BasisCurves."""
    knots = VOYAGER1_KNOTS if spacecraft_id == 1 else VOYAGER2_KNOTS
    pts: List[Tuple[float, float, float]] = []
    # Dense sampling across planetary tour (1977.685 -> 1992) + interstellar cruise (1992 -> 4800)
    years_inner = np.linspace(knots[0].year, 1992.0, int(num_samples * 0.65))
    years_outer = np.geomspace(1992.1, knots[-1].year, num_samples - len(years_inner))
    for yr in np.concatenate([years_inner, years_outer]):
        pos, _, _, _ = _catmull_rom_eval(knots, float(yr))
        pts.append((float(pos[0]), float(pos[1]), float(pos[2])))
    return pts


def get_heliopause_crossing_sim(spacecraft_id: int = 1) -> Tuple[float, float, float]:
    """Return exact simulation world position (m) of Voyager 1 (2012.65) or Voyager 2 (2018.85) Heliopause crossing."""
    knots = VOYAGER1_KNOTS if spacecraft_id == 1 else VOYAGER2_KNOTS
    cross_yr = 2012.650 if spacecraft_id == 1 else 2018.850
    pos, _, _, _ = _catmull_rom_eval(knots, cross_yr)
    return (float(pos[0]), float(pos[1]), float(pos[2]))


# ---------------------------------------------------------------------------
# Quaternion Attitude Helpers (+Z Dish Boresight -> Target)
# ---------------------------------------------------------------------------
def quat_look_along_plus_z(target_dir: np.ndarray, up_hint: np.ndarray = np.array([0.0, 0.0, 1.0])) -> Tuple[float, float, float, float]:
    """Return (w, x, y, z) quaternion rotating local +Z onto `target_dir`."""
    z_axis = np.asarray(target_dir, dtype=np.float64)
    zn = np.linalg.norm(z_axis)
    if zn < 1e-8:
        return (1.0, 0.0, 0.0, 0.0)
    z_axis /= zn

    up = np.asarray(up_hint, dtype=np.float64)
    if abs(np.dot(z_axis, up)) > 0.96:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(up, z_axis)
    x_axis /= max(np.linalg.norm(x_axis), 1e-8)
    y_axis = np.cross(z_axis, x_axis)

    rot = np.column_stack([x_axis, y_axis, z_axis])
    trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / norm, x / norm, y / norm, z / norm)


# ---------------------------------------------------------------------------
# Complete Frame State & Telemetry Bundle
# ---------------------------------------------------------------------------
@dataclass
class RocketStageState:
    visible: bool
    root_pos: Tuple[float, float, float]
    root_quat_wxyz: Tuple[float, float, float, float]
    scale: float
    srb_sep_offset: float
    srb_peel_deg: float
    srb_plume_scale: float
    core_sep_z: float
    core_plume_scale: float
    centaur_plume_scale: float
    fairing_sep_x: float
    fairing_peel_deg: float


@dataclass
class SpacecraftState:
    pos_sim: Tuple[float, float, float]
    quat_wxyz: Tuple[float, float, float, float]
    vel_dir: Tuple[float, float, float]
    boom_deploy_01: float          # 0 = stowed inside Centaur shroud, 1 = fully deployed
    earth_align_dot: float         # Cosine alignment of +Z HGA dish boresight to Earth
    heliocentric_au: float
    heliocentric_billion_km: float
    earth_distance_au: float
    velocity_km_s: float
    round_trip_light_time_str: str
    plasma_regime: str
    plasma_color_hex: int          # ABGR color for omni.ui


@dataclass
class SimulationFrameState:
    scrub_s: float
    year: float
    earth_pos_sim: Tuple[float, float, float]
    planet_positions_sim: Dict[str, Tuple[float, float, float]]
    planet_spins_deg: Dict[str, float]
    titan_local_pos: Tuple[float, float, float]
    voyager1: SpacecraftState
    voyager2: SpacecraftState
    rocket: RocketStageState


def _format_rtlt(earth_dist_au: float) -> str:
    one_way_sec = (earth_dist_au * AU_KM) / SPEED_OF_LIGHT_KM_S
    rt_sec = 2.0 * one_way_sec
    if rt_sec < 3600.0 * 99.0:
        hrs = int(rt_sec // 3600)
        mins = int((rt_sec % 3600) // 60)
        secs = int(rt_sec % 60)
        return f"{hrs:02d}h {mins:02d}m {secs:02d}s"
    days = rt_sec / 86400.0
    if days < 365.25:
        return f"{days:.1f} days"
    return f"{days / 365.25:.2f} light-yrs"


def _classify_plasma_regime(r_au: float, pos_sim: np.ndarray) -> Tuple[str, int]:
    if r_au < 1.03:
        return ("Earth Magnetosphere / LEO Ascent", 0xFFFFB040)
    d_unit = pos_sim / max(np.linalg.norm(pos_sim), 1e-6)
    ts_au = heliosphere_boundary_radius_au(d_unit, nose_au=89.8, tail_stretch=2.35)
    hp_au = heliosphere_boundary_radius_au(d_unit, nose_au=116.8, tail_stretch=2.55)
    if r_au < ts_au:
        return ("Supersonic Solar Wind (Heliosphere)", 0xFF45C8FF)
    if r_au < hp_au - 1.2:
        return ("Subsonic Heliosheath (Post-Shock)", 0xFF3898FF)
    if r_au <= hp_au + 1.8:
        return ("Heliopause Magnetic Depletion Boundary!", 0xFF60FFFF)
    if r_au < 850.0:
        return ("Interstellar Space (VLISM Cosmic-Ray Regime)", 0xFFFF88C8)
    return ("Oort Cloud Cometary Halo (Deep Interstellar)", 0xFFE8D080)


def evaluate_simulation_frame(scrub_s: float, active_spacecraft: int = 1) -> SimulationFrameState:
    """Master kinematic evaluator for any timeline scrubber position `scrub_s` in [0.0, 1.0]."""
    s = max(0.0, min(1.0, float(scrub_s)))
    year = scrub_to_year(s)

    planet_pos: Dict[str, Tuple[float, float, float]] = {}
    planet_spin: Dict[str, float] = {}
    for p in PLANETS:
        ps = planet_position_sim(p, year)
        planet_pos[p.name] = (float(ps[0]), float(ps[1]), float(ps[2]))
        planet_spin[p.name] = planet_spin_deg(p, year)

    earth_np = np.array(planet_pos["Earth"], dtype=np.float64)
    earth_rad = earth_np / max(np.linalg.norm(earth_np), 1e-6)
    earth_tan = np.array([-earth_rad[1], earth_rad[0], 0.18], dtype=np.float64)
    earth_tan /= max(np.linalg.norm(earth_tan), 1e-6)

    # Titan local orbit around Saturn (r = 17.5 m outside F-ring, inclined 0.3 deg to ring plane)
    titan_ang = 2.0 * math.pi * ((year - 1977.68) * 365.25 / 15.945) * 0.08 + 1.15
    titan_local = (17.5 * math.cos(titan_ang), 17.5 * math.sin(titan_ang), 0.45 * math.sin(titan_ang))

    # Launch & Deployment Phase (s in [0.0, 0.14])
    in_launch = s <= 0.14
    u_launch = min(1.0, max(0.0, s / 0.14))

    # Ascent path rising smoothly from Earth's day-lit LC-41 surface (r_earth_vis = 2.60 m) to orbital injection knot
    surface_anchor = earth_np + earth_rad * (PLANET_BY_NAME["Earth"].visual_radius_m + 0.12)
    v1_orb_knot = VOYAGER1_KNOTS[0].pos_sim
    v2_orb_knot = VOYAGER2_KNOTS[0].pos_sim

    # Rocket visual scale: 0.075 (48m rocket -> ~3.6m cinematic vehicle rising off 5.2m diameter Earth!)
    rocket_scale = 0.075

    if in_launch:
        smooth_u = u_launch * u_launch * (3.0 - 2.0 * u_launch)
        ascent_dir = (1.0 - 0.68 * smooth_u) * earth_rad + (0.68 * smooth_u) * earth_tan
        ascent_dir /= max(np.linalg.norm(ascent_dir), 1e-6)

        target_orb = v1_orb_knot if active_spacecraft == 1 else v2_orb_knot
        rocket_pos_np = (1.0 - smooth_u) * surface_anchor + smooth_u * target_orb
        rocket_quat = quat_look_along_plus_z(ascent_dir, up_hint=np.array([0.0, 0.0, 1.0]))

        # Staging sub-events
        # 1) SRB burn [0..0.30], jettison [0.30..1.0]
        srb_u = max(0.0, (u_launch - 0.28) / 0.72)
        srb_plume = 1.0 if u_launch < 0.28 else max(0.001, 1.0 - (u_launch - 0.28) * 12.0)
        # 2) Core burn [0..0.52], core separation [0.52..1.0]
        core_u = max(0.0, (u_launch - 0.52) / 0.48)
        core_plume = 1.0 if u_launch < 0.52 else max(0.001, 1.0 - (u_launch - 0.52) * 14.0)
        # 3) Fairing jettison [0.46..1.0]
        fair_u = max(0.0, (u_launch - 0.46) / 0.54)
        # 4) Centaur burn [0.52..0.78]
        cent_plume = 1.0 if (0.52 <= u_launch < 0.78) else 0.001
        # 5) Voyager separation & boom unfolding [0.74..1.0]
        sep_u = max(0.0, min(1.0, (u_launch - 0.74) / 0.26))
        boom_01 = sep_u * sep_u * (3.0 - 2.0 * sep_u)

        # Position Voyager inside Centaur shroud (z_local = 38.5 * rocket_scale) before separation, then pulling ahead
        lead_dist = (38.5 * rocket_scale) + sep_u * 6.5
        sc_active_pos = rocket_pos_np + ascent_dir * lead_dist

        # Attitude: transitions smoothly from ascent_dir to Earth-pointing (+Z -> Earth) during deployment!
        to_earth = earth_np - sc_active_pos
        to_earth /= max(np.linalg.norm(to_earth), 1e-6)
        point_dir = (1.0 - boom_01) * ascent_dir + boom_01 * to_earth
        point_dir /= max(np.linalg.norm(point_dir), 1e-6)

        rocket_state = RocketStageState(
            visible=True,
            root_pos=(float(rocket_pos_np[0]), float(rocket_pos_np[1]), float(rocket_pos_np[2])),
            root_quat_wxyz=rocket_quat,
            scale=rocket_scale,
            srb_sep_offset=float(srb_u * 18.0),
            srb_peel_deg=float(srb_u * 46.0),
            srb_plume_scale=float(srb_plume),
            core_sep_z=float(-core_u * 28.0),
            core_plume_scale=float(core_plume),
            centaur_plume_scale=float(cent_plume),
            fairing_sep_x=float(fair_u * 16.0),
            fairing_peel_deg=float(fair_u * 52.0),
        )
    else:
        boom_01 = 1.0
        rocket_state = RocketStageState(
            visible=False,
            root_pos=(float(surface_anchor[0]), float(surface_anchor[1]), float(surface_anchor[2])),
            root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            scale=rocket_scale,
            srb_sep_offset=0.0,
            srb_peel_deg=0.0,
            srb_plume_scale=0.001,
            core_sep_z=0.0,
            core_plume_scale=0.001,
            centaur_plume_scale=0.001,
            fairing_sep_x=0.0,
            fairing_peel_deg=0.0,
        )

    def _build_sc_state(sc_id: int) -> SpacecraftState:
        knots = VOYAGER1_KNOTS if sc_id == 1 else VOYAGER2_KNOTS
        if in_launch:
            if sc_id == active_spacecraft:
                pos_np = sc_active_pos
                p_dir = point_dir
                v_dir = ascent_dir
            else:
                # Sister spacecraft parked on orbital departure slot
                pos_np = v1_orb_knot if sc_id == 1 else v2_orb_knot
                v_dir = earth_tan
                p_dir = (earth_np - pos_np) / max(np.linalg.norm(earth_np - pos_np), 1e-6)
            r_au = 1.0001 + 0.0015 * u_launch
            vel_kms = 7.8 + 30.8 * u_launch
            earth_dist_au = 0.00004 + 0.0018 * u_launch
        else:
            pos_np, v_dir, r_au, vel_kms = _catmull_rom_eval(knots, year)
            to_e = earth_np - pos_np
            p_dir = to_e / max(np.linalg.norm(to_e), 1e-6)
            # True AU vector distance to Earth
            sc_unit = pos_np / max(np.linalg.norm(pos_np), 1e-6)
            sc_au_vec = sc_unit * r_au
            e_au_vec = np.array(planet_position_au(PLANET_BY_NAME["Earth"], year), dtype=np.float64)
            earth_dist_au = float(np.linalg.norm(sc_au_vec - e_au_vec))

        quat = quat_look_along_plus_z(p_dir, up_hint=np.array([0.0, 0.0, 1.0]))
        to_earth_unit = (earth_np - pos_np) / max(np.linalg.norm(earth_np - pos_np), 1e-6)
        align_dot = float(np.clip(np.dot(p_dir, to_earth_unit), -1.0, 1.0))
        regime_str, regime_col = _classify_plasma_regime(r_au, pos_np)

        return SpacecraftState(
            pos_sim=(float(pos_np[0]), float(pos_np[1]), float(pos_np[2])),
            quat_wxyz=quat,
            vel_dir=(float(v_dir[0]), float(v_dir[1]), float(v_dir[2])),
            boom_deploy_01=boom_01 if sc_id == active_spacecraft else 1.0,
            earth_align_dot=align_dot,
            heliocentric_au=r_au,
            heliocentric_billion_km=r_au * AU_BILLION_KM,
            earth_distance_au=earth_dist_au,
            velocity_km_s=vel_kms,
            round_trip_light_time_str=_format_rtlt(earth_dist_au),
            plasma_regime=regime_str,
            plasma_color_hex=regime_col,
        )

    v1_state = _build_sc_state(1)
    v2_state = _build_sc_state(2)

    return SimulationFrameState(
        scrub_s=s,
        year=year,
        earth_pos_sim=planet_pos["Earth"],
        planet_positions_sim=planet_pos,
        planet_spins_deg=planet_spin,
        titan_local_pos=titan_local,
        voyager1=v1_state,
        voyager2=v2_state,
        rocket=rocket_state,
    )
