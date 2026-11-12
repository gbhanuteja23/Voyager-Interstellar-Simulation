"""OpenUSD Stage Assembly, GPU PointInstancer Belts/Oort Cloud, Dynamic Prim Updater,
and 5-Mode Cinematic Camera Rig for the Voyager Interstellar Journey Simulation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

from voyager_interstellar_sim.config import (
    ASSETS_USD_DIR,
    HELIO_NOSE_DIR,
    ISM_INFLOW_DIR,
    PLANETS,
    PLANET_BY_NAME,
    SPACECRAFT_VISUAL_SCALE,
    SUN_CORONA_INNER_M,
    SUN_CORONA_OUTER_M,
    SUN_VISUAL_RADIUS_M,
    TEXTURES_DIR,
    au_to_sim_radius,
    au_vec_to_sim,
)
from voyager_interstellar_sim.kinematics import (
    SimulationFrameState,
    evaluate_simulation_frame,
    get_heliopause_crossing_sim,
    get_trajectory_au_milestones,
    heliosphere_boundary_radius_au,
    sample_orbit_curve_sim,
    sample_voyager_curve_sim,
)
from voyager_interstellar_sim.materials_vfx import (
    create_omnipbr_material,
    define_annulus_ring_mesh,
    define_heliosphere_teardrop_shell,
    define_uv_sphere_mesh,
    ensure_vfx_textures,
)


def make_camera_transform_matrix(
    eye: Tuple[float, float, float],
    target: Tuple[float, float, float],
    up_world: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Gf.Matrix4d:
    """Build right-handed USD Camera-to-World `Gf.Matrix4d` (row-vector convention, -Z look, +Y up)."""
    eye_v = np.asarray(eye, dtype=np.float64)
    tgt_v = np.asarray(target, dtype=np.float64)
    fwd = tgt_v - eye_v
    fn = np.linalg.norm(fwd)
    if fn < 1e-6:
        fwd = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        fwd /= fn

    up_v = np.asarray(up_world, dtype=np.float64)
    up_v /= max(np.linalg.norm(up_v), 1e-8)
    if abs(np.dot(fwd, up_v)) > 0.97:
        up_v = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = np.cross(fwd, up_v)
    right /= max(np.linalg.norm(right), 1e-8)
    cam_up = np.cross(right, fwd)
    cam_up /= max(np.linalg.norm(cam_up), 1e-8)
    back = -fwd

    return Gf.Matrix4d(
        float(right[0]), float(right[1]), float(right[2]), 0.0,
        float(cam_up[0]), float(cam_up[1]), float(cam_up[2]), 0.0,
        float(back[0]), float(back[1]), float(back[2]), 0.0,
        float(eye_v[0]), float(eye_v[1]), float(eye_v[2]), 1.0,
    )


def _resolve_portable_asset(raw_path: object, fallback_dir: Path, default_name: str = "") -> str:
    """Resolve an asset path portably across machines/OSes when a repo clone moves directories."""
    if raw_path:
        p = Path(str(raw_path).replace("\\", "/"))
        if p.exists():
            return p.as_posix()
        candidate = fallback_dir / p.name
        if candidate.exists():
            return candidate.as_posix()
    if default_name:
        return (fallback_dir / default_name).as_posix()
    return str(raw_path or "")


class VoyagerSceneManager:
    """Builds and animates the complete OpenUSD Voyager Solar-to-Interstellar stage."""

    def __init__(self, stage: Usd.Stage, manifest: Dict[str, object]) -> None:
        self.stage: Usd.Stage = stage
        self.manifest: Dict[str, object] = manifest
        raw_tex = dict(manifest.get("textures", {}))  # type: ignore
        self.textures: Dict[str, str] = {
            k: _resolve_portable_asset(v, TEXTURES_DIR) for k, v in raw_tex.items()
        }
        self.vfx_textures: Dict[str, str] = ensure_vfx_textures()

        # Cached fast XformOp handles for zero-overhead per-frame updates
        self._planet_translate_ops: Dict[str, UsdGeom.XformOp] = {}
        self._planet_spin_ops: Dict[str, UsdGeom.XformOp] = {}
        self._earth_cloud_spin_op: Optional[UsdGeom.XformOp] = None
        self._titan_translate_op: Optional[UsdGeom.XformOp] = None
        self._corona_spin_op: Optional[UsdGeom.XformOp] = None
        self._sun_key_rot_op: Optional[UsdGeom.XformOp] = None

        self._belt_spin_ops: Dict[str, UsdGeom.XformOp] = {}
        self._stream_offset_ops: Dict[str, UsdGeom.XformOp] = {}
        self._beacon_scale_ops: List[UsdGeom.XformOp] = []
        self._milestone_scale_ops: List[UsdGeom.XformOp] = []

        # Spacecraft cached ops: {1: {...}, 2: {...}}
        self._sc_ops: Dict[int, Dict[str, object]] = {1: {}, 2: {}}
        self._rocket_ops: Dict[str, object] = {}
        self._camera_ops: Dict[str, UsdGeom.XformOp] = {}
        self._free_orbit_initialized: bool = False
        self._last_wide_cam: Optional[bool] = None
        self._pulse_phase: float = 0.0

    # ------------------------------------------------------------------
    # Master Stage Build
    # ------------------------------------------------------------------
    def build_full_scene(self) -> None:
        """Assemble the entire USD hierarchy under `/World`."""
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)

        world = UsdGeom.Xform.Define(self.stage, "/World")
        self.stage.SetDefaultPrim(world.GetPrim())

        UsdGeom.Scope.Define(self.stage, "/World/Materials")
        self._configure_rtx_settings()
        self._build_environment_and_sun()
        self._build_planets_and_orbits()
        self._build_belts_and_oort_instancers()
        self._build_heliosphere_and_streams()
        self._build_launch_complex_and_spacecraft()
        self._build_camera_rig()

        # Initialize stage transforms to Launch preset (s=0.025) or default frame
        init_state = evaluate_simulation_frame(0.025, active_spacecraft=1)
        self.apply_frame_state(init_state, active_spacecraft=1, active_camera="VoyagerChaseCam", dt_wall=0.0)

    def _configure_rtx_settings(self) -> None:
        try:
            import carb.settings
            settings = carb.settings.get_settings()
            settings.set("/app/viewport/grid/enabled", False)
            settings.set("/rtx/post/glare/enabled", True)
            settings.set("/rtx/post/glare/gain", 0.35)
            settings.set("/rtx/raytracing/fractionalCutoutOpacity", True)
            settings.set("/rtx/indirectDiffuse/enabled", False)  # Saves VRAM & boosts FPS in deep space
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 1. Environment & Sun + Volumetric-Style Multi-Shell Corona
    # ------------------------------------------------------------------
    def _build_environment_and_sun(self) -> None:
        UsdGeom.Xform.Define(self.stage, "/World/Environment")

        dome = UsdLux.DomeLight.Define(self.stage, "/World/Environment/MilkyWayDomeLight")
        dome.CreateIntensityAttr(85.0)
        dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
        mw_tex = self.textures.get("milky_way")
        if mw_tex:
            dome.CreateTextureFileAttr(Sdf.AssetPath(mw_tex))

        # Dynamic sun-tracking DistantLight so planets & Voyager receive crisp solar specular glint & day/night terminators
        sun_key = UsdLux.DistantLight.Define(self.stage, "/World/Environment/SolarDirectionalKeyLight")
        sun_key.CreateIntensityAttr(3400.0)
        sun_key.CreateAngleAttr(0.53)
        sun_key.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.91))
        self._sun_key_rot_op = sun_key.AddTransformOp()

        fill = UsdLux.DistantLight.Define(self.stage, "/World/Environment/GalacticFillLight")
        fill.CreateIntensityAttr(18.0)
        fill.CreateColorAttr(Gf.Vec3f(0.55, 0.68, 1.0))
        fill.AddRotateXYZOp().Set(Gf.Vec3f(-55.0, 30.0, 0.0))

        # Blazing Incandescent Sun at origin
        UsdGeom.Xform.Define(self.stage, "/World/SolarSystem")
        UsdGeom.Xform.Define(self.stage, "/World/SolarSystem/Sun")

        mat_sun = create_omnipbr_material(
            self.stage,
            "/World/Materials/Mat_SunPhotosphere",
            diffuse_color=(1.0, 0.95, 0.80),
            diffuse_texture=self.textures.get("sun"),
            roughness=0.85,
            emissive_color=(1.0, 0.88, 0.56),
            emissive_texture=self.textures.get("sun"),
            emissive_intensity=28000.0,
        )
        sun_mesh = define_uv_sphere_mesh(
            self.stage, "/World/SolarSystem/Sun/Photosphere", SUN_VISUAL_RADIUS_M, 44, 88, (1.0, 0.92, 0.65)
        )
        UsdShade.MaterialBindingAPI.Apply(sun_mesh.GetPrim()).Bind(mat_sun)

        # Volumetric-style inner + outer corona shells
        mat_cor_in = create_omnipbr_material(
            self.stage,
            "/World/Materials/Mat_SunCoronaInner",
            diffuse_color=(1.0, 0.78, 0.30),
            emissive_color=(1.0, 0.72, 0.24),
            emissive_intensity=6500.0,
            opacity_constant=0.22,
            opacity_texture=self.vfx_textures.get("corona_opacity"),
        )
        mat_cor_out = create_omnipbr_material(
            self.stage,
            "/World/Materials/Mat_SunCoronaOuter",
            diffuse_color=(1.0, 0.55, 0.14),
            emissive_color=(1.0, 0.50, 0.12),
            emissive_intensity=2200.0,
            opacity_constant=0.09,
            opacity_texture=self.vfx_textures.get("corona_opacity"),
        )

        cor_xf = UsdGeom.Xform.Define(self.stage, "/World/SolarSystem/Sun/CoronaGroup")
        self._corona_spin_op = cor_xf.AddRotateZOp(opSuffix="spin")
        self._corona_spin_op.Set(0.0)

        cor_in = define_uv_sphere_mesh(
            self.stage, "/World/SolarSystem/Sun/CoronaGroup/CoronaInner", SUN_CORONA_INNER_M, 28, 56, (1.0, 0.75, 0.3)
        )
        UsdShade.MaterialBindingAPI.Apply(cor_in.GetPrim()).Bind(mat_cor_in)

        cor_out = define_uv_sphere_mesh(
            self.stage, "/World/SolarSystem/Sun/CoronaGroup/CoronaOuter", SUN_CORONA_OUTER_M, 28, 56, (1.0, 0.50, 0.14)
        )
        UsdShade.MaterialBindingAPI.Apply(cor_out.GetPrim()).Bind(mat_cor_out)

        s_light = UsdLux.SphereLight.Define(self.stage, "/World/SolarSystem/Sun/SolarSphereLight")
        s_light.CreateRadiusAttr(SUN_VISUAL_RADIUS_M)
        s_light.CreateIntensityAttr(780000.0)
        s_light.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.90))

    # ------------------------------------------------------------------
    # 2. All 8 Planets, Rayleigh Atmospheres, Saturn Rings, Titan & Orbits
    # ------------------------------------------------------------------
    def _build_planets_and_orbits(self) -> None:
        UsdGeom.Scope.Define(self.stage, "/World/SolarSystem/Orbits")
        UsdGeom.Xform.Define(self.stage, "/World/SolarSystem/Planets")

        for spec in PLANETS:
            # Sleek, Translucent + Emissive Keplerian Orbit Curve
            orb_pts = sample_orbit_curve_sim(spec, num_samples=180)
            oc = tuple(min(1.0, c * 0.78 + 0.12) for c in spec.base_color)
            mat_orb = create_omnipbr_material(
                self.stage,
                f"/World/Materials/Mat_{spec.name}_Orbit",
                diffuse_color=oc,
                roughness=0.22,
                specular_level=0.60,
                emissive_color=oc,
                emissive_intensity=220.0,
                opacity_constant=0.22,
            )
            curve = UsdGeom.BasisCurves.Define(self.stage, f"/World/SolarSystem/Orbits/{spec.name}_Orbit")
            curve.CreateTypeAttr(UsdGeom.Tokens.linear)
            curve.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in orb_pts]))
            curve.CreateCurveVertexCountsAttr(Vt.IntArray([len(orb_pts)]))
            curve.CreateWidthsAttr(Vt.FloatArray([0.10] * len(orb_pts)))
            curve.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*oc)]))
            curve.CreateDisplayOpacityAttr(Vt.FloatArray([0.22]))
            UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(mat_orb)

            # Tangent-space normal & ocean-roughness maps where available
            normal_tex = None
            if spec.name == "Earth":
                normal_tex = self.vfx_textures.get("earth_normal")
            elif spec.name == "Mercury":
                normal_tex = self.vfx_textures.get("mercury_normal")
            elif spec.name == "Mars":
                normal_tex = self.vfx_textures.get("mars_normal")

            rough_tex = self.vfx_textures.get("earth_roughness") if spec.name == "Earth" else None

            # Pure physically-based planetary surface (emissive_intensity=0 -> realistic sunlight reflection & crisp shadow terminator!)
            mat_p = create_omnipbr_material(
                self.stage,
                f"/World/Materials/Mat_{spec.name}",
                diffuse_color=spec.base_color,
                diffuse_texture=self.textures.get(spec.texture_key),
                bump_texture=normal_tex,
                bump_factor=0.85 if spec.name == "Earth" else 0.72,
                roughness=spec.roughness,
                roughness_texture=rough_tex,
                specular_level=spec.specular,
                metallic=0.02,
                emissive_intensity=0.0,
            )

            p_root = UsdGeom.Xform.Define(self.stage, f"/World/SolarSystem/Planets/{spec.name}")
            t_op = p_root.AddTranslateOp()
            self._planet_translate_ops[spec.name] = t_op

            tilt_xf = UsdGeom.Xform.Define(self.stage, f"/World/SolarSystem/Planets/{spec.name}/AxialTilt")
            tilt_xf.AddRotateXOp().Set(float(spec.tilt_deg))

            spin_xf = UsdGeom.Xform.Define(self.stage, f"/World/SolarSystem/Planets/{spec.name}/SpinBody")
            s_op = spin_xf.AddRotateZOp()
            self._planet_spin_ops[spec.name] = s_op

            globe = define_uv_sphere_mesh(
                self.stage,
                f"/World/SolarSystem/Planets/{spec.name}/SpinBody/GlobeMesh",
                spec.visual_radius_m,
                44,
                88,
                spec.base_color,
            )
            UsdShade.MaterialBindingAPI.Apply(globe.GetPrim()).Bind(mat_p)

            # Sunlit Rayleigh / Haze Atmospheric Rim Shell
            if spec.atmo_opacity > 0.0:
                mat_atmo = create_omnipbr_material(
                    self.stage,
                    f"/World/Materials/Mat_{spec.name}_Atmosphere",
                    diffuse_color=spec.atmo_color,
                    roughness=0.22,
                    specular_level=0.88,
                    emissive_intensity=0.0,
                    opacity_constant=spec.atmo_opacity,
                )
                atmo_mesh = define_uv_sphere_mesh(
                    self.stage,
                    f"/World/SolarSystem/Planets/{spec.name}/AxialTilt/AtmosphereHalo",
                    spec.visual_radius_m * 1.028,
                    32,
                    64,
                    spec.atmo_color,
                )
                UsdShade.MaterialBindingAPI.Apply(atmo_mesh.GetPrim()).Bind(mat_atmo)

            # Earth Sunlit Cloud Layer
            if spec.name == "Earth":
                mat_clouds = create_omnipbr_material(
                    self.stage,
                    "/World/Materials/Mat_EarthClouds",
                    diffuse_color=(0.98, 0.99, 1.0),
                    diffuse_texture=self.textures.get("earth_clouds"),
                    roughness=0.68,
                    specular_level=0.45,
                    emissive_intensity=0.0,
                    opacity_constant=0.78,
                    opacity_texture=self.textures.get("earth_clouds"),
                )
                cloud_xf = UsdGeom.Xform.Define(self.stage, "/World/SolarSystem/Planets/Earth/AxialTilt/CloudLayer")
                self._earth_cloud_spin_op = cloud_xf.AddRotateZOp()
                c_mesh = define_uv_sphere_mesh(
                    self.stage,
                    "/World/SolarSystem/Planets/Earth/AxialTilt/CloudLayer/CloudsMesh",
                    spec.visual_radius_m * 1.014,
                    36,
                    72,
                    (1.0, 1.0, 1.0),
                )
                UsdShade.MaterialBindingAPI.Apply(c_mesh.GetPrim()).Bind(mat_clouds)

            # Saturn Translucent Sunlit Rings + Titan Moon
            if spec.has_rings:
                mat_rings = create_omnipbr_material(
                    self.stage,
                    "/World/Materials/Mat_SaturnRings",
                    diffuse_color=(0.94, 0.86, 0.70),
                    diffuse_texture=self.textures.get("saturn_rings"),
                    roughness=0.34,
                    specular_level=0.78,
                    emissive_intensity=0.0,
                    opacity_constant=0.92,
                    opacity_texture=self.textures.get("saturn_rings_opacity"),
                )
                rings_mesh = define_annulus_ring_mesh(
                    self.stage,
                    "/World/SolarSystem/Planets/Saturn/AxialTilt/RingsAnnulusMesh",
                    inner_radius=spec.visual_radius_m * 1.25,
                    outer_radius=spec.visual_radius_m * 2.34,
                    n_radial=28,
                    n_seg=128,
                )
                UsdShade.MaterialBindingAPI.Apply(rings_mesh.GetPrim()).Bind(mat_rings)

                mat_titan = create_omnipbr_material(
                    self.stage,
                    "/World/Materials/Mat_Titan",
                    diffuse_color=(0.90, 0.64, 0.30),
                    diffuse_texture=self.textures.get("titan"),
                    roughness=0.52,
                    specular_level=0.48,
                    emissive_intensity=0.0,
                )
                titan_xf = UsdGeom.Xform.Define(self.stage, "/World/SolarSystem/Planets/Saturn/AxialTilt/TitanMoon")
                self._titan_translate_op = titan_xf.AddTranslateOp()
                titan_mesh = define_uv_sphere_mesh(
                    self.stage,
                    "/World/SolarSystem/Planets/Saturn/AxialTilt/TitanMoon/Globe",
                    2.40,
                    24,
                    48,
                    (0.90, 0.64, 0.30),
                )
                UsdShade.MaterialBindingAPI.Apply(titan_mesh.GetPrim()).Bind(mat_titan)

    # ------------------------------------------------------------------
    # 3. Asteroid Belt, Kuiper Belt & Oort Cloud (`UsdGeom.PointInstancer`)
    # ------------------------------------------------------------------
    def _build_belts_and_oort_instancers(self) -> None:
        UsdGeom.Xform.Define(self.stage, "/World/Belts")
        rel_proto_path = _resolve_portable_asset(
            self.manifest.get("asteroid_prototypes_usd"), ASSETS_USD_DIR, "asteroid_prototypes.usd"
        )

        mat_rock = create_omnipbr_material(
            self.stage,
            "/World/Materials/Mat_AsteroidSilicate",
            diffuse_color=(0.55, 0.50, 0.44),
            diffuse_texture=self.textures.get("asteroid_rock"),
            roughness=0.88,
            emissive_color=(0.45, 0.40, 0.35),
            emissive_intensity=28.0,
        )
        mat_ice = create_omnipbr_material(
            self.stage,
            "/World/Materials/Mat_KuiperOortIce",
            diffuse_color=(0.68, 0.80, 0.92),
            diffuse_texture=self.textures.get("kuiper_ice"),
            roughness=0.52,
            emissive_color=(0.55, 0.72, 0.90),
            emissive_intensity=35.0,
        )

        def _populate_ring_instancer(
            parent_path: str,
            instancer_name: str,
            count: int,
            r_au_min: float,
            r_au_max: float,
            inc_sigma_deg: float,
            scale_range: Tuple[float, float],
            proto_names: List[str],
            bind_mat: UsdShade.Material,
            spherical_shell: bool = False,
            seed: int = 42,
        ) -> UsdGeom.PointInstancer:
            grp = UsdGeom.Xform.Define(self.stage, f"{parent_path}/{instancer_name}_Group")
            spin_op = grp.AddRotateZOp(opSuffix="orbitSpin")
            spin_op.Set(0.0)
            self._belt_spin_ops[instancer_name] = spin_op

            inst = UsdGeom.PointInstancer.Define(self.stage, f"{parent_path}/{instancer_name}_Group/{instancer_name}")
            proto_container = UsdGeom.Scope.Define(
                self.stage, f"{parent_path}/{instancer_name}_Group/{instancer_name}/Prototypes"
            )
            proto_rel = inst.GetPrototypesRel()

            for p_idx, p_name in enumerate(proto_names):
                p_prim_path = f"{proto_container.GetPath()}/Proto_{p_idx}"
                xf = UsdGeom.Xform.Define(self.stage, p_prim_path)
                xf.GetPrim().GetReferences().AddReference(rel_proto_path, f"/AsteroidPrototypes/{p_name}")
                UsdShade.MaterialBindingAPI.Apply(xf.GetPrim()).Bind(bind_mat)
                proto_rel.AddTarget(Sdf.Path(p_prim_path))

            rng = np.random.default_rng(seed)
            if spherical_shell:
                # Log-uniform radial distribution across Oort Cloud (1,000 .. 24,000 AU)
                log_r = rng.uniform(math.log(r_au_min), math.log(r_au_max), size=count)
                r_au_arr = np.exp(log_r)
                cos_lat = rng.uniform(-1.0, 1.0, size=count)
                lat = np.arcsin(cos_lat)
                lon = rng.uniform(0.0, 2.0 * np.pi, size=count)
            else:
                r_au_arr = rng.uniform(r_au_min, r_au_max, size=count)
                lat = rng.normal(0.0, math.radians(inc_sigma_deg), size=count)
                lon = rng.uniform(0.0, 2.0 * np.pi, size=count)

            positions: List[Gf.Vec3f] = []
            orientations: List[Gf.Quath] = []
            scales: List[Gf.Vec3f] = []
            proto_indices: List[int] = rng.integers(0, len(proto_names), size=count).tolist()

            s_vals = rng.uniform(scale_range[0], scale_range[1], size=(count, 3))
            q_raw = rng.normal(0.0, 1.0, size=(count, 4))
            q_raw /= np.linalg.norm(q_raw, axis=1, keepdims=True)

            for i in range(count):
                r_sim = au_to_sim_radius(float(r_au_arr[i]))
                clat = math.cos(float(lat[i]))
                px = r_sim * clat * math.cos(float(lon[i]))
                py = r_sim * clat * math.sin(float(lon[i]))
                pz = r_sim * math.sin(float(lat[i]))
                positions.append(Gf.Vec3f(float(px), float(py), float(pz)))
                orientations.append(Gf.Quath(float(q_raw[i, 0]), float(q_raw[i, 1]), float(q_raw[i, 2]), float(q_raw[i, 3])))
                scales.append(Gf.Vec3f(float(s_vals[i, 0]), float(s_vals[i, 1]), float(s_vals[i, 2])))

            inst.CreatePositionsAttr(Vt.Vec3fArray(positions))
            inst.CreateOrientationsAttr(Vt.QuathArray(orientations))
            inst.CreateScalesAttr(Vt.Vec3fArray(scales))
            inst.CreateProtoIndicesAttr(Vt.IntArray(proto_indices))
            return inst

        # Main Asteroid Belt (2.15 - 3.30 AU): 2,400 instances across 5 prototypes
        _populate_ring_instancer(
            "/World/Belts",
            "MainAsteroidBelt",
            count=2400,
            r_au_min=2.15,
            r_au_max=3.30,
            inc_sigma_deg=6.8,
            scale_range=(0.55, 1.45),
            proto_names=["Proto_Silicate_0", "Proto_Silicate_1", "Proto_Carbonaceous_2", "Proto_Silicate_0", "Proto_KuiperTholin_4"],
            bind_mat=mat_rock,
            spherical_shell=False,
            seed=2001,
        )

        # Kuiper Belt (32.5 - 51.5 AU): 2,800 icy planetesimals across 5 prototypes
        _populate_ring_instancer(
            "/World/Belts",
            "KuiperBelt",
            count=2800,
            r_au_min=32.5,
            r_au_max=51.5,
            inc_sigma_deg=11.5,
            scale_range=(0.95, 2.40),
            proto_names=["Proto_KuiperIce_3", "Proto_KuiperTholin_4", "Proto_OortComet_5", "Proto_Carbonaceous_2", "Proto_KuiperIce_3"],
            bind_mat=mat_ice,
            spherical_shell=False,
            seed=2002,
        )

        # Oort Cloud Spherical Shell (1,000 - 24,000 AU): 3,200 icy nuclei
        UsdGeom.Xform.Define(self.stage, "/World/Interstellar")
        _populate_ring_instancer(
            "/World/Interstellar",
            "OortCloud",
            count=3200,
            r_au_min=1050.0,
            r_au_max=24000.0,
            inc_sigma_deg=90.0,
            scale_range=(1.80, 4.40),
            proto_names=["Proto_OortComet_5", "Proto_KuiperIce_3", "Proto_KuiperTholin_4", "Proto_Silicate_1", "Proto_OortComet_5"],
            bind_mat=mat_ice,
            spherical_shell=True,
            seed=2003,
        )

    # ------------------------------------------------------------------
    # 4. Heliosphere, Termination Shock, Heliopause, Crossing Markers & Streams
    # ------------------------------------------------------------------
    def _build_heliosphere_and_streams(self) -> None:
        # Termination Shock (~90 AU nose)
        mat_ts = create_omnipbr_material(
            self.stage,
            "/World/Materials/Mat_TerminationShock",
            diffuse_color=(1.0, 0.62, 0.25),
            diffuse_texture=self.vfx_textures.get("heliopause_color"),
            emissive_color=(1.0, 0.58, 0.20),
            emissive_intensity=180.0,
            opacity_constant=0.05,
            opacity_texture=self.vfx_textures.get("heliopause_opacity"),
        )
        ts_mesh = define_heliosphere_teardrop_shell(
            self.stage,
            "/World/Interstellar/TerminationShock",
            nose_au=89.8,
            tail_stretch=2.35,
            n_polar=36,
            n_azim=60,
            wireframe_curves_path="/World/Interstellar/TerminationShock_Contours",
            wire_color=(1.0, 0.65, 0.25),
        )
        UsdShade.MaterialBindingAPI.Apply(ts_mesh.GetPrim()).Bind(mat_ts)
        ts_contours_prim = self.stage.GetPrimAtPath("/World/Interstellar/TerminationShock_Contours")
        if ts_contours_prim.IsValid():
            UsdShade.MaterialBindingAPI.Apply(ts_contours_prim).Bind(mat_ts)

        # Heliopause (~121 AU nose, comet-tail stretched downwind by VLISM)
        mat_hp = create_omnipbr_material(
            self.stage,
            "/World/Materials/Mat_Heliopause",
            diffuse_color=(0.20, 0.85, 1.0),
            diffuse_texture=self.vfx_textures.get("heliopause_color"),
            emissive_color=(0.18, 0.82, 1.0),
            emissive_texture=self.vfx_textures.get("heliopause_color"),
            emissive_intensity=340.0,
            opacity_constant=0.11,
            opacity_texture=self.vfx_textures.get("heliopause_opacity"),
        )
        hp_mesh = define_heliosphere_teardrop_shell(
            self.stage,
            "/World/Interstellar/Heliopause",
            nose_au=116.8,
            tail_stretch=2.55,
            n_polar=44,
            n_azim=72,
            wireframe_curves_path="/World/Interstellar/Heliopause_Contours",
            wire_color=(0.22, 0.88, 1.0),
        )
        UsdShade.MaterialBindingAPI.Apply(hp_mesh.GetPrim()).Bind(mat_hp)
        hp_contours_prim = self.stage.GetPrimAtPath("/World/Interstellar/Heliopause_Contours")
        if hp_contours_prim.IsValid():
            UsdShade.MaterialBindingAPI.Apply(hp_contours_prim).Bind(mat_hp)

        # Glowing Translucent Heliopause Crossing Waypoint Markers: Voyager 1 (2012) & Voyager 2 (2018)
        UsdGeom.Xform.Define(self.stage, "/World/Interstellar/CrossingMarkers")
        beacon_specs = [
            ("Voyager1_Heliopause_2012", get_heliopause_crossing_sim(1), (1.0, 0.82, 0.22)),
            ("Voyager2_Heliopause_2018", get_heliopause_crossing_sim(2), (0.22, 0.95, 1.0)),
        ]
        for b_name, b_pos, b_col in beacon_specs:
            mat_b = create_omnipbr_material(
                self.stage,
                f"/World/Materials/Mat_{b_name}",
                diffuse_color=b_col,
                roughness=0.20,
                emissive_color=b_col,
                emissive_intensity=1800.0,
                opacity_constant=0.24,
            )
            b_xf = UsdGeom.Xform.Define(self.stage, f"/World/Interstellar/CrossingMarkers/{b_name}")
            b_xf.AddTranslateOp().Set(Gf.Vec3d(*b_pos))
            sc_op = b_xf.AddScaleOp(opSuffix="pulse")
            sc_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))
            self._beacon_scale_ops.append(sc_op)

            core = UsdGeom.Sphere.Define(self.stage, f"/World/Interstellar/CrossingMarkers/{b_name}/Core")
            core.CreateRadiusAttr(5.5)
            UsdShade.MaterialBindingAPI.Apply(core.GetPrim()).Bind(mat_b)

            # Dual orthogonal beacon halo rings
            for r_idx, rot_xyz in enumerate(((0.0, 0.0, 0.0), (90.0, 0.0, 0.0))):
                ring_xf = UsdGeom.Xform.Define(self.stage, f"/World/Interstellar/CrossingMarkers/{b_name}/Halo_{r_idx}")
                ring_xf.AddRotateXYZOp().Set(Gf.Vec3f(*rot_xyz))
                ring_m = define_annulus_ring_mesh(
                    self.stage,
                    f"/World/Interstellar/CrossingMarkers/{b_name}/Halo_{r_idx}/Ring",
                    inner_radius=12.0,
                    outer_radius=14.2,
                    n_radial=2,
                    n_seg=48,
                )
                UsdShade.MaterialBindingAPI.Apply(ring_m.GetPrim()).Bind(mat_b)

        # Directional Solar-Wind vs. Interstellar Cosmic-Ray Particle Streams (confined to outer Heliosphere / VLISM!)
        UsdGeom.Xform.Define(self.stage, "/World/Interstellar/ParticleStreams")
        mat_sw = create_omnipbr_material(
            self.stage,
            "/World/Materials/Mat_SolarWindIon",
            diffuse_color=(1.0, 0.85, 0.35),
            roughness=0.22,
            emissive_color=(1.0, 0.80, 0.30),
            emissive_intensity=950.0,
            opacity_constant=0.26,
        )
        mat_cr = create_omnipbr_material(
            self.stage,
            "/World/Materials/Mat_CosmicRayIon",
            diffuse_color=(1.0, 0.38, 0.88),
            roughness=0.22,
            emissive_color=(0.95, 0.32, 0.88),
            emissive_intensity=1100.0,
            opacity_constant=0.28,
        )

        # Solar Wind radial plasma streaks in Outer Heliosphere / Heliosheath (58 - 114 AU -> never clutters planets!)
        sw_grp = UsdGeom.Xform.Define(self.stage, "/World/Interstellar/ParticleStreams/SolarWindGroup")
        self._stream_offset_ops["solar_wind"] = sw_grp.AddScaleOp(opSuffix="wave")
        sw_inst = UsdGeom.PointInstancer.Define(self.stage, "/World/Interstellar/ParticleStreams/SolarWindGroup/SolarWindStream")
        sw_proto = UsdGeom.Capsule.Define(self.stage, "/World/Interstellar/ParticleStreams/SolarWindGroup/SolarWindStream/IonStreak")
        sw_proto.CreateRadiusAttr(0.08)
        sw_proto.CreateHeightAttr(2.8)
        sw_proto.CreateAxisAttr("Z")
        UsdShade.MaterialBindingAPI.Apply(sw_proto.GetPrim()).Bind(mat_sw)
        sw_inst.GetPrototypesRel().AddTarget(sw_proto.GetPath())

        rng = np.random.default_rng(7701)
        n_sw = 550
        sw_pos, sw_quat, sw_sc = [], [], []
        sw_r_min = au_to_sim_radius(58.0)
        sw_r_max = au_to_sim_radius(114.0)
        from voyager_interstellar_sim.kinematics import quat_look_along_plus_z
        for _ in range(n_sw):
            d_vec = rng.normal(0.0, 1.0, size=3)
            d_vec /= max(np.linalg.norm(d_vec), 1e-6)
            r_sim = rng.uniform(sw_r_min, sw_r_max)
            p = d_vec * r_sim
            q = quat_look_along_plus_z(d_vec)
            sw_pos.append(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))
            sw_quat.append(Gf.Quath(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
            sw_sc.append(Gf.Vec3f(1.0, 1.0, 1.0))
        sw_inst.CreatePositionsAttr(Vt.Vec3fArray(sw_pos))
        sw_inst.CreateOrientationsAttr(Vt.QuathArray(sw_quat))
        sw_inst.CreateScalesAttr(Vt.Vec3fArray(sw_sc))
        sw_inst.CreateProtoIndicesAttr(Vt.IntArray([0] * n_sw))

        # Cosmic-Ray / VLISM directional particle stream outside Heliopause (650 instances)
        cr_grp = UsdGeom.Xform.Define(self.stage, "/World/Interstellar/ParticleStreams/CosmicRayGroup")
        self._stream_offset_ops["cosmic_ray"] = cr_grp.AddTranslateOp(opSuffix="drift")
        cr_inst = UsdGeom.PointInstancer.Define(self.stage, "/World/Interstellar/ParticleStreams/CosmicRayGroup/CosmicRayStream")
        cr_proto = UsdGeom.Capsule.Define(self.stage, "/World/Interstellar/ParticleStreams/CosmicRayGroup/CosmicRayStream/CosmicStreak")
        cr_proto.CreateRadiusAttr(0.12)
        cr_proto.CreateHeightAttr(4.2)
        cr_proto.CreateAxisAttr("Z")
        UsdShade.MaterialBindingAPI.Apply(cr_proto.GetPrim()).Bind(mat_cr)
        cr_inst.GetPrototypesRel().AddTarget(cr_proto.GetPath())

        inflow = np.asarray(ISM_INFLOW_DIR, dtype=np.float64)
        inflow /= np.linalg.norm(inflow)
        q_inflow = quat_look_along_plus_z(inflow)
        n_cr = 650
        cr_pos, cr_quat, cr_sc = [], [], []
        for _ in range(n_cr):
            d_vec = rng.normal(0.0, 1.0, size=3)
            d_vec /= max(np.linalg.norm(d_vec), 1e-6)
            # Bias toward upstream Heliosphere Nose hemisphere
            if np.dot(d_vec, -inflow) < -0.35:
                d_vec = -d_vec
            hp_r = au_to_sim_radius(heliosphere_boundary_radius_au(d_vec, nose_au=116.8, tail_stretch=2.55))
            r_sim = hp_r + rng.uniform(28.0, 340.0)
            p = d_vec * r_sim
            cr_pos.append(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))
            cr_quat.append(Gf.Quath(float(q_inflow[0]), float(q_inflow[1]), float(q_inflow[2]), float(q_inflow[3])))
            cr_sc.append(Gf.Vec3f(1.0, 1.0, 1.0))
        cr_inst.CreatePositionsAttr(Vt.Vec3fArray(cr_pos))
        cr_inst.CreateOrientationsAttr(Vt.QuathArray(cr_quat))
        cr_inst.CreateScalesAttr(Vt.Vec3fArray(cr_sc))
        cr_inst.CreateProtoIndicesAttr(Vt.IntArray([0] * n_cr))

    # ------------------------------------------------------------------
    # 5. Titan IIIE / Centaur Launch Vehicle & Voyager 1 / 2 Spacecraft
    # ------------------------------------------------------------------
    def _build_launch_complex_and_spacecraft(self) -> None:
        # Titan IIIE / Centaur Reference
        UsdGeom.Xform.Define(self.stage, "/World/LaunchComplex")
        titan_usd_path = _resolve_portable_asset(
            self.manifest.get("titan_iiie_centaur_usd"), ASSETS_USD_DIR, "titan_iiie_centaur.usd"
        )

        titan_xf = UsdGeom.Xform.Define(self.stage, "/World/LaunchComplex/TitanIIIE_Centaur")
        titan_xf.GetPrim().GetReferences().AddReference(titan_usd_path)
        self._rocket_ops["imageable"] = UsdGeom.Imageable(titan_xf.GetPrim())
        self._rocket_ops["translate"] = titan_xf.AddTranslateOp(opSuffix="worldPos")
        self._rocket_ops["orient"] = titan_xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble, opSuffix="worldRot")
        self._rocket_ops["scale"] = titan_xf.AddScaleOp(opSuffix="worldScale")

        # Sub-stage override ops on composed TitanIIIE children
        def _get_or_add_op(prim_path: str, op_type: str, suffix: str):
            xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(prim_path))
            for existing in xf.GetOrderedXformOps():
                if existing.GetOpName().endswith(suffix):
                    return existing
            if op_type == "translate":
                return xf.AddTranslateOp(opSuffix=suffix)
            if op_type == "rotateX":
                return xf.AddRotateXOp(opSuffix=suffix)
            if op_type == "rotateY":
                return xf.AddRotateYOp(opSuffix=suffix)
            return xf.AddScaleOp(opSuffix=suffix)

        self._rocket_ops["srb_l_sep"] = _get_or_add_op("/World/LaunchComplex/TitanIIIE_Centaur/SRB_Left", "translate", "sep")
        self._rocket_ops["srb_r_sep"] = _get_or_add_op("/World/LaunchComplex/TitanIIIE_Centaur/SRB_Right", "translate", "sep")
        self._rocket_ops["srb_l_peel"] = _get_or_add_op("/World/LaunchComplex/TitanIIIE_Centaur/SRB_Left", "rotateX", "peel")
        self._rocket_ops["srb_r_peel"] = _get_or_add_op("/World/LaunchComplex/TitanIIIE_Centaur/SRB_Right", "rotateX", "peel")
        self._rocket_ops["srb_l_plume"] = _get_or_add_op("/World/LaunchComplex/TitanIIIE_Centaur/SRB_Left/SRBPlume", "scale", "throttle")
        self._rocket_ops["srb_r_plume"] = _get_or_add_op("/World/LaunchComplex/TitanIIIE_Centaur/SRB_Right/SRBPlume", "scale", "throttle")

        self._rocket_ops["core_sep"] = _get_or_add_op("/World/LaunchComplex/TitanIIIE_Centaur/CoreStage", "translate", "sep")
        self._rocket_ops["core_plume"] = _get_or_add_op("/World/LaunchComplex/TitanIIIE_Centaur/CoreStage/CorePlume", "scale", "throttle")
        self._rocket_ops["centaur_plume"] = _get_or_add_op("/World/LaunchComplex/TitanIIIE_Centaur/CentaurStage/CentaurPlume", "scale", "throttle")

        self._rocket_ops["fair_l_sep"] = _get_or_add_op("/World/LaunchComplex/TitanIIIE_Centaur/Fairing_Left", "translate", "sep")
        self._rocket_ops["fair_r_sep"] = _get_or_add_op("/World/LaunchComplex/TitanIIIE_Centaur/Fairing_Right", "translate", "sep")
        self._rocket_ops["fair_l_peel"] = _get_or_add_op("/World/LaunchComplex/TitanIIIE_Centaur/Fairing_Left", "rotateY", "peel")
        self._rocket_ops["fair_r_peel"] = _get_or_add_op("/World/LaunchComplex/TitanIIIE_Centaur/Fairing_Right", "rotateY", "peel")

        # Spacecraft & Trajectory Splines + 3D AU Milestone Nodes
        UsdGeom.Xform.Define(self.stage, "/World/Spacecraft")
        UsdGeom.Scope.Define(self.stage, "/World/Spacecraft/TrajectoryCurves")
        UsdGeom.Xform.Define(self.stage, "/World/Spacecraft/TrajectoryMilestones")

        voyager_usd_path = _resolve_portable_asset(
            self.manifest.get("voyager_articulated_usd"), ASSETS_USD_DIR, "voyager_articulated.usd"
        )

        sc_configs = [
            (1, "Voyager1", (1.0, 0.78, 0.20)),
            (2, "Voyager2", (0.20, 0.90, 1.0)),
        ]
        for sc_id, sc_name, sc_color in sc_configs:
            # Smooth Catmull-Rom Trajectory Curve with Translucent + Emissive Holographic Material
            mat_traj = create_omnipbr_material(
                self.stage,
                f"/World/Materials/Mat_{sc_name}_TrajCurve",
                diffuse_color=sc_color,
                roughness=0.18,
                specular_level=0.65,
                emissive_color=sc_color,
                emissive_intensity=520.0,
                opacity_constant=0.22,
            )
            c_pts = sample_voyager_curve_sim(sc_id, num_samples=340)
            traj = UsdGeom.BasisCurves.Define(self.stage, f"/World/Spacecraft/TrajectoryCurves/{sc_name}_Trajectory")
            traj.CreateTypeAttr(UsdGeom.Tokens.linear)
            traj.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in c_pts]))
            traj.CreateCurveVertexCountsAttr(Vt.IntArray([len(c_pts)]))
            traj.CreateWidthsAttr(Vt.FloatArray([0.010] * len(c_pts)))
            traj.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*sc_color)]))
            traj.CreateDisplayOpacityAttr(Vt.FloatArray([0.22]))
            UsdShade.MaterialBindingAPI.Apply(traj.GetPrim()).Bind(mat_traj)

            sc_xf = UsdGeom.Xform.Define(self.stage, f"/World/Spacecraft/{sc_name}")
            sc_xf.GetPrim().GetReferences().AddReference(voyager_usd_path)
            t_op = sc_xf.AddTranslateOp(opSuffix="worldPos")
            r_op = sc_xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble, opSuffix="worldRot")
            s_op = sc_xf.AddScaleOp(opSuffix="worldScale")
            s_op.Set(Gf.Vec3f(SPACECRAFT_VISUAL_SCALE, SPACECRAFT_VISUAL_SCALE, SPACECRAFT_VISUAL_SCALE))

            # Distant translucent telemetry locator halo (scales up only in overview cameras so Voyager is always easy to spot!)
            mat_loc = create_omnipbr_material(
                self.stage,
                f"/World/Materials/Mat_{sc_name}_Locator",
                diffuse_color=sc_color,
                roughness=0.20,
                emissive_color=sc_color,
                emissive_intensity=1800.0,
                opacity_constant=0.24,
            )
            loc_sphere = UsdGeom.Sphere.Define(self.stage, f"/World/Spacecraft/{sc_name}/TelemetryBeaconHalo")
            loc_sphere.CreateRadiusAttr(1.0)
            loc_scale_op = loc_sphere.AddScaleOp(opSuffix="haloScale")
            loc_scale_op.Set(Gf.Vec3f(0.001, 0.001, 0.001))
            UsdShade.MaterialBindingAPI.Apply(loc_sphere.GetPrim()).Bind(mat_loc)

            # 3D AU Milestone Nodes along trajectory (> 1.5 AU so Earth launch neighborhood stays crisp)
            for m_idx, (m_label, m_au, m_pos) in enumerate(get_trajectory_au_milestones(sc_id)):
                if m_au < 1.5:
                    continue
                ms_xf = UsdGeom.Xform.Define(
                    self.stage, f"/World/Spacecraft/TrajectoryMilestones/{sc_name}_Node_{m_idx}"
                )
                ms_xf.AddTranslateOp().Set(Gf.Vec3d(*m_pos))
                ms_scale_op = ms_xf.AddScaleOp(opSuffix="camScale")
                ms_scale_op.Set(Gf.Vec3f(0.05, 0.05, 0.05))
                self._milestone_scale_ops.append(ms_scale_op)
                ms_xf.GetPrim().SetCustomDataByKey("milestone_label", f"{m_label} ({m_au:.1f} AU)")
                ms_sp = UsdGeom.Sphere.Define(
                    self.stage, f"/World/Spacecraft/TrajectoryMilestones/{sc_name}_Node_{m_idx}/Marker"
                )
                ms_sp.CreateRadiusAttr(0.75)
                UsdShade.MaterialBindingAPI.Apply(ms_sp.GetPrim()).Bind(mat_loc)

            # Articulated boom deployment ops
            mag_op = _get_or_add_op(f"/World/Spacecraft/{sc_name}/MagnetometerBoomPivot", "rotateY", "deploy")
            rtg_op = _get_or_add_op(f"/World/Spacecraft/{sc_name}/RTGBoomPivot", "rotateY", "deploy")
            scan_op = _get_or_add_op(f"/World/Spacecraft/{sc_name}/ScanPlatformPivot", "rotateX", "deploy")

            # Live Translucent + Emissive Earth Radio Link Curve (2-point BasisCurve updated per frame)
            mat_radio = create_omnipbr_material(
                self.stage,
                f"/World/Materials/Mat_{sc_name}_RadioCurve",
                diffuse_color=sc_color,
                roughness=0.18,
                specular_level=0.65,
                emissive_color=sc_color,
                emissive_intensity=620.0,
                opacity_constant=0.18,
            )
            link_curve = UsdGeom.BasisCurves.Define(self.stage, f"/World/Spacecraft/TrajectoryCurves/{sc_name}_EarthRadioLink")
            link_curve.CreateTypeAttr(UsdGeom.Tokens.linear)
            link_curve.CreateCurveVertexCountsAttr(Vt.IntArray([2]))
            link_curve.CreateWidthsAttr(Vt.FloatArray([0.004, 0.004]))
            link_curve.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*sc_color)]))
            link_curve.CreateDisplayOpacityAttr(Vt.FloatArray([0.18]))
            UsdShade.MaterialBindingAPI.Apply(link_curve.GetPrim()).Bind(mat_radio)

            self._sc_ops[sc_id] = {
                "translate": t_op,
                "orient": r_op,
                "scale": s_op,
                "mag_deploy": mag_op,
                "rtg_deploy": rtg_op,
                "scan_deploy": scan_op,
                "halo_scale": loc_scale_op,
                "traj_curve": traj,
                "traj_len": len(c_pts),
                "radio_link": link_curve,
            }

    # ------------------------------------------------------------------
    # 6. Multi-Camera Rig (`/World/Cameras/...`)
    # ------------------------------------------------------------------
    def _build_camera_rig(self) -> None:
        UsdGeom.Xform.Define(self.stage, "/World/Cameras")
        cam_specs = [
            ("VoyagerChaseCam", 24.0),
            ("RocketStagingCam", 26.0),
            ("EclipticOverviewCam", 20.0),
            ("HeliopauseCrossSectionCam", 22.0),
            ("FreeOrbitCam", 24.0),
        ]
        for c_name, focal_mm in cam_specs:
            cam = UsdGeom.Camera.Define(self.stage, f"/World/Cameras/{c_name}")
            cam.CreateFocalLengthAttr(float(focal_mm))
            cam.CreateHorizontalApertureAttr(36.0)
            cam.CreateVerticalApertureAttr(20.25)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.04, 65000.0))
            self._camera_ops[c_name] = cam.AddTransformOp()

    # ------------------------------------------------------------------
    # Dynamic Per-Frame Scene & Camera Update
    # ------------------------------------------------------------------
    def apply_frame_state(
        self,
        state: SimulationFrameState,
        active_spacecraft: int = 1,
        active_camera: str = "VoyagerChaseCam",
        dt_wall: float = 0.016,
    ) -> None:
        """Update all planetary orbits, rocket staging, Voyager articulation/attitude, VFX & cameras."""
        self._pulse_phase += dt_wall * 2.4

        # 1. Planets, Clouds & Titan
        for p_name, pos in state.planet_positions_sim.items():
            self._planet_translate_ops[p_name].Set(Gf.Vec3d(*pos))
            self._planet_spin_ops[p_name].Set(float(state.planet_spins_deg[p_name]))

        if self._earth_cloud_spin_op is not None:
            self._earth_cloud_spin_op.Set(float(state.planet_spins_deg["Earth"] * 1.28))
        if self._titan_translate_op is not None:
            self._titan_translate_op.Set(Gf.Vec3d(*state.titan_local_pos))
        if self._corona_spin_op is not None:
            self._corona_spin_op.Set(float((self._pulse_phase * 4.5) % 360.0))

        # 2. Belt & Stream subtle motion + pulsing Heliopause crossing beacons
        if "MainAsteroidBelt" in self._belt_spin_ops:
            self._belt_spin_ops["MainAsteroidBelt"].Set(float(((state.year - 1977.68) * 14.0) % 360.0))
        if "KuiperBelt" in self._belt_spin_ops:
            self._belt_spin_ops["KuiperBelt"].Set(float(((state.year - 1977.68) * 1.2) % 360.0))

        if "solar_wind" in self._stream_offset_ops:
            w = 0.96 + 0.06 * math.sin(self._pulse_phase * 1.8)
            self._stream_offset_ops["solar_wind"].Set(Gf.Vec3f(w, w, w))
        if "cosmic_ray" in self._stream_offset_ops:
            drift = ((self._pulse_phase * 12.0) % 45.0) - 22.5
            inflow = ISM_INFLOW_DIR
            self._stream_offset_ops["cosmic_ray"].Set(
                Gf.Vec3d(inflow[0] * drift, inflow[1] * drift, inflow[2] * drift)
            )
        wide_cam = active_camera in ("EclipticOverviewCam", "HeliopauseCrossSectionCam")
        beacon_mult = 1.0 if wide_cam else 0.06
        pulse_s = beacon_mult * (1.0 + 0.16 * math.sin(self._pulse_phase * 2.2))
        for b_op in self._beacon_scale_ops:
            b_op.Set(Gf.Vec3f(pulse_s, pulse_s, pulse_s))

        ms_mult = 1.0 if wide_cam else 0.05
        for ms_op in self._milestone_scale_ops:
            ms_op.Set(Gf.Vec3f(ms_mult, ms_mult, ms_mult))

        # 3. Titan IIIE / Centaur Launch Staging
        rk = state.rocket
        rk_img: UsdGeom.Imageable = self._rocket_ops["imageable"]  # type: ignore
        if rk.visible:
            rk_img.MakeVisible()
            self._rocket_ops["translate"].Set(Gf.Vec3d(*rk.root_pos))  # type: ignore
            qw, qx, qy, qz = rk.root_quat_wxyz
            self._rocket_ops["orient"].Set(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)))  # type: ignore
            self._rocket_ops["scale"].Set(Gf.Vec3f(rk.scale, rk.scale, rk.scale))  # type: ignore

            self._rocket_ops["srb_l_sep"].Set(Gf.Vec3d(0.0, -3.15 - rk.srb_sep_offset, -0.45 * rk.srb_sep_offset))  # type: ignore
            self._rocket_ops["srb_r_sep"].Set(Gf.Vec3d(0.0, 3.15 + rk.srb_sep_offset, -0.45 * rk.srb_sep_offset))  # type: ignore
            self._rocket_ops["srb_l_peel"].Set(float(rk.srb_peel_deg))  # type: ignore
            self._rocket_ops["srb_r_peel"].Set(float(-rk.srb_peel_deg))  # type: ignore

            sp = rk.srb_plume_scale
            self._rocket_ops["srb_l_plume"].Set(Gf.Vec3f(sp, sp, sp))  # type: ignore
            self._rocket_ops["srb_r_plume"].Set(Gf.Vec3f(sp, sp, sp))  # type: ignore

            self._rocket_ops["core_sep"].Set(Gf.Vec3d(0.0, 0.0, rk.core_sep_z))  # type: ignore
            cp = rk.core_plume_scale
            self._rocket_ops["core_plume"].Set(Gf.Vec3f(cp, cp, cp))  # type: ignore
            ctp = rk.centaur_plume_scale
            self._rocket_ops["centaur_plume"].Set(Gf.Vec3f(ctp, ctp, ctp))  # type: ignore

            self._rocket_ops["fair_l_sep"].Set(Gf.Vec3d(-rk.fairing_sep_x, 0.0, 35.0 - 0.3 * rk.fairing_sep_x))  # type: ignore
            self._rocket_ops["fair_r_sep"].Set(Gf.Vec3d(rk.fairing_sep_x, 0.0, 35.0 - 0.3 * rk.fairing_sep_x))  # type: ignore
            self._rocket_ops["fair_l_peel"].Set(float(-rk.fairing_peel_deg))  # type: ignore
            self._rocket_ops["fair_r_peel"].Set(float(rk.fairing_peel_deg))  # type: ignore
        else:
            rk_img.MakeInvisible()

        # 4. Voyager 1 & Voyager 2 Transform, Boom Unfolding, Adaptive Curve Widths & Earth Radio Sightline
        # Child of /World/Spacecraft/VoyagerX (scale = 0.12), so local 52.0 -> 6.24 m glowing beacon in overview cameras
        halo_r = 52.0 if wide_cam else 0.001
        update_curve_widths = (wide_cam != self._last_wide_cam)
        if update_curve_widths:
            self._last_wide_cam = wide_cam
        traj_w = 0.28 if wide_cam else 0.010
        radio_w = 0.12 if wide_cam else 0.004

        earth_p = np.asarray(state.earth_pos_sim, dtype=np.float64)

        for sc_id, sc_st in ((1, state.voyager1), (2, state.voyager2)):
            ops = self._sc_ops[sc_id]
            ops["translate"].Set(Gf.Vec3d(*sc_st.pos_sim))  # type: ignore
            qw, qx, qy, qz = sc_st.quat_wxyz
            ops["orient"].Set(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)))  # type: ignore

            # Sleek aerospace scale (0.12x -> 0.44m dish next to towering 28m Earth & 72m Jupiter!)
            sc_scale = SPACECRAFT_VISUAL_SCALE * (0.38 + 0.62 * sc_st.boom_deploy_01 if state.rocket.visible else 1.0)
            ops["scale"].Set(Gf.Vec3f(sc_scale, sc_scale, sc_scale))  # type: ignore

            stowed_frac = 1.0 - sc_st.boom_deploy_01
            ops["mag_deploy"].Set(float(-84.0 * stowed_frac))  # type: ignore
            ops["rtg_deploy"].Set(float(78.0 * stowed_frac))  # type: ignore
            ops["scan_deploy"].Set(float(-72.0 * stowed_frac))  # type: ignore
            ops["halo_scale"].Set(Gf.Vec3f(halo_r, halo_r, halo_r))  # type: ignore

            link_curve: UsdGeom.BasisCurves = ops["radio_link"]  # type: ignore
            if update_curve_widths:
                t_curve: UsdGeom.BasisCurves = ops["traj_curve"]  # type: ignore
                t_len: int = int(ops["traj_len"])  # type: ignore
                t_curve.GetWidthsAttr().Set(Vt.FloatArray([traj_w] * t_len))
                link_curve.GetWidthsAttr().Set(Vt.FloatArray([radio_w, radio_w]))

            # Start radio sightline at tip of High-Gain Dish feed (+0.26m toward Earth) so it never cuts through bus
            sc_p_vec = np.asarray(sc_st.pos_sim, dtype=np.float64)
            to_e = earth_p - sc_p_vec
            dist_e = float(np.linalg.norm(to_e))
            if dist_e > 0.40:
                beam_start = sc_p_vec + (to_e / dist_e) * 0.26
            else:
                beam_start = sc_p_vec
            link_curve.GetPointsAttr().Set(
                Vt.Vec3fArray([Gf.Vec3f(float(beam_start[0]), float(beam_start[1]), float(beam_start[2])), Gf.Vec3f(*state.earth_pos_sim)])
            )

        # 5. Aim Solar Directional Key Light from Sun (0,0,0) toward active Voyager
        active_sc = state.voyager1 if active_spacecraft == 1 else state.voyager2
        if self._sun_key_rot_op is not None:
            sc_p = active_sc.pos_sim
            self._sun_key_rot_op.Set(make_camera_transform_matrix((0.0, 0.0, 0.0), sc_p))

        # 6. Update all Camera Rigs
        self.update_cameras(state, active_spacecraft=active_spacecraft, active_camera=active_camera)

    def update_cameras(
        self,
        state: SimulationFrameState,
        active_spacecraft: int = 1,
        active_camera: str = "VoyagerChaseCam",
    ) -> None:
        """Compute smooth framing matrices for all 5 mission cameras."""
        sc = state.voyager1 if active_spacecraft == 1 else state.voyager2
        sc_pos = np.array(sc.pos_sim, dtype=np.float64)
        earth_pos = np.array(state.earth_pos_sim, dtype=np.float64)

        to_earth = earth_pos - sc_pos
        to_earth_u = to_earth / max(np.linalg.norm(to_earth), 1e-6)

        # Dynamically frame Voyager against the nearest planet during Launch & Giant Planet Flybys
        backdrop_dir = to_earth_u.copy()
        best_weight = 0.0
        best_p_dir = to_earth_u
        for p_spec in PLANETS:
            p_pos = np.array(state.planet_positions_sim[p_spec.name], dtype=np.float64)
            vec_to_p = p_pos - sc_pos
            dist_p = float(np.linalg.norm(vec_to_p))
            capture_r = p_spec.visual_radius_m * (5.2 if p_spec.has_rings else 4.6)
            if dist_p < capture_r and dist_p > 1e-5:
                u_w = 1.0 - (dist_p / capture_r)
                w = min(1.0, u_w * 1.8)
                if w > best_weight:
                    best_weight = w
                    best_p_dir = vec_to_p / dist_p

        if best_weight > 0.0:
            backdrop_dir = (1.0 - best_weight) * to_earth_u + best_weight * best_p_dir
            backdrop_dir /= max(np.linalg.norm(backdrop_dir), 1e-6)

        side = np.cross(np.array([0.0, 0.0, 1.0]), backdrop_dir)
        if np.linalg.norm(side) < 1e-4:
            side = np.array([1.0, 0.0, 0.0])
        side /= np.linalg.norm(side)

        # 1. Voyager Chase Cam: sleek close framing on 0.44m Voyager dish with towering sunlit planets filling the horizon!
        chase_eye = sc_pos - backdrop_dir * 1.85 + side * 0.92 + np.array([0.0, 0.0, 0.46])
        chase_tgt = sc_pos + backdrop_dir * 0.55
        self._camera_ops["VoyagerChaseCam"].Set(
            make_camera_transform_matrix(tuple(chase_eye), tuple(chase_tgt))
        )

        # 2. Rocket Staging Cam: frames 1.56m Titan IIIE / Centaur & Voyager directly against 28m sunlit Earth horizon!
        if state.rocket.visible:
            rk_pos = np.array(state.rocket.root_pos, dtype=np.float64)
            mid_tgt = 0.42 * rk_pos + 0.58 * sc_pos
            earth_up_u = mid_tgt - earth_pos
            earth_up_u /= max(np.linalg.norm(earth_up_u), 1e-6)
            cam_horiz_u = np.cross(earth_up_u, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(cam_horiz_u) < 1e-4:
                cam_horiz_u = np.array([1.0, 0.0, 0.0])
            cam_horiz_u /= np.linalg.norm(cam_horiz_u)
            stg_eye = mid_tgt + cam_horiz_u * 2.55 + earth_up_u * 0.85
            stg_look = mid_tgt - earth_up_u * 0.65
            self._camera_ops["RocketStagingCam"].Set(
                make_camera_transform_matrix(tuple(stg_eye), tuple(stg_look))
            )
        else:
            # Post-staging: dramatic front-quarter Scan Platform / Golden Record close inspection cam
            insp_eye = sc_pos + backdrop_dir * 1.65 + side * 1.25 + np.array([0.0, 0.0, 0.48])
            self._camera_ops["RocketStagingCam"].Set(
                make_camera_transform_matrix(tuple(insp_eye), tuple(sc_pos))
            )

        # 3. Solar System Ecliptic Overview Cam (zooms out smoothly if Voyager enters Oort Cloud)
        sc_r = float(np.linalg.norm(sc_pos))
        overview_dist = max(2350.0, sc_r * 1.62)
        ov_eye = (0.0, -overview_dist * 0.78, overview_dist * 0.74)
        self._camera_ops["EclipticOverviewCam"].Set(
            make_camera_transform_matrix(ov_eye, (0.0, 0.0, 0.0), up_world=(0.0, 0.0, 1.0))
        )

        # 4. Heliopause Cross-Section Cam (perpendicular to HELIO_NOSE_DIR so nose + heliotail + V1/V2 crossings frame cleanly)
        nose = np.asarray(HELIO_NOSE_DIR, dtype=np.float64)
        nose /= np.linalg.norm(nose)
        perp = np.cross(nose, np.array([0.0, 0.0, 1.0]))
        perp /= max(np.linalg.norm(perp), 1e-6)
        hp_center = -nose * 320.0
        hp_eye = hp_center + perp * 3650.0 + np.array([0.0, 0.0, 520.0])
        self._camera_ops["HeliopauseCrossSectionCam"].Set(
            make_camera_transform_matrix(tuple(hp_eye), tuple(hp_center), up_world=(0.0, 0.0, 1.0))
        )

        # 5. Free Orbit Cam: initialized once so user viewport camera manipulation is never overwritten
        if not self._free_orbit_initialized:
            free_eye = sc_pos - backdrop_dir * 2.8 + side * 2.2 + np.array([0.0, 0.0, 1.1])
            self._camera_ops["FreeOrbitCam"].Set(
                make_camera_transform_matrix(tuple(free_eye), tuple(sc_pos))
            )
            self._free_orbit_initialized = True


def set_active_viewport_camera(camera_prim_path: str) -> bool:
    """Switch the primary Omniverse Kit viewport to `camera_prim_path` cleanly."""
    try:
        import omni.kit.viewport.utility as vp_util
        vp = vp_util.get_active_viewport()
        if vp is not None:
            vp.camera_path = Sdf.Path(camera_prim_path)
            return True
    except Exception:
        pass
    return False
