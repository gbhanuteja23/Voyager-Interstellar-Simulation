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
    SUN_CORONA_INNER_M,
    SUN_CORONA_OUTER_M,
    SUN_VISUAL_RADIUS_M,
    au_to_sim_radius,
    au_vec_to_sim,
)
from voyager_interstellar_sim.kinematics import (
    SimulationFrameState,
    evaluate_simulation_frame,
    get_heliopause_crossing_sim,
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


class VoyagerSceneManager:
    """Builds and animates the complete OpenUSD Voyager Solar-to-Interstellar stage."""

    def __init__(self, stage: Usd.Stage, manifest: Dict[str, object]) -> None:
        self.stage: Usd.Stage = stage
        self.manifest: Dict[str, object] = manifest
        self.textures: Dict[str, str] = dict(manifest.get("textures", {}))  # type: ignore
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

        # Spacecraft cached ops: {1: {...}, 2: {...}}
        self._sc_ops: Dict[int, Dict[str, object]] = {1: {}, 2: {}}
        self._rocket_ops: Dict[str, object] = {}
        self._camera_ops: Dict[str, UsdGeom.XformOp] = {}
        self._free_orbit_initialized: bool = False
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
            settings.set("/rtx/indirectDiffuse/enabled", False)  # Saves VRAM & boosts FPS in deep space
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 1. Environment & Sun + Volumetric-Style Multi-Shell Corona
    # ------------------------------------------------------------------
    def _build_environment_and_sun(self) -> None:
        UsdGeom.Xform.Define(self.stage, "/World/Environment")

        dome = UsdLux.DomeLight.Define(self.stage, "/World/Environment/MilkyWayDomeLight")
        dome.CreateIntensityAttr(520.0)
        dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
        mw_tex = self.textures.get("milky_way")
        if mw_tex:
            dome.CreateTextureFileAttr(Sdf.AssetPath(mw_tex))

        # Dynamic sun-tracking DistantLight so outer planets & Voyager at 166+ AU receive crisp directional sunlight
        sun_key = UsdLux.DistantLight.Define(self.stage, "/World/Environment/SolarDirectionalKeyLight")
        sun_key.CreateIntensityAttr(2100.0)
        sun_key.CreateAngleAttr(0.53)
        sun_key.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.88))
        self._sun_key_rot_op = sun_key.AddTransformOp()

        fill = UsdLux.DistantLight.Define(self.stage, "/World/Environment/GalacticFillLight")
        fill.CreateIntensityAttr(110.0)
        fill.CreateColorAttr(Gf.Vec3f(0.68, 0.78, 1.0))
        fill.AddRotateXYZOp().Set(Gf.Vec3f(-55.0, 30.0, 0.0))

        # Sun at origin
        UsdGeom.Xform.Define(self.stage, "/World/SolarSystem")
        UsdGeom.Xform.Define(self.stage, "/World/SolarSystem/Sun")

        mat_sun = create_omnipbr_material(
            self.stage,
            "/World/Materials/Mat_SunPhotosphere",
            diffuse_color=(1.0, 0.78, 0.28),
            diffuse_texture=self.textures.get("sun"),
            roughness=0.9,
            emissive_color=(1.0, 0.72, 0.22),
            emissive_texture=self.textures.get("sun"),
            emissive_intensity=5500.0,
        )
        sun_mesh = define_uv_sphere_mesh(
            self.stage, "/World/SolarSystem/Sun/Photosphere", SUN_VISUAL_RADIUS_M, 40, 80, (1.0, 0.78, 0.25)
        )
        UsdShade.MaterialBindingAPI.Apply(sun_mesh.GetPrim()).Bind(mat_sun)

        # Volumetric-style inner + outer corona shells
        mat_cor_in = create_omnipbr_material(
            self.stage,
            "/World/Materials/Mat_SunCoronaInner",
            diffuse_color=(1.0, 0.65, 0.18),
            emissive_color=(1.0, 0.60, 0.15),
            emissive_intensity=2600.0,
            opacity_constant=0.26,
            opacity_texture=self.vfx_textures.get("corona_opacity"),
        )
        mat_cor_out = create_omnipbr_material(
            self.stage,
            "/World/Materials/Mat_SunCoronaOuter",
            diffuse_color=(1.0, 0.45, 0.10),
            emissive_color=(1.0, 0.42, 0.08),
            emissive_intensity=1100.0,
            opacity_constant=0.11,
            opacity_texture=self.vfx_textures.get("corona_opacity"),
        )

        cor_xf = UsdGeom.Xform.Define(self.stage, "/World/SolarSystem/Sun/CoronaGroup")
        self._corona_spin_op = cor_xf.AddRotateZOp(opSuffix="spin")
        self._corona_spin_op.Set(0.0)

        cor_in = define_uv_sphere_mesh(
            self.stage, "/World/SolarSystem/Sun/CoronaGroup/CoronaInner", SUN_CORONA_INNER_M, 28, 56, (1.0, 0.65, 0.2)
        )
        UsdShade.MaterialBindingAPI.Apply(cor_in.GetPrim()).Bind(mat_cor_in)

        cor_out = define_uv_sphere_mesh(
            self.stage, "/World/SolarSystem/Sun/CoronaGroup/CoronaOuter", SUN_CORONA_OUTER_M, 28, 56, (1.0, 0.45, 0.1)
        )
        UsdShade.MaterialBindingAPI.Apply(cor_out.GetPrim()).Bind(mat_cor_out)

        s_light = UsdLux.SphereLight.Define(self.stage, "/World/SolarSystem/Sun/SolarSphereLight")
        s_light.CreateRadiusAttr(SUN_VISUAL_RADIUS_M)
        s_light.CreateIntensityAttr(420000.0)
        s_light.CreateColorAttr(Gf.Vec3f(1.0, 0.94, 0.82))

    # ------------------------------------------------------------------
    # 2. All 8 Planets, Saturn Translucent Rings, Titan & Keplerian Orbits
    # ------------------------------------------------------------------
    def _build_planets_and_orbits(self) -> None:
        UsdGeom.Scope.Define(self.stage, "/World/SolarSystem/Orbits")
        UsdGeom.Xform.Define(self.stage, "/World/SolarSystem/Planets")

        for spec in PLANETS:
            # Keplerian Orbit Curve
            orb_pts = sample_orbit_curve_sim(spec, num_samples=180)
            curve = UsdGeom.BasisCurves.Define(self.stage, f"/World/SolarSystem/Orbits/{spec.name}_Orbit")
            curve.CreateTypeAttr(UsdGeom.Tokens.linear)
            curve.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in orb_pts]))
            curve.CreateCurveVertexCountsAttr(Vt.IntArray([len(orb_pts)]))
            curve.CreateWidthsAttr(Vt.FloatArray([0.42] * len(orb_pts)))
            oc = tuple(min(1.0, c * 0.85 + 0.15) for c in spec.base_color)
            curve.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*oc)]))

            # Planet Material (with subtle self-emission floor so shaded hemispheres remain visible against black space)
            bump_tex = self.textures.get("earth_bump") if spec.name == "Earth" else None
            mat_p = create_omnipbr_material(
                self.stage,
                f"/World/Materials/Mat_{spec.name}",
                diffuse_color=spec.base_color,
                diffuse_texture=self.textures.get(spec.texture_key),
                bump_texture=bump_tex,
                roughness=spec.roughness,
                metallic=0.02,
                emissive_color=spec.base_color,
                emissive_texture=self.textures.get(spec.texture_key),
                emissive_intensity=18.0,
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
                36,
                72,
                spec.base_color,
            )
            UsdShade.MaterialBindingAPI.Apply(globe.GetPrim()).Bind(mat_p)

            # Earth Clouds + LC-41 Launch Anchor
            if spec.name == "Earth":
                mat_clouds = create_omnipbr_material(
                    self.stage,
                    "/World/Materials/Mat_EarthClouds",
                    diffuse_color=(0.96, 0.98, 1.0),
                    diffuse_texture=self.textures.get("earth_clouds"),
                    roughness=0.75,
                    opacity_constant=0.72,
                    opacity_texture=self.textures.get("earth_clouds"),
                )
                cloud_xf = UsdGeom.Xform.Define(self.stage, "/World/SolarSystem/Planets/Earth/AxialTilt/CloudLayer")
                self._earth_cloud_spin_op = cloud_xf.AddRotateZOp()
                c_mesh = define_uv_sphere_mesh(
                    self.stage,
                    "/World/SolarSystem/Planets/Earth/AxialTilt/CloudLayer/CloudsMesh",
                    spec.visual_radius_m * 1.018,
                    32,
                    64,
                    (1.0, 1.0, 1.0),
                )
                UsdShade.MaterialBindingAPI.Apply(c_mesh.GetPrim()).Bind(mat_clouds)

            # Saturn Translucent Rings + Titan Moon
            if spec.has_rings:
                mat_rings = create_omnipbr_material(
                    self.stage,
                    "/World/Materials/Mat_SaturnRings",
                    diffuse_color=(0.88, 0.80, 0.64),
                    diffuse_texture=self.textures.get("saturn_rings"),
                    roughness=0.65,
                    emissive_color=(0.85, 0.76, 0.58),
                    emissive_texture=self.textures.get("saturn_rings"),
                    emissive_intensity=22.0,
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
                    diffuse_color=(0.86, 0.60, 0.26),
                    diffuse_texture=self.textures.get("titan"),
                    roughness=0.72,
                    emissive_color=(0.86, 0.60, 0.26),
                    emissive_intensity=25.0,
                )
                titan_xf = UsdGeom.Xform.Define(self.stage, "/World/SolarSystem/Planets/Saturn/AxialTilt/TitanMoon")
                self._titan_translate_op = titan_xf.AddTranslateOp()
                titan_mesh = define_uv_sphere_mesh(
                    self.stage,
                    "/World/SolarSystem/Planets/Saturn/AxialTilt/TitanMoon/Globe",
                    0.72,
                    20,
                    40,
                    (0.86, 0.60, 0.26),
                )
                UsdShade.MaterialBindingAPI.Apply(titan_mesh.GetPrim()).Bind(mat_titan)

    # ------------------------------------------------------------------
    # 3. Asteroid Belt, Kuiper Belt & Oort Cloud (`UsdGeom.PointInstancer`)
    # ------------------------------------------------------------------
    def _build_belts_and_oort_instancers(self) -> None:
        UsdGeom.Xform.Define(self.stage, "/World/Belts")
        proto_Usd = Path(str(self.manifest.get("asteroid_prototypes_usd", ASSETS_USD_DIR / "asteroid_prototypes.usd")))
        rel_proto_path = proto_Usd.as_posix()

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
            scale_range=(0.24, 0.68),
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
            scale_range=(0.45, 1.15),
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
            scale_range=(0.95, 2.35),
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

        # Glowing Heliopause Crossing Waypoint Markers: Voyager 1 (2012) & Voyager 2 (2018)
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
                emissive_color=b_col,
                emissive_intensity=3200.0,
            )
            b_xf = UsdGeom.Xform.Define(self.stage, f"/World/Interstellar/CrossingMarkers/{b_name}")
            b_xf.AddTranslateOp().Set(Gf.Vec3d(*b_pos))
            sc_op = b_xf.AddScaleOp(opSuffix="pulse")
            sc_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))
            self._beacon_scale_ops.append(sc_op)

            core = UsdGeom.Sphere.Define(self.stage, f"/World/Interstellar/CrossingMarkers/{b_name}/Core")
            core.CreateRadiusAttr(3.2)
            UsdShade.MaterialBindingAPI.Apply(core.GetPrim()).Bind(mat_b)

            # Dual orthogonal beacon halo rings
            for r_idx, rot_xyz in enumerate(((0.0, 0.0, 0.0), (90.0, 0.0, 0.0))):
                ring_xf = UsdGeom.Xform.Define(self.stage, f"/World/Interstellar/CrossingMarkers/{b_name}/Halo_{r_idx}")
                ring_xf.AddRotateXYZOp().Set(Gf.Vec3f(*rot_xyz))
                ring_m = define_annulus_ring_mesh(
                    self.stage,
                    f"/World/Interstellar/CrossingMarkers/{b_name}/Halo_{r_idx}/Ring",
                    inner_radius=7.2,
                    outer_radius=8.4,
                    n_radial=2,
                    n_seg=48,
                )
                UsdShade.MaterialBindingAPI.Apply(ring_m.GetPrim()).Bind(mat_b)

        # Directional Solar-Wind vs. Interstellar Cosmic-Ray Particle Streams
        UsdGeom.Xform.Define(self.stage, "/World/Interstellar/ParticleStreams")
        mat_sw = create_omnipbr_material(
            self.stage,
            "/World/Materials/Mat_SolarWindIon",
            diffuse_color=(1.0, 0.85, 0.35),
            emissive_color=(1.0, 0.80, 0.30),
            emissive_intensity=1800.0,
        )
        mat_cr = create_omnipbr_material(
            self.stage,
            "/World/Materials/Mat_CosmicRayIon",
            diffuse_color=(1.0, 0.38, 0.88),
            emissive_color=(0.95, 0.32, 0.88),
            emissive_intensity=2000.0,
        )

        # Solar Wind radial plasma streaks inside Heliosphere (600 instances)
        sw_grp = UsdGeom.Xform.Define(self.stage, "/World/Interstellar/ParticleStreams/SolarWindGroup")
        self._stream_offset_ops["solar_wind"] = sw_grp.AddScaleOp(opSuffix="wave")
        sw_inst = UsdGeom.PointInstancer.Define(self.stage, "/World/Interstellar/ParticleStreams/SolarWindGroup/SolarWindStream")
        sw_proto = UsdGeom.Capsule.Define(self.stage, "/World/Interstellar/ParticleStreams/SolarWindGroup/SolarWindStream/IonStreak")
        sw_proto.CreateRadiusAttr(0.32)
        sw_proto.CreateHeightAttr(5.5)
        sw_proto.CreateAxisAttr("Z")
        UsdShade.MaterialBindingAPI.Apply(sw_proto.GetPrim()).Bind(mat_sw)
        sw_inst.GetPrototypesRel().AddTarget(sw_proto.GetPath())

        rng = np.random.default_rng(7701)
        n_sw = 550
        sw_pos, sw_quat, sw_sc = [], [], []
        from voyager_interstellar_sim.kinematics import quat_look_along_plus_z
        for _ in range(n_sw):
            d_vec = rng.normal(0.0, 1.0, size=3)
            d_vec /= max(np.linalg.norm(d_vec), 1e-6)
            r_sim = rng.uniform(50.0, 620.0)
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
        cr_proto.CreateRadiusAttr(0.38)
        cr_proto.CreateHeightAttr(8.5)
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
            r_sim = hp_r + rng.uniform(18.0, 210.0)
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
        titan_usd_path = Path(str(self.manifest.get("titan_iiie_centaur_usd", ASSETS_USD_DIR / "titan_iiie_centaur.usd"))).as_posix()

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

        # Spacecraft & Trajectory Splines
        UsdGeom.Xform.Define(self.stage, "/World/Spacecraft")
        UsdGeom.Scope.Define(self.stage, "/World/Spacecraft/TrajectoryCurves")

        voyager_usd_path = Path(str(self.manifest.get("voyager_articulated_usd", ASSETS_USD_DIR / "voyager_articulated.usd"))).as_posix()

        sc_configs = [
            (1, "Voyager1", (1.0, 0.78, 0.20)),
            (2, "Voyager2", (0.20, 0.90, 1.0)),
        ]
        for sc_id, sc_name, sc_color in sc_configs:
            # Smooth Catmull-Rom Trajectory Curve from Earth -> Giant Planets -> Heliopause -> Oort Cloud
            c_pts = sample_voyager_curve_sim(sc_id, num_samples=340)
            traj = UsdGeom.BasisCurves.Define(self.stage, f"/World/Spacecraft/TrajectoryCurves/{sc_name}_Trajectory")
            traj.CreateTypeAttr(UsdGeom.Tokens.linear)
            traj.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in c_pts]))
            traj.CreateCurveVertexCountsAttr(Vt.IntArray([len(c_pts)]))
            traj.CreateWidthsAttr(Vt.FloatArray([0.65] * len(c_pts)))
            traj.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*sc_color)]))

            sc_xf = UsdGeom.Xform.Define(self.stage, f"/World/Spacecraft/{sc_name}")
            sc_xf.GetPrim().GetReferences().AddReference(voyager_usd_path)
            t_op = sc_xf.AddTranslateOp(opSuffix="worldPos")
            r_op = sc_xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble, opSuffix="worldRot")
            s_op = sc_xf.AddScaleOp(opSuffix="worldScale")
            s_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))

            # Distant telemetry locator halo (scales up only in overview cameras so Voyager is always easy to spot!)
            mat_loc = create_omnipbr_material(
                self.stage,
                f"/World/Materials/Mat_{sc_name}_Locator",
                diffuse_color=sc_color,
                emissive_color=sc_color,
                emissive_intensity=3800.0,
            )
            loc_sphere = UsdGeom.Sphere.Define(self.stage, f"/World/Spacecraft/{sc_name}/TelemetryBeaconHalo")
            loc_sphere.CreateRadiusAttr(1.0)
            loc_scale_op = loc_sphere.AddScaleOp(opSuffix="haloScale")
            loc_scale_op.Set(Gf.Vec3f(0.01, 0.01, 0.01))
            UsdShade.MaterialBindingAPI.Apply(loc_sphere.GetPrim()).Bind(mat_loc)

            # Articulated boom deployment ops
            mag_op = _get_or_add_op(f"/World/Spacecraft/{sc_name}/MagnetometerBoomPivot", "rotateY", "deploy")
            rtg_op = _get_or_add_op(f"/World/Spacecraft/{sc_name}/RTGBoomPivot", "rotateY", "deploy")
            scan_op = _get_or_add_op(f"/World/Spacecraft/{sc_name}/ScanPlatformPivot", "rotateX", "deploy")

            # Live Earth radio link curve (2-point BasisCurve updated per frame)
            link_curve = UsdGeom.BasisCurves.Define(self.stage, f"/World/Spacecraft/TrajectoryCurves/{sc_name}_EarthRadioLink")
            link_curve.CreateTypeAttr(UsdGeom.Tokens.linear)
            link_curve.CreateCurveVertexCountsAttr(Vt.IntArray([2]))
            link_curve.CreateWidthsAttr(Vt.FloatArray([0.18, 0.18]))
            link_curve.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*sc_color)]))

            self._sc_ops[sc_id] = {
                "translate": t_op,
                "orient": r_op,
                "scale": s_op,
                "mag_deploy": mag_op,
                "rtg_deploy": rtg_op,
                "scan_deploy": scan_op,
                "halo_scale": loc_scale_op,
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
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.15, 35000.0))
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
        pulse_s = 1.0 + 0.16 * math.sin(self._pulse_phase * 2.2)
        for b_op in self._beacon_scale_ops:
            b_op.Set(Gf.Vec3f(pulse_s, pulse_s, pulse_s))

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

        # 4. Voyager 1 & Voyager 2 Transform, Boom Unfolding & Earth Radio Sightline
        wide_cam = active_camera in ("EclipticOverviewCam", "HeliopauseCrossSectionCam")
        halo_r = 5.2 if wide_cam else 0.01

        for sc_id, sc_st in ((1, state.voyager1), (2, state.voyager2)):
            ops = self._sc_ops[sc_id]
            ops["translate"].Set(Gf.Vec3d(*sc_st.pos_sim))  # type: ignore
            qw, qx, qy, qz = sc_st.quat_wxyz
            ops["orient"].Set(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)))  # type: ignore

            # Compact stowed scale inside Centaur fairing before separation, 1.0x after separation
            sc_scale = 0.24 + 0.76 * sc_st.boom_deploy_01 if state.rocket.visible else 1.0
            ops["scale"].Set(Gf.Vec3f(sc_scale, sc_scale, sc_scale))  # type: ignore

            stowed_frac = 1.0 - sc_st.boom_deploy_01
            ops["mag_deploy"].Set(float(-84.0 * stowed_frac))  # type: ignore
            ops["rtg_deploy"].Set(float(78.0 * stowed_frac))  # type: ignore
            ops["scan_deploy"].Set(float(-72.0 * stowed_frac))  # type: ignore
            ops["halo_scale"].Set(Gf.Vec3f(halo_r, halo_r, halo_r))  # type: ignore

            link_curve: UsdGeom.BasisCurves = ops["radio_link"]  # type: ignore
            link_curve.GetPointsAttr().Set(
                Vt.Vec3fArray([Gf.Vec3f(*sc_st.pos_sim), Gf.Vec3f(*state.earth_pos_sim)])
            )

        # 5. Aim Solar Directional Key Light from Sun (0,0,0) toward active Voyager
        active_sc = state.voyager1 if active_spacecraft == 1 else state.voyager2
        if self._sun_key_rot_op is not None:
            sc_p = active_sc.pos_sim
            # DistantLight shines along local -Z, so make_camera_transform_matrix((0,0,0), sc_p) aims -Z straight at Voyager!
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
        vel_dir = np.array(sc.vel_dir, dtype=np.float64)
        vel_dir /= max(np.linalg.norm(vel_dir), 1e-6)

        to_earth = earth_pos - sc_pos
        to_earth_u = to_earth / max(np.linalg.norm(to_earth), 1e-6)
        side = np.cross(np.array([0.0, 0.0, 1.0]), to_earth_u)
        if np.linalg.norm(side) < 1e-4:
            side = np.array([1.0, 0.0, 0.0])
        side /= np.linalg.norm(side)

        # 1. Voyager Chase Cam: frames Voyager's 3.7m dish, booms, AND the Solar System / flyby planet ahead!
        chase_eye = sc_pos - to_earth_u * 8.2 + side * 6.4 + np.array([0.0, 0.0, 3.6])
        chase_tgt = sc_pos + to_earth_u * 2.2
        self._camera_ops["VoyagerChaseCam"].Set(
            make_camera_transform_matrix(tuple(chase_eye), tuple(chase_tgt))
        )

        # 2. Rocket Staging Cam: tracks Titan IIIE ascent, SRB/Fairing jettison, and Voyager separation
        if state.rocket.visible:
            rk_pos = np.array(state.rocket.root_pos, dtype=np.float64)
            rk_rad = rk_pos / max(np.linalg.norm(rk_pos), 1e-6)
            rk_side = np.cross(rk_rad, np.array([0.0, 0.0, 1.0]))
            rk_side /= max(np.linalg.norm(rk_side), 1e-6)
            mid_tgt = 0.45 * rk_pos + 0.55 * sc_pos
            stg_eye = mid_tgt + rk_side * 10.5 + np.array([0.0, 0.0, 3.2])
            self._camera_ops["RocketStagingCam"].Set(
                make_camera_transform_matrix(tuple(stg_eye), tuple(mid_tgt))
            )
        else:
            # Post-staging: dramatic front-quarter Scan Platform / Golden Record inspection cam
            insp_eye = sc_pos + to_earth_u * 6.5 + side * 5.2 + np.array([0.0, 0.0, 2.2])
            self._camera_ops["RocketStagingCam"].Set(
                make_camera_transform_matrix(tuple(insp_eye), tuple(sc_pos))
            )

        # 3. Solar System Ecliptic Overview Cam (zooms out smoothly if Voyager enters Oort Cloud)
        sc_r = float(np.linalg.norm(sc_pos))
        overview_dist = max(860.0, sc_r * 1.65)
        ov_eye = (0.0, -overview_dist * 0.78, overview_dist * 0.74)
        self._camera_ops["EclipticOverviewCam"].Set(
            make_camera_transform_matrix(ov_eye, (0.0, 0.0, 0.0), up_world=(0.0, 0.0, 1.0))
        )

        # 4. Heliopause Cross-Section Cam (perpendicular to HELIO_NOSE_DIR so nose + heliotail + V1/V2 crossings frame cleanly)
        nose = np.asarray(HELIO_NOSE_DIR, dtype=np.float64)
        nose /= np.linalg.norm(nose)
        perp = np.cross(nose, np.array([0.0, 0.0, 1.0]))
        perp /= max(np.linalg.norm(perp), 1e-6)
        hp_center = -nose * 110.0
        hp_eye = hp_center + perp * 1450.0 + np.array([0.0, 0.0, 210.0])
        self._camera_ops["HeliopauseCrossSectionCam"].Set(
            make_camera_transform_matrix(tuple(hp_eye), tuple(hp_center), up_world=(0.0, 0.0, 1.0))
        )

        # 5. Free Orbit Cam: initialized once so user viewport camera manipulation is never overwritten
        if not self._free_orbit_initialized:
            free_eye = sc_pos + np.array([-16.0, -18.0, 11.0])
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
