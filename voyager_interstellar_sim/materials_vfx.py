"""MDL Shader Builders (`OmniPBR`, `OmniPBR_Opacity`, `OmniPBR_ClearCoat`) and Interstellar
Boundary / Corona / Ring / Particle-Stream VFX Geometry Generators.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from voyager_interstellar_sim.config import (
    HELIO_NOSE_DIR,
    KIT_MDL_BASE_DIR,
    TEXTURES_DIR,
    au_vec_to_sim,
)
from voyager_interstellar_sim.kinematics import heliosphere_boundary_radius_au


def _mdl_asset_path(mdl_filename: str) -> Sdf.AssetPath:
    full = KIT_MDL_BASE_DIR / mdl_filename
    if full.exists():
        return Sdf.AssetPath(full.as_posix())
    return Sdf.AssetPath(mdl_filename)


def ensure_vfx_textures() -> Dict[str, str]:
    """Ensure specialized alpha/emissive VFX maps (corona, heliopause plasma drapery) exist on disk."""
    from PIL import Image

    TEXTURES_DIR.mkdir(parents=True, exist_ok=True)
    out: Dict[str, str] = {}

    # 1. Heliopause plasma compression & magnetic draping map (UV: u=azimuth, v=0 at Nose -> 1 at Tail)
    hp_col_path = TEXTURES_DIR / "vfx_heliopause_color.png"
    hp_op_path = TEXTURES_DIR / "vfx_heliopause_opacity.png"
    if not hp_col_path.exists() or not hp_op_path.exists():
        W, H = 512, 256
        u = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
        v = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]  # v=0 nose, v=1 tail

        nose_glow = np.exp(-((v / 0.28) ** 2))
        filaments = 0.5 + 0.5 * np.sin(36.0 * v + 4.0 * np.sin(2.0 * np.pi * u))
        spiral = 0.5 + 0.5 * np.cos(16.0 * 2.0 * np.pi * u + 10.0 * v)
        band = 0.35 + 0.45 * filaments + 0.20 * spiral

        rgb = np.zeros((H, W, 3), dtype=np.float32)
        # Electric cyan-blue at nose -> deep violet-indigo along heliotail
        rgb[..., 0] = np.clip((0.18 * nose_glow + 0.36 * (1.0 - nose_glow)) * band, 0.0, 1.0)
        rgb[..., 1] = np.clip((0.85 * nose_glow + 0.28 * (1.0 - nose_glow)) * band, 0.0, 1.0)
        rgb[..., 2] = np.clip((1.00 * nose_glow + 0.88 * (1.0 - nose_glow)) * band, 0.0, 1.0)

        # Keep interior translucent (0.05..0.22) so planets & spacecraft remain crystal-clear through the shell!
        op_val = (0.20 * nose_glow + 0.07 * (1.0 - 0.65 * v)) * (0.55 + 0.45 * filaments)
        op_val = np.clip(op_val, 0.02, 0.26)
        op_rgb = np.stack([op_val, op_val, op_val], axis=-1)

        Image.fromarray((rgb * 255.0).astype(np.uint8)).save(str(hp_col_path), format="PNG")
        Image.fromarray((op_rgb * 255.0).astype(np.uint8)).save(str(hp_op_path), format="PNG")

    out["heliopause_color"] = hp_col_path.as_posix()
    out["heliopause_opacity"] = hp_op_path.as_posix()

    # 2. Solar Corona streamer opacity map
    cor_op_path = TEXTURES_DIR / "vfx_corona_opacity.png"
    if not cor_op_path.exists():
        W, H = 512, 256
        u = np.linspace(0.0, 2.0 * np.pi, W, endpoint=False, dtype=np.float32)[None, :]
        v = np.linspace(-0.5 * np.pi, 0.5 * np.pi, H, dtype=np.float32)[:, None]
        eq_belt = np.exp(-((v / 0.55) ** 2))
        streamers = 0.55 + 0.45 * np.sin(11.0 * u + 3.0 * np.cos(4.0 * v))
        op = np.clip(0.26 * eq_belt * streamers + 0.06, 0.03, 0.34)
        op_rgb = np.stack([op, op, op], axis=-1)
        Image.fromarray((op_rgb * 255.0).astype(np.uint8)).save(str(cor_op_path), format="PNG")
    out["corona_opacity"] = cor_op_path.as_posix()

    return out


def create_omnipbr_material(
    stage: Usd.Stage,
    mat_path: str,
    diffuse_color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
    diffuse_texture: Optional[str] = None,
    bump_texture: Optional[str] = None,
    roughness: float = 0.55,
    metallic: float = 0.0,
    emissive_color: Optional[Tuple[float, float, float]] = None,
    emissive_texture: Optional[str] = None,
    emissive_intensity: float = 0.0,
    opacity_constant: Optional[float] = None,
    opacity_texture: Optional[str] = None,
) -> UsdShade.Material:
    """Create an Omniverse RTX MDL material (`OmniPBR` or `OmniPBR_Opacity`) with verified file asset paths."""
    use_opacity = (opacity_constant is not None) or (opacity_texture is not None)
    mdl_file = "OmniPBR_Opacity.mdl" if use_opacity else "OmniPBR.mdl"
    sub_id = "OmniPBR_Opacity" if use_opacity else "OmniPBR"

    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(_mdl_asset_path(mdl_file), "mdl")
    shader.SetSourceAssetSubIdentifier(sub_id, "mdl")

    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse_color))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(metallic))

    if diffuse_texture:
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(diffuse_texture))

    if bump_texture:
        shader.CreateInput("bump_factor", Sdf.ValueTypeNames.Float).Set(0.65)
        shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(bump_texture))

    if emissive_intensity > 0.0 and (emissive_color is not None or emissive_texture is not None):
        shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
        ecol = emissive_color if emissive_color is not None else diffuse_color
        shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*ecol))
        shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(float(emissive_intensity))
        if emissive_texture:
            shader.CreateInput("emissive_color_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(emissive_texture))

    if use_opacity:
        shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
        if opacity_constant is not None:
            shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(float(opacity_constant))
        if opacity_texture is not None:
            shader.CreateInput("enable_opacity_texture", Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(opacity_texture))
            shader.CreateInput("opacity_mode", Sdf.ValueTypeNames.Int).Set(1)  # mono_average

    mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")

    # Add lightweight UsdPreviewSurface fallback so Storm / non-MDL tools also display colors/opacity
    prev = UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
    prev.CreateIdAttr("UsdPreviewSurface")
    prev.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse_color))
    prev.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    prev.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    if emissive_intensity > 0.0 and emissive_color is not None:
        prev.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive_color))
    if opacity_constant is not None:
        prev.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity_constant))
    mat.CreateSurfaceOutput().ConnectToSource(prev.ConnectableAPI(), "surface")

    return mat


# ---------------------------------------------------------------------------
# High-Tessellation UV Sphere, Saturn Ring Annulus & Heliopause Teardrop Meshes
# ---------------------------------------------------------------------------
def define_uv_sphere_mesh(
    stage: Usd.Stage,
    prim_path: str,
    radius: float,
    n_lat: int = 36,
    n_lon: int = 72,
    display_color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
) -> UsdGeom.Mesh:
    """Create a Z-up textured equirectangular UV sphere mesh with smooth vertex normals."""
    pts: List[Gf.Vec3f] = []
    norms: List[Gf.Vec3f] = []
    uvs: List[Gf.Vec2f] = []

    for ilat in range(n_lat + 1):
        v = ilat / n_lat
        lat = (v - 0.5) * math.pi
        clat = math.cos(lat)
        slat = math.sin(lat)
        for ilon in range(n_lon + 1):
            u = ilon / n_lon
            lon = (u - 0.5) * 2.0 * math.pi
            nx = clat * math.cos(lon)
            ny = clat * math.sin(lon)
            nz = slat
            pts.append(Gf.Vec3f(float(radius * nx), float(radius * ny), float(radius * nz)))
            norms.append(Gf.Vec3f(float(nx), float(ny), float(nz)))
            uvs.append(Gf.Vec2f(float(u), float(v)))

    counts: List[int] = []
    indices: List[int] = []
    stride = n_lon + 1
    for ilat in range(n_lat):
        for ilon in range(n_lon):
            i0 = ilat * stride + ilon
            i1 = i0 + 1
            i2 = (ilat + 1) * stride + ilon + 1
            i3 = (ilat + 1) * stride + ilon
            counts.append(4)
            indices.extend([i0, i1, i2, i3])

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateNormalsAttr(Vt.Vec3fArray(norms))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*display_color)]))

    primvars = UsdGeom.PrimvarsAPI(mesh)
    st = primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray(uvs))
    return mesh


def define_annulus_ring_mesh(
    stage: Usd.Stage,
    prim_path: str,
    inner_radius: float,
    outer_radius: float,
    n_radial: int = 24,
    n_seg: int = 128,
) -> UsdGeom.Mesh:
    """Create a double-sided planetary ring annulus mesh in the XY plane with radial U in [0, 1]."""
    pts: List[Gf.Vec3f] = []
    uvs: List[Gf.Vec2f] = []
    norms: List[Gf.Vec3f] = []
    for ir in range(n_radial + 1):
        u = ir / n_radial
        r = inner_radius + u * (outer_radius - inner_radius)
        for iseg in range(n_seg + 1):
            v = iseg / n_seg
            ang = 2.0 * math.pi * v
            pts.append(Gf.Vec3f(float(r * math.cos(ang)), float(r * math.sin(ang)), 0.0))
            uvs.append(Gf.Vec2f(float(u), float(v)))
            norms.append(Gf.Vec3f(0.0, 0.0, 1.0))

    counts: List[int] = []
    indices: List[int] = []
    stride = n_seg + 1
    for ir in range(n_radial):
        for iseg in range(n_seg):
            i0 = ir * stride + iseg
            i1 = i0 + 1
            i2 = (ir + 1) * stride + iseg + 1
            i3 = (ir + 1) * stride + iseg
            counts.append(4)
            indices.extend([i0, i1, i2, i3])

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateNormalsAttr(Vt.Vec3fArray(norms))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    primvars = UsdGeom.PrimvarsAPI(mesh)
    st = primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray(uvs))
    return mesh


def define_heliosphere_teardrop_shell(
    stage: Usd.Stage,
    prim_path: str,
    nose_au: float = 116.8,
    tail_stretch: float = 2.55,
    n_polar: int = 44,
    n_azim: int = 72,
    wireframe_curves_path: Optional[str] = None,
    wire_color: Tuple[float, float, float] = (0.25, 0.85, 1.0),
) -> UsdGeom.Mesh:
    """Build an ISM-deformed teardrop Termination Shock / Heliopause surface in piecewise-log scene space."""
    nose = np.asarray(HELIO_NOSE_DIR, dtype=np.float64)
    nose /= np.linalg.norm(nose)
    # Orthonormal basis (e1, e2, nose)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    e1 = np.cross(ref, nose)
    e1 /= max(np.linalg.norm(e1), 1e-8)
    e2 = np.cross(nose, e1)

    pts: List[Gf.Vec3f] = []
    uvs: List[Gf.Vec2f] = []
    grid_pts_sim = np.zeros((n_polar + 1, n_azim + 1, 3), dtype=np.float32)

    for ip in range(n_polar + 1):
        v = ip / n_polar  # 0 at Nose (+nose), 1 at Tail (-nose)
        theta = v * math.pi
        cth, sth = math.cos(theta), math.sin(theta)
        for ia in range(n_azim + 1):
            u = ia / n_azim
            phi = 2.0 * math.pi * u
            d_unit = cth * nose + sth * (math.cos(phi) * e1 + math.sin(phi) * e2)
            r_au = heliosphere_boundary_radius_au(d_unit, nose_au=nose_au, tail_stretch=tail_stretch)
            p_sim = au_vec_to_sim((float(d_unit[0] * r_au), float(d_unit[1] * r_au), float(d_unit[2] * r_au)))
            pts.append(Gf.Vec3f(*p_sim))
            uvs.append(Gf.Vec2f(float(u), float(v)))
            grid_pts_sim[ip, ia] = p_sim

    counts: List[int] = []
    indices: List[int] = []
    stride = n_azim + 1
    for ip in range(n_polar):
        for ia in range(n_azim):
            i0 = ip * stride + ia
            i1 = i0 + 1
            i2 = (ip + 1) * stride + ia + 1
            i3 = (ip + 1) * stride + ia
            counts.append(4)
            indices.extend([i0, i1, i2, i3])

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*wire_color)]))
    primvars = UsdGeom.PrimvarsAPI(mesh)
    st = primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray(uvs))

    # Optional magnetic-field contour curves on the shell so cross-sections read with scientific precision!
    if wireframe_curves_path:
        curves = UsdGeom.BasisCurves.Define(stage, wireframe_curves_path)
        curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        c_pts: List[Gf.Vec3f] = []
        c_cnts: List[int] = []
        # 6 proportional polar rib rings + 8 meridians
        rib_indices = [min(n_polar, max(1, int(round(n_polar * f)))) for f in (0.14, 0.28, 0.42, 0.56, 0.72, 0.88)]
        for ip in rib_indices:
            for ia in range(n_azim + 1):
                p = grid_pts_sim[ip, ia]
                c_pts.append(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))
            c_cnts.append(n_azim + 1)
        for ia in range(0, n_azim, max(1, n_azim // 8)):
            for ip in range(n_polar + 1):
                p = grid_pts_sim[ip, ia]
                c_pts.append(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))
            c_cnts.append(n_polar + 1)
        curves.CreatePointsAttr(Vt.Vec3fArray(c_pts))
        curves.CreateCurveVertexCountsAttr(Vt.IntArray(c_cnts))
        curves.CreateWidthsAttr(Vt.FloatArray([0.65] * len(c_pts)))
        curves.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*wire_color)]))

    return mesh
