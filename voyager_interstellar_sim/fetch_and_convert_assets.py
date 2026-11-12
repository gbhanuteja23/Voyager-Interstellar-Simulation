"""Automated Free Asset Acquisition, OpenUSD Conversion, and Procedural Fallback Builder.

Run standalone via:
    D:\\0-NVIDIA-Omniverse\\isaac-sim\\python.bat ./voyager_interstellar_sim/fetch_and_convert_assets.py

Responsibilities:
1. Scans local SimReady directory (`simready-central`) and Kit core MDL library, writing
   an asset manifest (`assets_usd/asset_manifest.json`).
2. Queries NASA's official `NASA-3D-Resources` repository (`https://github.com/nasa/NASA-3D-Resources`)
   and public-domain / CC-BY planetary texture repositories to download:
   - Voyager 3D spacecraft model (`.glb` / `.obj` / `.stl`)
   - Launch vehicle 3D model if available
   - Equirectangular planetary albedo/bump/ring/starfield maps
3. Converts downloaded non-USD meshes into `.usd` files in `./voyager_interstellar_sim/assets_usd/`
   using Isaac Sim's `omni.kit.asset_converter` with embedded textures and Z-up normalization.
4. Synthesizes high-resolution equirectangular fallback textures for any map that fails to download,
   and builds articulated OpenUSD assemblies (`voyager_articulated.usd`, `titan_iiie_centaur.usd`,
   `asteroid_prototypes.usd`) so the simulation pipeline never blocks and supports full Phase 1
   boom deployment and multi-stage rocket separation.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure package root is importable when executed directly
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from voyager_interstellar_sim.config import (
    ASSETS_RAW_DIR,
    ASSETS_USD_DIR,
    ISAAC_SIM_ROOT,
    KIT_MDL_BASE_DIR,
    SIMREADY_CENTRAL_DIR,
    TEXTURES_DIR,
)

# ---------------------------------------------------------------------------
# Public-Domain / CC-BY Download Sources
# ---------------------------------------------------------------------------
NASA_TREE_API_URLS: List[str] = [
    "https://api.github.com/repos/nasa/NASA-3D-Resources/git/trees/master?recursive=1",
    "https://api.github.com/repos/nasa/NASA-3D-Resources/git/trees/main?recursive=1",
]

DIRECT_NASA_VOYAGER_CANDIDATES: List[str] = [
    "https://raw.githubusercontent.com/nasa/NASA-3D-Resources/master/3D%20Models/Voyager/Voyager.glb",
    "https://raw.githubusercontent.com/nasa/NASA-3D-Resources/master/3D%20Models/Voyager%20Probe/Voyager.glb",
]

# Public-domain NASA / Wikimedia / Three.js planetary texture URLs (gracefully falls back to procedural)
PLANETARY_TEXTURE_URLS: Dict[str, List[str]] = {
    "earth": [
        "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_atmos_2048.jpg",
    ],
    "earth_bump": [
        "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_normal_2048.jpg",
    ],
    "earth_clouds": [
        "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_clouds_1024.png",
    ],
    "mercury": [
        "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/moon_1024.jpg",
    ],
    "jupiter": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/Jupiter.jpg/1024px-Jupiter.jpg",
    ],
    "mars": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/OSIRIS_Mars_true_color.jpg/1024px-OSIRIS_Mars_true_color.jpg",
    ],
}


def _http_get_bytes(url: str, timeout: float = 15.0) -> Optional[bytes]:
    """Download URL bytes safely with standard User-Agent and timeout."""
    ctx = ssl.create_default_context()
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "VoyagerInterstellarSim/1.0 (NVIDIA-IsaacSim-OpenUSD; Educational-Research)",
            "Accept": "*/*",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            if resp.status == 200:
                return resp.read()
    except Exception as exc:
        print(f"[AssetFetcher] Notice: URL skipped ({url}): {exc}")
    return None


def scan_local_simready_and_mdl() -> Dict[str, object]:
    """Scan local `simready-central` and Isaac Sim core MDL directories."""
    local_usd_files: List[str] = []
    if SIMREADY_CENTRAL_DIR.exists():
        for ext in ("*.usd", "*.usda", "*.usdc", "*.mdl"):
            for p in SIMREADY_CENTRAL_DIR.rglob(ext):
                local_usd_files.append(str(p))

    mdl_files: Dict[str, str] = {}
    if KIT_MDL_BASE_DIR.exists():
        for p in sorted(KIT_MDL_BASE_DIR.glob("*.mdl")):
            mdl_files[p.stem] = p.as_posix()

    report = {
        "isaac_sim_root": ISAAC_SIM_ROOT.as_posix(),
        "simready_central_dir": SIMREADY_CENTRAL_DIR.as_posix(),
        "simready_central_exists": SIMREADY_CENTRAL_DIR.exists(),
        "simready_local_usd_count": len(local_usd_files),
        "simready_local_usd_files": local_usd_files[:25],
        "simready_note": (
            "simready-central on disk is the SimReady Central USD Search HF Space service repo "
            "(no local .usd/.mdl rock/space assets on disk); using Kit core Base MDLs + NASA/procedural USD assets."
            if len(local_usd_files) == 0
            else "Indexed local SimReady USD/MDL files."
        ),
        "kit_mdl_shaders": mdl_files,
    }
    print(
        f"[AssetFetcher] Local scan: simready_local_usd={len(local_usd_files)}, "
        f"kit_base_mdls={len(mdl_files)}"
    )
    return report


def discover_and_download_nasa_models() -> Dict[str, Path]:
    """Search NASA-3D-Resources Git tree for Voyager and rocket meshes and download them."""
    ASSETS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: Dict[str, Path] = {}

    # Check cache first
    for cached_key, exts in [("voyager", (".glb", ".obj", ".fbx", ".stl")), ("rocket", (".glb", ".obj", ".fbx"))]:
        for ext in exts:
            candidate = ASSETS_RAW_DIR / f"nasa_{cached_key}{ext}"
            if candidate.exists() and candidate.stat().st_size > 1024:
                downloaded[cached_key] = candidate
                break

    if "voyager" in downloaded:
        print(f"[AssetFetcher] Using cached NASA Voyager raw mesh: {downloaded['voyager']}")
        return downloaded

    tree_entries: List[Dict[str, object]] = []
    branch_used = "master"
    for api_url in NASA_TREE_API_URLS:
        raw = _http_get_bytes(api_url, timeout=18.0)
        if raw:
            try:
                payload = json.loads(raw.decode("utf-8"))
                if isinstance(payload.get("tree"), list):
                    tree_entries = payload["tree"]
                    branch_used = "main" if "/main" in api_url else "master"
                    print(f"[AssetFetcher] Indexed {len(tree_entries)} entries in NASA-3D-Resources ({branch_used}).")
                    break
            except Exception as exc:
                print(f"[AssetFetcher] Tree JSON parse warning: {exc}")

    def _pick_best_entry(keywords: List[str], preferred_exts: Tuple[str, ...]) -> Optional[str]:
        matches: List[Tuple[int, str]] = []
        for item in tree_entries:
            if item.get("type") != "blob":
                continue
            path_str = str(item.get("path", ""))
            lower = path_str.lower()
            ext = Path(lower).suffix
            if ext not in preferred_exts:
                continue
            if any(kw in lower for kw in keywords):
                prio = preferred_exts.index(ext)
                matches.append((prio, path_str))
        matches.sort(key=lambda x: (x[0], len(x[1])))
        return matches[0][1] if matches else None

    # 1. Voyager spacecraft model
    voyager_rel = _pick_best_entry(["voyager"], (".glb", ".gltf", ".obj", ".fbx", ".stl"))
    if voyager_rel:
        encoded = "/".join(urllib.parse.quote(seg) for seg in voyager_rel.split("/"))
        raw_url = f"https://raw.githubusercontent.com/nasa/NASA-3D-Resources/{branch_used}/{encoded}"
        data = _http_get_bytes(raw_url, timeout=30.0)
        if data and len(data) > 1024:
            ext = Path(voyager_rel).suffix.lower()
            dest = ASSETS_RAW_DIR / f"nasa_voyager{ext}"
            dest.write_bytes(data)
            downloaded["voyager"] = dest
            print(f"[AssetFetcher] Downloaded NASA Voyager model ({len(data) // 1024} KB) -> {dest.name}")

    if "voyager" not in downloaded:
        for url in DIRECT_NASA_VOYAGER_CANDIDATES:
            data = _http_get_bytes(url, timeout=20.0)
            if data and len(data) > 1024:
                ext = Path(urllib.parse.urlparse(url).path).suffix.lower() or ".glb"
                dest = ASSETS_RAW_DIR / f"nasa_voyager{ext}"
                dest.write_bytes(data)
                downloaded["voyager"] = dest
                print(f"[AssetFetcher] Downloaded direct NASA Voyager model -> {dest.name}")
                break

    # 2. Launch Vehicle (Titan IIIE / Centaur) if available in NASA-3D-Resources
    rocket_rel = _pick_best_entry(["titan iiie", "titan_iiie", "centaur"], (".glb", ".obj", ".fbx"))
    if rocket_rel:
        encoded = "/".join(urllib.parse.quote(seg) for seg in rocket_rel.split("/"))
        raw_url = f"https://raw.githubusercontent.com/nasa/NASA-3D-Resources/{branch_used}/{encoded}"
        data = _http_get_bytes(raw_url, timeout=30.0)
        if data and len(data) > 1024:
            ext = Path(rocket_rel).suffix.lower()
            dest = ASSETS_RAW_DIR / f"nasa_rocket{ext}"
            dest.write_bytes(data)
            downloaded["rocket"] = dest
            print(f"[AssetFetcher] Downloaded NASA launch vehicle model ({rocket_rel}) -> {dest.name}")

    return downloaded


# ---------------------------------------------------------------------------
# Procedural & Downloaded Planetary Equirectangular Texture Suite
# ---------------------------------------------------------------------------
def _fbm_2d(lon_grid: np.ndarray, lat_grid: np.ndarray, seed: int, octaves: int = 5) -> np.ndarray:
    """Seamless 3D-spherical multi-octave trigonometric/hash noise on (lon, lat)."""
    rng = np.random.default_rng(seed)
    cx = np.cos(lat_grid) * np.cos(lon_grid)
    cy = np.cos(lat_grid) * np.sin(lon_grid)
    cz = np.sin(lat_grid)

    val = np.zeros_like(lon_grid, dtype=np.float32)
    amp = 0.5
    freq = 2.2
    for _ in range(octaves):
        kx, ky, kz = rng.normal(0.0, 1.0, size=(3, 6)).astype(np.float32)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=6).astype(np.float32)
        layer = np.zeros_like(val)
        for j in range(6):
            arg = freq * (kx[j] * cx + ky[j] * cy + kz[j] * cz) + phases[j]
            layer += np.sin(arg)
        val += amp * (layer / 6.0)
        freq *= 2.07
        amp *= 0.52
    # Normalize to [0, 1]
    vmin, vmax = float(val.min()), float(val.max())
    if vmax - vmin > 1e-6:
        val = (val - vmin) / (vmax - vmin)
    return val


def _save_png(arr_uint8: np.ndarray, path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr_uint8)
    img.save(str(path), format="PNG", optimize=False)
    return path


def build_or_fetch_textures() -> Dict[str, str]:
    """Download public planetary maps where available and synthesize crisp equirectangular maps for the rest."""
    from PIL import Image
    import io

    TEXTURES_DIR.mkdir(parents=True, exist_ok=True)
    tex_paths: Dict[str, str] = {}

    # Try downloading public-domain textures first (re-encoded cleanly as local PNGs so USD never has format issues)
    for key, urls in PLANETARY_TEXTURE_URLS.items():
        out_png = TEXTURES_DIR / f"{key}.png"
        if out_png.exists() and out_png.stat().st_size > 4096:
            tex_paths[key] = out_png.as_posix()
            continue
        for url in urls:
            raw = _http_get_bytes(url, timeout=15.0)
            if not raw:
                continue
            try:
                im = Image.open(io.BytesIO(raw))
                if key == "earth_clouds":
                    im = im.convert("RGBA")
                else:
                    im = im.convert("RGB")
                if im.width > 2048:
                    im = im.resize((2048, 1024), Image.Resampling.LANCZOS)
                im.save(str(out_png), format="PNG")
                tex_paths[key] = out_png.as_posix()
                print(f"[AssetFetcher] Cached public planetary texture: {out_png.name} ({im.width}x{im.height})")
                break
            except Exception as exc:
                print(f"[AssetFetcher] Image decode fallback for {key}: {exc}")

    # Common equirectangular coordinate grid (1024x512) for fast, crisp procedural maps
    W, H = 1024, 512
    u = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)[None, :]
    v = np.linspace(-0.5 * np.pi, 0.5 * np.pi, H, dtype=np.float32)[:, None]
    lon = np.broadcast_to(u, (H, W))
    lat = np.broadcast_to(v, (H, W))

    def _ensure_rgb(key: str, builder_fn) -> None:
        out_png = TEXTURES_DIR / f"{key}.png"
        if key not in tex_paths:
            if not (out_png.exists() and out_png.stat().st_size > 4096):
                rgb = np.clip(builder_fn() * 255.0, 0, 255).astype(np.uint8)
                _save_png(rgb, out_png)
                print(f"[AssetFetcher] Synthesized equirectangular map: {out_png.name}")
            tex_paths[key] = out_png.as_posix()

    # 1. Sun photosphere granulation
    def _sun_tex() -> np.ndarray:
        g1 = _fbm_2d(lon, lat, seed=101, octaves=6)
        g2 = _fbm_2d(lon * 2.0, lat * 2.0, seed=102, octaves=4)
        spots = np.clip((g1 - 0.18) / 0.82, 0.15, 1.0)
        limb = 0.88 + 0.12 * np.cos(lat)
        r = np.clip(1.00 * spots * limb + 0.12 * g2, 0.0, 1.0)
        g = np.clip(0.68 * (spots ** 1.15) + 0.18 * g2, 0.0, 1.0)
        b = np.clip(0.18 * (spots ** 1.8) + 0.08 * g2, 0.0, 1.0)
        return np.stack([r, g, b], axis=-1)

    _ensure_rgb("sun", _sun_tex)

    # 2. Mercury cratered regolith
    def _mercury_tex() -> np.ndarray:
        n1 = _fbm_2d(lon, lat, seed=201, octaves=6)
        n2 = _fbm_2d(lon, lat, seed=202, octaves=5)
        craters = np.abs(np.sin(14.0 * n1 + 6.0 * n2))
        lum = 0.34 + 0.36 * n1 - 0.12 * craters
        return np.stack([lum * 1.03, lum * 0.99, lum * 0.94], axis=-1)

    _ensure_rgb("mercury", _mercury_tex)

    # 3. Venus super-rotating clouds
    def _venus_tex() -> np.ndarray:
        n = _fbm_2d(lon * 0.5, lat, seed=301, octaves=5)
        chevrons = np.sin(8.0 * lat + 2.5 * np.cos(lon + 3.0 * n))
        band = 0.72 + 0.14 * chevrons + 0.12 * n
        return np.stack([band * 1.02, band * 0.86, band * 0.58], axis=-1)

    _ensure_rgb("venus", _venus_tex)

    # 4. Earth + Bump + Clouds (fallback if offline)
    def _earth_tex() -> np.ndarray:
        elev = _fbm_2d(lon, lat, seed=401, octaves=6)
        moist = _fbm_2d(lon, lat, seed=402, octaves=4)
        land = elev > 0.52
        ice = (np.abs(lat) > 1.18 + 0.08 * elev)
        r = np.where(land, 0.18 + 0.35 * (1.0 - moist), 0.05 + 0.08 * elev)
        g = np.where(land, 0.36 + 0.25 * moist, 0.18 + 0.22 * elev)
        b = np.where(land, 0.14 + 0.10 * moist, 0.48 + 0.32 * elev)
        rgb = np.stack([r, g, b], axis=-1)
        rgb[ice] = np.array([0.93, 0.96, 0.99], dtype=np.float32)
        return rgb

    _ensure_rgb("earth", _earth_tex)

    def _earth_bump_tex() -> np.ndarray:
        bmap = _fbm_2d(lon, lat, seed=401, octaves=6)
        return np.stack([bmap, bmap, bmap], axis=-1)

    _ensure_rgb("earth_bump", _earth_bump_tex)

    clouds_path = TEXTURES_DIR / "earth_clouds.png"
    if "earth_clouds" not in tex_paths:
        if not (clouds_path.exists() and clouds_path.stat().st_size > 4096):
            cnoise = _fbm_2d(lon, lat, seed=409, octaves=5)
            alpha = np.clip((cnoise - 0.47) * 2.3, 0.0, 0.88)
            rgba = np.ones((H, W, 4), dtype=np.float32)
            rgba[..., 3] = alpha
            _save_png((rgba * 255.0).astype(np.uint8), clouds_path)
        tex_paths["earth_clouds"] = clouds_path.as_posix()

    # 5. Mars rust desert + dark albedo + polar caps
    def _mars_tex() -> np.ndarray:
        n1 = _fbm_2d(lon, lat, seed=501, octaves=6)
        n2 = _fbm_2d(lon, lat, seed=502, octaves=4)
        basalt = np.clip((n2 - 0.55) * 2.5, 0.0, 0.65)
        r = (0.76 + 0.18 * n1) * (1.0 - 0.45 * basalt)
        g = (0.37 + 0.14 * n1) * (1.0 - 0.48 * basalt)
        b = (0.21 + 0.09 * n1) * (1.0 - 0.42 * basalt)
        rgb = np.stack([r, g, b], axis=-1)
        caps = np.abs(lat) > (1.34 + 0.06 * n1)
        rgb[caps] = np.array([0.92, 0.90, 0.88], dtype=np.float32)
        return rgb

    _ensure_rgb("mars", _mars_tex)

    # 6. Jupiter banded atmosphere + Great Red Spot
    def _jupiter_tex() -> np.ndarray:
        turb = _fbm_2d(lon * 0.6, lat * 2.2, seed=601, octaves=6)
        y = lat + 0.055 * (turb - 0.5)
        belts = 0.5 + 0.5 * np.sin(22.0 * y) + 0.25 * np.cos(41.0 * y)
        belts = np.clip(belts / 1.25, 0.0, 1.0)
        r = 0.58 + 0.34 * belts + 0.06 * turb
        g = 0.42 + 0.42 * belts + 0.06 * turb
        b = 0.28 + 0.44 * belts + 0.05 * turb
        # Great Red Spot at lat = -22 deg, lon = 0.4 rad
        dlon = ((lon - 0.4 + np.pi) % (2 * np.pi)) - np.pi
        dlat = lat - math.radians(-22.0)
        grs_dist = (dlon / 0.26) ** 2 + (dlat / 0.13) ** 2
        grs_mask = np.exp(-1.8 * grs_dist)
        r = r * (1.0 - grs_mask) + 0.84 * grs_mask
        g = g * (1.0 - grs_mask) + 0.31 * grs_mask
        b = b * (1.0 - grs_mask) + 0.19 * grs_mask
        return np.stack([r, g, b], axis=-1)

    _ensure_rgb("jupiter", _jupiter_tex)

    # 7. Saturn butterscotch bands + RGBA Ring texture
    def _saturn_tex() -> np.ndarray:
        turb = _fbm_2d(lon * 0.4, lat * 1.8, seed=701, octaves=5)
        y = lat + 0.03 * (turb - 0.5)
        bands = 0.5 + 0.5 * np.sin(18.0 * y)
        polar_hex = np.clip((np.abs(lat) - 1.15) * 2.0, 0.0, 0.35)
        r = 0.74 + 0.18 * bands - 0.22 * polar_hex
        g = 0.65 + 0.18 * bands - 0.12 * polar_hex
        b = 0.45 + 0.16 * bands + 0.08 * polar_hex
        return np.stack([r, g, b], axis=-1)

    _ensure_rgb("saturn", _saturn_tex)

    ring_png = TEXTURES_DIR / "saturn_rings.png"
    if not (ring_png.exists() and ring_png.stat().st_size > 2048):
        # U coordinate in [0, 1] maps radially from inner C-ring (1.23 R_s) to outer F-ring (2.32 R_s)
        RW, RH = 1024, 64
        ru = np.linspace(0.0, 1.0, RW, dtype=np.float32)[None, :]
        fine = 0.5 + 0.5 * np.sin(240.0 * ru) * np.cos(97.0 * ru)
        # C ring [0.02..0.22], B ring [0.22..0.58], Cassini Division [0.58..0.64], A ring [0.64..0.92], Encke [0.84..0.855], F [0.96..0.98]
        alpha = np.zeros((RH, RW), dtype=np.float32)
        alpha += np.where((ru >= 0.03) & (ru < 0.22), 0.22 + 0.10 * fine, 0.0)
        alpha += np.where((ru >= 0.22) & (ru < 0.58), 0.78 + 0.18 * fine, 0.0)
        alpha += np.where((ru >= 0.58) & (ru < 0.64), 0.04, 0.0)  # Cassini Division!
        alpha += np.where((ru >= 0.64) & (ru < 0.92), 0.56 + 0.15 * fine, 0.0)
        alpha = np.where((ru >= 0.835) & (ru < 0.852), 0.05, alpha)  # Encke Gap!
        alpha += np.where((ru >= 0.955) & (ru < 0.975), 0.38, 0.0)   # F Ring!
        alpha = np.clip(alpha, 0.0, 0.95)
        tone = 0.72 + 0.22 * np.where((ru >= 0.22) & (ru < 0.58), 1.0, 0.65) + 0.06 * fine
        ring_rgba = np.zeros((RH, RW, 4), dtype=np.float32)
        ring_rgba[..., 0] = np.clip(tone * 1.02, 0.0, 1.0)
        ring_rgba[..., 1] = np.clip(tone * 0.93, 0.0, 1.0)
        ring_rgba[..., 2] = np.clip(tone * 0.76, 0.0, 1.0)
        ring_rgba[..., 3] = alpha
        _save_png((ring_rgba * 255.0).astype(np.uint8), ring_png)
        print(f"[AssetFetcher] Synthesized radial Saturn rings RGBA map: {ring_png.name}")
    tex_paths["saturn_rings"] = ring_png.as_posix()
    # Separate RGB + monochrome opacity map for maximum OmniPBR_Opacity compatibility
    ring_op_png = TEXTURES_DIR / "saturn_rings_opacity.png"
    if not ring_op_png.exists():
        from PIL import Image
        im_rgba = Image.open(str(ring_png)).convert("RGBA")
        alpha_ch = np.array(im_rgba)[..., 3]
        op_rgb = np.stack([alpha_ch, alpha_ch, alpha_ch], axis=-1)
        _save_png(op_rgb, ring_op_png)
    tex_paths["saturn_rings_opacity"] = ring_op_png.as_posix()

    # 8. Uranus, Neptune & Titan
    def _uranus_tex() -> np.ndarray:
        n = _fbm_2d(lon * 0.3, lat, seed=801, octaves=4)
        b_band = 0.82 + 0.08 * np.cos(6.0 * lat) + 0.04 * n
        return np.stack([0.55 * b_band, 0.86 * b_band, 0.90 * b_band], axis=-1)

    _ensure_rgb("uranus", _uranus_tex)

    def _neptune_tex() -> np.ndarray:
        n = _fbm_2d(lon * 0.5, lat * 1.5, seed=901, octaves=5)
        streaks = 0.75 + 0.16 * np.sin(14.0 * lat + 2.0 * n)
        # Great Dark Spot
        dlon = ((lon + 0.6 + np.pi) % (2 * np.pi)) - np.pi
        dlat = lat - math.radians(-20.0)
        gds = np.exp(-2.2 * ((dlon / 0.22) ** 2 + (dlat / 0.12) ** 2))
        r = 0.16 * streaks * (1.0 - 0.35 * gds) + 0.05 * n
        g = 0.35 * streaks * (1.0 - 0.35 * gds) + 0.08 * n
        b = 0.86 * streaks * (1.0 - 0.22 * gds) + 0.10 * n
        return np.stack([r, g, b], axis=-1)

    _ensure_rgb("neptune", _neptune_tex)

    def _titan_tex() -> np.ndarray:
        n = _fbm_2d(lon, lat, seed=777, octaves=4)
        haze = 0.78 + 0.12 * np.cos(2.0 * lat) + 0.08 * n
        return np.stack([0.86 * haze, 0.62 * haze, 0.28 * haze], axis=-1)

    _ensure_rgb("titan", _titan_tex)

    # 9. Asteroid / Kuiper / Oort procedural rock & ice surface textures
    def _rock_tex() -> np.ndarray:
        n = _fbm_2d(lon, lat, seed=1101, octaves=6)
        lum = 0.24 + 0.36 * n
        return np.stack([lum * 1.04, lum * 0.98, lum * 0.91], axis=-1)

    _ensure_rgb("asteroid_rock", _rock_tex)

    def _ice_tex() -> np.ndarray:
        n = _fbm_2d(lon, lat, seed=1201, octaves=5)
        lum = 0.55 + 0.38 * n
        return np.stack([lum * 0.84, lum * 0.93, np.clip(lum * 1.05, 0.0, 1.0)], axis=-1)

    _ensure_rgb("kuiper_ice", _ice_tex)

    # 10. High-resolution Milky Way & Deep Space Starfield Dome Panorama (2048x1024)
    dome_png = TEXTURES_DIR / "milky_way_starfield.png"
    if not (dome_png.exists() and dome_png.stat().st_size > 16384):
        DW, DH = 2048, 1024
        du = np.linspace(-np.pi, np.pi, DW, endpoint=False, dtype=np.float32)[None, :]
        dv = np.linspace(-0.5 * np.pi, 0.5 * np.pi, DH, dtype=np.float32)[:, None]
        dlon = np.broadcast_to(du, (DH, DW))
        dlat = np.broadcast_to(dv, (DH, DW))

        # Galactic plane tilted ~60.2 deg to the ecliptic
        inc = math.radians(60.2)
        cx = np.cos(dlat) * np.cos(dlon)
        cy = np.cos(dlat) * np.sin(dlon)
        cz = np.sin(dlat)
        gal_z = -math.sin(inc) * cy + math.cos(inc) * cz
        gal_x = cx
        gal_y = math.cos(inc) * cy + math.sin(inc) * cz
        gal_lat = np.arcsin(np.clip(gal_z, -1.0, 1.0))
        gal_lon = np.arctan2(gal_y, gal_x)

        neb1 = _fbm_2d(dlon, dlat, seed=9001, octaves=5)
        dust = _fbm_2d(dlon * 1.7, dlat * 1.7, seed=9002, octaves=5)
        core_boost = 0.45 + 0.55 * np.exp(-1.6 * (gal_lon ** 2))
        band = np.exp(-((gal_lat / 0.21) ** 2)) * core_boost
        dust_lane = np.exp(-((gal_lat / 0.055) ** 2)) * np.clip((dust - 0.42) * 2.1, 0.0, 0.78)
        mw = np.clip(band * (0.45 + 0.75 * neb1) - 0.55 * dust_lane, 0.0, 1.0)

        sky = np.zeros((DH, DW, 3), dtype=np.float32)
        sky[..., 0] = 0.008 + 0.22 * mw
        sky[..., 1] = 0.011 + 0.24 * mw
        sky[..., 2] = 0.020 + 0.34 * mw

        # Scatter 14,000 multi-spectral stars
        rng = np.random.default_rng(19770905)
        n_stars = 14000
        sx = rng.integers(0, DW, size=n_stars)
        # Concentrate 45% along galactic plane, 55% isotropic
        iso_z = rng.uniform(-1.0, 1.0, size=n_stars)
        sy = np.clip(((np.arcsin(iso_z) / np.pi + 0.5) * DH).astype(np.int32), 0, DH - 1)
        mags = rng.pareto(2.4, size=n_stars) * 0.22
        mags = np.clip(mags, 0.12, 1.0)
        temps = rng.uniform(0.0, 1.0, size=n_stars)
        sr = np.where(temps < 0.25, 0.75, np.where(temps < 0.75, 1.0, 1.0)) * mags
        sg = np.where(temps < 0.25, 0.86, np.where(temps < 0.75, 0.97, 0.74)) * mags
        sb = np.where(temps < 0.25, 1.00, np.where(temps < 0.75, 0.92, 0.52)) * mags

        sky[sy, sx, 0] = np.clip(sky[sy, sx, 0] + sr, 0.0, 1.0)
        sky[sy, sx, 1] = np.clip(sky[sy, sx, 1] + sg, 0.0, 1.0)
        sky[sy, sx, 2] = np.clip(sky[sy, sx, 2] + sb, 0.0, 1.0)

        # Bright 3x3 cross-glow on top 350 brightest stars
        bright_idx = np.argsort(mags)[-350:]
        for idx in bright_idx:
            bx, by = int(sx[idx]), int(sy[idx])
            for dx, dy, w in [(-1, 0, 0.45), (1, 0, 0.45), (0, -1, 0.45), (0, 1, 0.45)]:
                nx = (bx + dx) % DW
                ny = max(0, min(DH - 1, by + dy))
                sky[ny, nx] = np.clip(sky[ny, nx] + w * np.array([sr[idx], sg[idx], sb[idx]]), 0.0, 1.0)

        _save_png((sky * 255.0).astype(np.uint8), dome_png)
        print(f"[AssetFetcher] Synthesized 2K Milky Way starfield panorama: {dome_png.name}")
    tex_paths["milky_way"] = dome_png.as_posix()

    return tex_paths


# ---------------------------------------------------------------------------
# OpenUSD Conversion & Articulated Spacecraft / Rocket / Asteroid Builders
# ---------------------------------------------------------------------------
def _convert_mesh_with_kit(input_mesh: Path, output_usd: Path) -> bool:
    """Convert .glb/.obj/.fbx/.stl to .usd via Isaac Sim's omni.kit.asset_converter."""
    import omni.kit.app
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    ext_mgr.set_extension_enabled_immediate("omni.kit.asset_converter", True)

    import omni.kit.asset_converter as converter

    ctx = converter.AssetConverterContext()
    ctx.ignore_materials = False
    ctx.ignore_animations = False
    ctx.ignore_camera = True
    ctx.ignore_light = True
    ctx.embed_textures = False
    ctx.use_meter_as_world_unit = True
    ctx.convert_fbx_to_y_up = False
    ctx.convert_fbx_to_z_up = True
    ctx.merge_all_meshes = False

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    mgr = converter.get_instance()
    task = mgr.create_converter_task(str(input_mesh), str(output_usd), None, ctx)

    # Pump Kit app updates until async converter task finishes
    app = omni.kit.app.get_app()
    for _ in range(600):
        app.update()
        if task.get_status() != 0:  # 0 == in progress in AssetConverterFuture
            break

    loop = asyncio.get_event_loop()
    success = loop.run_until_complete(task.wait_until_finished())
    if not success or not output_usd.exists():
        err = task.get_detailed_error()
        print(f"[AssetConverter] Warning converting {input_mesh.name}: {err}")
        return False

    # Normalize stage upAxis = Z and verify zero broken asset paths inside converted USD
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(output_usd))
    if stage:
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.GetRootLayer().Save()
    print(f"[AssetConverter] Converted {input_mesh.name} -> {output_usd.name} successfully.")
    return True


def _add_mesh_prim(
    stage,
    path: str,
    points: np.ndarray,
    face_counts: List[int],
    face_indices: List[int],
    uvs: Optional[np.ndarray] = None,
    color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
):
    from pxr import Gf, Sdf, UsdGeom, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(face_counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(face_indices))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    if uvs is not None and len(uvs) == len(points):
        primvars = UsdGeom.PrimvarsAPI(mesh)
        st = primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray([Gf.Vec2f(float(u[0]), float(u[1])) for u in uvs]))
    return mesh


def _make_parabolic_dish_geometry(radius: float = 1.85, depth: float = 0.52, n_rad: int = 12, n_seg: int = 36):
    """Generate a smooth parabolic dish mesh whose boresight points along +Z."""
    pts = []
    uvs = []
    for ir in range(n_rad + 1):
        frac = ir / n_rad
        r = radius * frac
        # Dish concave toward +Z: rim at z = +depth, vertex at z = 0.0
        z = depth * (frac ** 2)
        for iseg in range(n_seg):
            ang = 2.0 * math.pi * iseg / n_seg
            pts.append((r * math.cos(ang), r * math.sin(ang), z))
            uvs.append((0.5 + 0.5 * frac * math.cos(ang), 0.5 + 0.5 * frac * math.sin(ang)))
    counts = []
    indices = []
    stride = n_seg
    for ir in range(n_rad):
        for iseg in range(n_seg):
            jn = (iseg + 1) % n_seg
            i0 = ir * stride + iseg
            i1 = ir * stride + jn
            i2 = (ir + 1) * stride + jn
            i3 = (ir + 1) * stride + iseg
            counts.append(4)
            indices.extend([i0, i1, i2, i3])
    return np.array(pts, dtype=np.float32), counts, indices, np.array(uvs, dtype=np.float32)


def _make_prism_geometry(n_sides: int, r_outer: float, r_inner: float, z0: float, z1: float):
    """Generate a hollow decagonal/hexagonal spacecraft bus ring prism."""
    pts = []
    for z in (z0, z1):
        for r in (r_outer, r_inner):
            for i in range(n_sides):
                ang = 2.0 * math.pi * i / n_sides
                pts.append((r * math.cos(ang), r * math.sin(ang), z))
    # Ring offsets: 0=bot_out, N=bot_in, 2N=top_out, 3N=top_in
    N = n_sides
    counts, idxs = [], []
    for i in range(N):
        j = (i + 1) % N
        # Outer wall
        counts.append(4)
        idxs.extend([i, j, 2 * N + j, 2 * N + i])
        # Inner wall
        counts.append(4)
        idxs.extend([N + j, N + i, 3 * N + i, 3 * N + j])
        # Top cap
        counts.append(4)
        idxs.extend([2 * N + i, 2 * N + j, 3 * N + j, 3 * N + i])
        # Bottom cap
        counts.append(4)
        idxs.extend([N + i, N + j, j, i])
    return np.array(pts, dtype=np.float32), counts, idxs


def build_voyager_articulated_usd(output_usd: Path, converted_nasa_usd: Optional[Path] = None) -> Path:
    """Build articulated Voyager OpenUSD spacecraft with unfolding booms and optional NASA reference."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    if output_usd.exists():
        output_usd.unlink()
    stage = Usd.Stage.CreateNew(str(output_usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Voyager")
    stage.SetDefaultPrim(root.GetPrim())

    # Local Materials Scope using Kit's OmniPBR.mdl
    mdl_path = (KIT_MDL_BASE_DIR / "OmniPBR.mdl").as_posix()
    mat_scope = UsdGeom.Scope.Define(stage, "/Voyager/Materials")

    def _local_pbr(name: str, rgb: Tuple[float, float, float], metallic: float, rough: float) -> UsdShade.Material:
        mat = UsdShade.Material.Define(stage, f"/Voyager/Materials/{name}")
        sh = UsdShade.Shader.Define(stage, f"/Voyager/Materials/{name}/Shader")
        sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        sh.SetSourceAsset(Sdf.AssetPath(mdl_path), "mdl")
        sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(metallic))
        sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(rough))
        mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        # Also add UsdPreviewSurface fallback output
        prev = UsdShade.Shader.Define(stage, f"/Voyager/Materials/{name}/Preview")
        prev.CreateIdAttr("UsdPreviewSurface")
        prev.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        prev.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
        prev.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
        mat.CreateSurfaceOutput().ConnectToSource(prev.ConnectableAPI(), "surface")
        return mat

    mat_dish = _local_pbr("Mat_WhiteDish", (0.94, 0.95, 0.96), 0.15, 0.38)
    mat_gold = _local_pbr("Mat_GoldKapton", (0.92, 0.68, 0.16), 0.88, 0.26)
    mat_record = _local_pbr("Mat_GoldenRecord", (0.98, 0.78, 0.22), 0.96, 0.12)
    mat_dark = _local_pbr("Mat_ThermalBlack", (0.14, 0.15, 0.17), 0.45, 0.62)
    mat_metal = _local_pbr("Mat_TitaniumStrut", (0.72, 0.75, 0.79), 0.85, 0.30)

    # 1. Decagonal Spacecraft Bus (1.78 m diameter, z in [-0.62, -0.05])
    bus_pts, bus_cnts, bus_idxs = _make_prism_geometry(10, 0.89, 0.48, -0.62, -0.06)
    bus_mesh = _add_mesh_prim(stage, "/Voyager/DecagonalBus", bus_pts, bus_cnts, bus_idxs, color=(0.18, 0.19, 0.22))
    UsdShade.MaterialBindingAPI.Apply(bus_mesh.GetPrim()).Bind(mat_dark)

    # Thermal Gold Kapton Hydrazine Tank Core inside the decagonal ring
    tank = UsdGeom.Sphere.Define(stage, "/Voyager/HydrazineCoreTank")
    tank.CreateRadiusAttr(0.46)
    tank.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.34))
    UsdShade.MaterialBindingAPI.Apply(tank.GetPrim()).Bind(mat_gold)

    # Voyager Golden Record mounted on +Y bus facet
    rec = UsdGeom.Cylinder.Define(stage, "/Voyager/GoldenRecord")
    rec.CreateRadiusAttr(0.18)
    rec.CreateHeightAttr(0.025)
    rec.CreateAxisAttr("Y")
    rec.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.90, -0.34))
    UsdShade.MaterialBindingAPI.Apply(rec.GetPrim()).Bind(mat_record)

    # 2. 3.7 m Parabolic High-Gain Antenna Dish (+Z boresight points toward Earth!)
    dish_pts, dish_cnts, dish_idxs, dish_uvs = _make_parabolic_dish_geometry(1.85, 0.52, 12, 40)
    dish_mesh = _add_mesh_prim(stage, "/Voyager/HighGainDish", dish_pts, dish_cnts, dish_idxs, dish_uvs, (0.95, 0.95, 0.96))
    UsdShade.MaterialBindingAPI.Apply(dish_mesh.GetPrim()).Bind(mat_dish)

    # Subreflector & Tripod Feed Struts along +Z
    subref = UsdGeom.Cylinder.Define(stage, "/Voyager/HighGainFeedSubreflector")
    subref.CreateRadiusAttr(0.24)
    subref.CreateHeightAttr(0.10)
    subref.CreateAxisAttr("Z")
    subref.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.12))
    UsdShade.MaterialBindingAPI.Apply(subref.GetPrim()).Bind(mat_gold)

    for s_idx in range(3):
        ang = math.radians(120.0 * s_idx + 30.0)
        strut = UsdGeom.Cylinder.Define(stage, f"/Voyager/FeedStrut_{s_idx}")
        strut.CreateRadiusAttr(0.022)
        strut.CreateHeightAttr(1.22)
        strut.CreateAxisAttr("Z")
        strut.AddTranslateOp().Set(Gf.Vec3d(0.55 * math.cos(ang), 0.55 * math.sin(ang), 0.58))
        strut.AddRotateXYZOp().Set(Gf.Vec3f(float(-24.0 * math.sin(ang)), float(24.0 * math.cos(ang)), 0.0))
        UsdShade.MaterialBindingAPI.Apply(strut.GetPrim()).Bind(mat_metal)

    # 3. Articulated 13-meter Magnetometer Boom (pivots at (-0.85, 0, -0.35), extends along -X when deployed)
    mag_pivot = UsdGeom.Xform.Define(stage, "/Voyager/MagnetometerBoomPivot")
    mag_pivot.AddTranslateOp().Set(Gf.Vec3d(-0.85, 0.0, -0.35))
    mag_pivot.AddRotateYOp(opSuffix="deploy").Set(0.0)

    mag_mast = UsdGeom.Cylinder.Define(stage, "/Voyager/MagnetometerBoomPivot/Astromast")
    mag_mast.CreateRadiusAttr(0.055)
    mag_mast.CreateHeightAttr(8.8)  # Compact true-proportion visual boom
    mag_mast.CreateAxisAttr("X")
    mag_mast.AddTranslateOp().Set(Gf.Vec3d(-4.4, 0.0, 0.0))
    UsdShade.MaterialBindingAPI.Apply(mag_mast.GetPrim()).Bind(mat_gold)

    mag_tip = UsdGeom.Cube.Define(stage, "/Voyager/MagnetometerBoomPivot/FluxgateSensor")
    mag_tip.CreateSizeAttr(0.20)
    mag_tip.AddTranslateOp().Set(Gf.Vec3d(-8.85, 0.0, 0.0))
    UsdShade.MaterialBindingAPI.Apply(mag_tip.GetPrim()).Bind(mat_gold)

    # 4. Articulated MHW-RTG Boom (pivots at (+0.85, -0.25, -0.38), extends along +X/-Y)
    rtg_pivot = UsdGeom.Xform.Define(stage, "/Voyager/RTGBoomPivot")
    rtg_pivot.AddTranslateOp().Set(Gf.Vec3d(0.85, -0.20, -0.38))
    rtg_pivot.AddRotateYOp(opSuffix="deploy").Set(0.0)

    rtg_strut = UsdGeom.Cylinder.Define(stage, "/Voyager/RTGBoomPivot/OutriggerStrut")
    rtg_strut.CreateRadiusAttr(0.06)
    rtg_strut.CreateHeightAttr(2.4)
    rtg_strut.CreateAxisAttr("X")
    rtg_strut.AddTranslateOp().Set(Gf.Vec3d(1.2, 0.0, 0.0))
    UsdShade.MaterialBindingAPI.Apply(rtg_strut.GetPrim()).Bind(mat_metal)

    for r_i in range(3):
        rtg_can = UsdGeom.Cylinder.Define(stage, f"/Voyager/RTGBoomPivot/MHW_RTG_{r_i}")
        rtg_can.CreateRadiusAttr(0.20)
        rtg_can.CreateHeightAttr(0.52)
        rtg_can.CreateAxisAttr("X")
        rtg_can.AddTranslateOp().Set(Gf.Vec3d(2.55 + r_i * 0.58, 0.0, 0.0))
        UsdShade.MaterialBindingAPI.Apply(rtg_can.GetPrim()).Bind(mat_dark)

    # 5. Articulated Science Scan Platform (extends along +Y/+X carrying ISS Cameras, IRIS, UVS)
    scan_pivot = UsdGeom.Xform.Define(stage, "/Voyager/ScanPlatformPivot")
    scan_pivot.AddTranslateOp().Set(Gf.Vec3d(0.35, 0.85, -0.32))
    scan_pivot.AddRotateXOp(opSuffix="deploy").Set(0.0)

    scan_boom = UsdGeom.Cylinder.Define(stage, "/Voyager/ScanPlatformPivot/ScanBoom")
    scan_boom.CreateRadiusAttr(0.055)
    scan_boom.CreateHeightAttr(2.1)
    scan_boom.CreateAxisAttr("Y")
    scan_boom.AddTranslateOp().Set(Gf.Vec3d(0.0, 1.05, 0.0))
    UsdShade.MaterialBindingAPI.Apply(scan_boom.GetPrim()).Bind(mat_metal)

    head = UsdGeom.Cube.Define(stage, "/Voyager/ScanPlatformPivot/InstrumentHead")
    head.CreateSizeAttr(0.46)
    head.AddTranslateOp().Set(Gf.Vec3d(0.0, 2.25, 0.0))
    UsdShade.MaterialBindingAPI.Apply(head.GetPrim()).Bind(mat_dark)

    nac = UsdGeom.Cylinder.Define(stage, "/Voyager/ScanPlatformPivot/ISS_NarrowAngleCam")
    nac.CreateRadiusAttr(0.11)
    nac.CreateHeightAttr(0.48)
    nac.CreateAxisAttr("Y")
    nac.AddTranslateOp().Set(Gf.Vec3d(0.16, 2.52, 0.12))
    UsdShade.MaterialBindingAPI.Apply(nac.GetPrim()).Bind(mat_metal)

    # 6. Dual 10m PRA / PWS Radio Astronomy Whip Antennas
    for w_idx, yaw in enumerate((-38.0, 38.0)):
        whip = UsdGeom.Cylinder.Define(stage, f"/Voyager/PRA_WhipAntenna_{w_idx}")
        whip.CreateRadiusAttr(0.015)
        whip.CreateHeightAttr(6.5)
        whip.CreateAxisAttr("Y")
        whip.AddTranslateOp().Set(Gf.Vec3d(math.sin(math.radians(yaw)) * 3.25, -math.cos(math.radians(yaw)) * 3.25, -0.52))
        whip.AddRotateZOp().Set(float(yaw))
        UsdShade.MaterialBindingAPI.Apply(whip.GetPrim()).Bind(mat_metal)

    # Reference converted public-domain NASA Voyager 3D mesh if all internal asset paths resolve cleanly
    if converted_nasa_usd and converted_nasa_usd.exists():
        root.GetPrim().SetCustomDataByKey("nasa_converted_usd", converted_nasa_usd.name)
        nasa_stage = Usd.Stage.Open(str(converted_nasa_usd))
        if nasa_stage:
            broken_assets = 0
            base_dir = converted_nasa_usd.parent
            for p in nasa_stage.Traverse():
                for attr in p.GetAttributes():
                    val = attr.Get()
                    if isinstance(val, Sdf.AssetPath) and val.path:
                        raw_p = val.resolvedPath or val.path
                        if not raw_p.endswith(".mdl") and not Path(raw_p).exists() and not (base_dir / val.path).exists():
                            broken_assets += 1
            if broken_assets == 0:
                bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
                bound = bbox_cache.ComputeWorldBound(nasa_stage.GetPseudoRoot()).ComputeAlignedBox()
                size_vec = bound.GetMax() - bound.GetMin()
                max_dim = max(float(size_vec[0]), float(size_vec[1]), float(size_vec[2]), 1e-4)
                center_vec = (bound.GetMax() + bound.GetMin()) * 0.5
                # Normalize NASA mesh to ~2.8m core span so it nests cleanly with our articulated booms & dish
                norm_s = 2.8 / max_dim
                nasa_ref_xf = UsdGeom.Xform.Define(stage, "/Voyager/NASA_ReferenceMesh")
                nasa_ref_xf.AddScaleOp().Set(Gf.Vec3f(norm_s, norm_s, norm_s))
                nasa_ref_xf.AddTranslateOp().Set(Gf.Vec3d(-float(center_vec[0]), -float(center_vec[1]), -float(center_vec[2])))
                def_prim = nasa_stage.GetDefaultPrim()
                rel_asset = f"./{converted_nasa_usd.name}"
                if def_prim and def_prim.IsValid():
                    nasa_ref_xf.GetPrim().GetReferences().AddReference(rel_asset)
                print(f"[AssetBuilder] Composed NASA 3D Voyager mesh ({rel_asset}, scale={norm_s:.4f}) into /Voyager/NASA_ReferenceMesh.")

    stage.GetRootLayer().Save()
    print(f"[AssetBuilder] Saved articulated Voyager OpenUSD: {output_usd.name}")
    return output_usd


def build_titan_iiie_centaur_usd(output_usd: Path) -> Path:
    """Build articulated Titan IIIE / Centaur D-1T multi-stage rocket OpenUSD for Phase 1."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    if output_usd.exists():
        output_usd.unlink()
    stage = Usd.Stage.CreateNew(str(output_usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/TitanIIIE")
    stage.SetDefaultPrim(root.GetPrim())

    mdl_path = (KIT_MDL_BASE_DIR / "OmniPBR.mdl").as_posix()

    def _mat(name: str, rgb: Tuple[float, float, float], met: float, rough: float, emissive: Optional[Tuple[float, float, float]] = None):
        mat = UsdShade.Material.Define(stage, f"/TitanIIIE/Materials/{name}")
        sh = UsdShade.Shader.Define(stage, f"/TitanIIIE/Materials/{name}/Shader")
        sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        sh.SetSourceAsset(Sdf.AssetPath(mdl_path), "mdl")
        sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(met))
        sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(rough))
        if emissive is not None:
            sh.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
            sh.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
            sh.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(4500.0)
        mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        return mat

    m_white = _mat("AeroWhite", (0.93, 0.94, 0.96), 0.20, 0.35)
    m_silver = _mat("TitanAlloy", (0.72, 0.74, 0.78), 0.82, 0.28)
    m_red = _mat("NASAAccentRed", (0.82, 0.18, 0.12), 0.25, 0.40)
    m_dark = _mat("EngineNozzle", (0.16, 0.17, 0.20), 0.78, 0.42)
    m_plume_srb = _mat("PlumeSRB", (1.0, 0.78, 0.32), 0.0, 0.9, emissive=(1.0, 0.68, 0.22))
    m_plume_centaur = _mat("PlumeHydrolox", (0.45, 0.78, 1.0), 0.0, 0.9, emissive=(0.38, 0.74, 1.0))

    # Core Stage (Stage 1 + Stage 2, total height ~30 m along +Z)
    core_xf = UsdGeom.Xform.Define(stage, "/TitanIIIE/CoreStage")
    core_xf.AddTranslateOp(opSuffix="sep").Set(Gf.Vec3d(0.0, 0.0, 0.0))

    c1 = UsdGeom.Cylinder.Define(stage, "/TitanIIIE/CoreStage/Body")
    c1.CreateRadiusAttr(1.52)
    c1.CreateHeightAttr(26.0)
    c1.CreateAxisAttr("Z")
    c1.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 13.0))
    UsdShade.MaterialBindingAPI.Apply(c1.GetPrim()).Bind(m_silver)

    nozzle_core = UsdGeom.Cone.Define(stage, "/TitanIIIE/CoreStage/LR87_Nozzles")
    nozzle_core.CreateRadiusAttr(1.15)
    nozzle_core.CreateHeightAttr(2.4)
    nozzle_core.CreateAxisAttr("Z")
    nozzle_core.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.8))
    UsdShade.MaterialBindingAPI.Apply(nozzle_core.GetPrim()).Bind(m_dark)

    plume_core = UsdGeom.Cone.Define(stage, "/TitanIIIE/CoreStage/CorePlume")
    plume_core.CreateRadiusAttr(1.55)
    plume_core.CreateHeightAttr(12.0)
    plume_core.CreateAxisAttr("Z")
    plume_core.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -7.2))
    plume_core.AddRotateXOp().Set(180.0)
    plume_core.AddScaleOp(opSuffix="throttle").Set(Gf.Vec3f(1.0, 1.0, 1.0))
    UsdShade.MaterialBindingAPI.Apply(plume_core.GetPrim()).Bind(m_plume_srb)

    # Twin UA1205 Solid Rocket Boosters (SRB_Left at -Y, SRB_Right at +Y)
    for side, y_sign in (("SRB_Left", -1.0), ("SRB_Right", 1.0)):
        srb_xf = UsdGeom.Xform.Define(stage, f"/TitanIIIE/{side}")
        srb_xf.AddTranslateOp(opSuffix="sep").Set(Gf.Vec3d(0.0, y_sign * 3.15, 0.0))
        srb_xf.AddRotateXOp(opSuffix="peel").Set(0.0)

        body = UsdGeom.Cylinder.Define(stage, f"/TitanIIIE/{side}/MotorCase")
        body.CreateRadiusAttr(1.52)
        body.CreateHeightAttr(21.5)
        body.CreateAxisAttr("Z")
        body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 10.75))
        UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(m_white)

        band = UsdGeom.Cylinder.Define(stage, f"/TitanIIIE/{side}/AccentBand")
        band.CreateRadiusAttr(1.54)
        band.CreateHeightAttr(1.2)
        band.CreateAxisAttr("Z")
        band.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 19.5))
        UsdShade.MaterialBindingAPI.Apply(band.GetPrim()).Bind(m_red)

        nose = UsdGeom.Cone.Define(stage, f"/TitanIIIE/{side}/NoseCone")
        nose.CreateRadiusAttr(1.52)
        nose.CreateHeightAttr(4.2)
        nose.CreateAxisAttr("Z")
        nose.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 23.6))
        UsdShade.MaterialBindingAPI.Apply(nose.GetPrim()).Bind(m_white)

        plume = UsdGeom.Cone.Define(stage, f"/TitanIIIE/{side}/SRBPlume")
        plume.CreateRadiusAttr(1.85)
        plume.CreateHeightAttr(16.0)
        plume.CreateAxisAttr("Z")
        plume.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -8.5))
        plume.AddRotateXOp().Set(180.0)
        plume.AddScaleOp(opSuffix="throttle").Set(Gf.Vec3f(1.0, 1.0, 1.0))
        UsdShade.MaterialBindingAPI.Apply(plume.GetPrim()).Bind(m_plume_srb)

    # Centaur D-1T Cryogenic Upper Stage (z in [26.0, 35.2])
    cent_xf = UsdGeom.Xform.Define(stage, "/TitanIIIE/CentaurStage")
    cent_xf.AddTranslateOp(opSuffix="sep").Set(Gf.Vec3d(0.0, 0.0, 26.0))

    cent_body = UsdGeom.Cylinder.Define(stage, "/TitanIIIE/CentaurStage/Tank")
    cent_body.CreateRadiusAttr(1.55)
    cent_body.CreateHeightAttr(9.0)
    cent_body.CreateAxisAttr("Z")
    cent_body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 4.5))
    UsdShade.MaterialBindingAPI.Apply(cent_body.GetPrim()).Bind(m_white)

    cent_plume = UsdGeom.Cone.Define(stage, "/TitanIIIE/CentaurStage/CentaurPlume")
    cent_plume.CreateRadiusAttr(1.25)
    cent_plume.CreateHeightAttr(9.5)
    cent_plume.CreateAxisAttr("Z")
    cent_plume.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -4.8))
    cent_plume.AddRotateXOp().Set(180.0)
    cent_plume.AddScaleOp(opSuffix="throttle").Set(Gf.Vec3f(0.001, 0.001, 0.001))
    UsdShade.MaterialBindingAPI.Apply(cent_plume.GetPrim()).Bind(m_plume_centaur)

    # Clamshell Payload Fairing Halves (Fairing_Left at -X, Fairing_Right at +X, z in [35.0, 48.5])
    for f_name, x_sign in (("Fairing_Left", -1.0), ("Fairing_Right", 1.0)):
        f_xf = UsdGeom.Xform.Define(stage, f"/TitanIIIE/{f_name}")
        f_xf.AddTranslateOp(opSuffix="sep").Set(Gf.Vec3d(0.0, 0.0, 35.0))
        f_xf.AddRotateYOp(opSuffix="peel").Set(0.0)

        # Half-cylinder + ogive nose mesh
        pts = []
        n_seg = 16
        ang_start = 0.5 * math.pi if x_sign < 0 else -0.5 * math.pi
        ang_end = 1.5 * math.pi if x_sign < 0 else 0.5 * math.pi
        z_rings = [(0.0, 2.13), (8.0, 2.13), (11.2, 1.35), (13.2, 0.45), (14.2, 0.04)]
        for zr, rad in z_rings:
            for k in range(n_seg + 1):
                ang = ang_start + (ang_end - ang_start) * (k / n_seg)
                pts.append((rad * math.cos(ang), rad * math.sin(ang), zr))
        cnts, idxs = [], []
        stride = n_seg + 1
        for ir in range(len(z_rings) - 1):
            for k in range(n_seg):
                i0 = ir * stride + k
                i1 = i0 + 1
                i2 = (ir + 1) * stride + k + 1
                i3 = (ir + 1) * stride + k
                cnts.append(4)
                idxs.extend([i0, i1, i2, i3])
        f_mesh = _add_mesh_prim(stage, f"/TitanIIIE/{f_name}/ShroudHalf", np.array(pts, dtype=np.float32), cnts, idxs, color=(0.95, 0.95, 0.97))
        UsdShade.MaterialBindingAPI.Apply(f_mesh.GetPrim()).Bind(m_white)

    stage.GetRootLayer().Save()
    print(f"[AssetBuilder] Saved articulated Titan IIIE / Centaur OpenUSD: {output_usd.name}")
    return output_usd


def build_asteroid_prototypes_usd(output_usd: Path) -> Path:
    """Generate 6 distinct displaced rock/ice low-VRAM OpenUSD prototypes for PointInstancers."""
    from pxr import Usd, UsdGeom

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    if output_usd.exists():
        output_usd.unlink()
    stage = Usd.Stage.CreateNew(str(output_usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/AsteroidPrototypes")
    stage.SetDefaultPrim(root.GetPrim())

    # Subdivided UV-sphere / octa-sphere (~144 quads per prototype -> ultra-fast instancing)
    n_lat, n_lon = 10, 16
    specs = [
        ("Proto_Silicate_0", 101, (1.25, 0.92, 0.78), (0.48, 0.44, 0.39)),
        ("Proto_Silicate_1", 102, (0.88, 1.35, 0.82), (0.54, 0.49, 0.43)),
        ("Proto_Carbonaceous_2", 103, (1.10, 1.05, 0.95), (0.28, 0.27, 0.26)),
        ("Proto_KuiperIce_3", 104, (1.30, 0.85, 0.90), (0.62, 0.72, 0.82)),
        ("Proto_KuiperTholin_4", 105, (0.95, 1.20, 0.76), (0.58, 0.38, 0.32)),
        ("Proto_OortComet_5", 106, (1.18, 0.94, 1.08), (0.74, 0.84, 0.94)),
    ]

    for name, seed, axes, col in specs:
        rng = np.random.default_rng(seed)
        k_vecs = rng.normal(0.0, 1.0, size=(8, 3))
        phases = rng.uniform(0.0, 2.0 * math.pi, size=8)
        pts, uvs = [], []
        for ilat in range(n_lat + 1):
            v = ilat / n_lat
            phi = (v - 0.5) * math.pi
            for ilon in range(n_lon):
                u = ilon / n_lon
                theta = 2.0 * math.pi * u
                nx = math.cos(phi) * math.cos(theta)
                ny = math.cos(phi) * math.sin(theta)
                nz = math.sin(phi)
                disp = 1.0
                for j in range(8):
                    freq = 1.8 + 0.45 * j
                    disp += 0.085 * math.sin(freq * (k_vecs[j, 0] * nx + k_vecs[j, 1] * ny + k_vecs[j, 2] * nz) + phases[j])
                pts.append((nx * axes[0] * disp, ny * axes[1] * disp, nz * axes[2] * disp))
                uvs.append((u, v))
        cnts, idxs = [], []
        for ilat in range(n_lat):
            for ilon in range(n_lon):
                jn = (ilon + 1) % n_lon
                i0 = ilat * n_lon + ilon
                i1 = ilat * n_lon + jn
                i2 = (ilat + 1) * n_lon + jn
                i3 = (ilat + 1) * n_lon + ilon
                cnts.append(4)
                idxs.extend([i0, i1, i2, i3])
        _add_mesh_prim(stage, f"/AsteroidPrototypes/{name}", np.array(pts, dtype=np.float32), cnts, idxs, np.array(uvs, dtype=np.float32), col)

    stage.GetRootLayer().Save()
    print(f"[AssetBuilder] Saved 6 displaced asteroid/ice prototypes: {output_usd.name}")
    return output_usd


def ensure_all_assets_ready(app_already_running: bool = False) -> Dict[str, object]:
    """Master idempotent pipeline: scan -> download -> convert -> synthesize USD & textures."""
    ASSETS_USD_DIR.mkdir(parents=True, exist_ok=True)
    TEXTURES_DIR.mkdir(parents=True, exist_ok=True)

    sim_app = None
    if not app_already_running:
        try:
            from isaacsim.simulation_app import SimulationApp
        except ImportError:
            from omni.isaac.kit import SimulationApp  # type: ignore
        sim_app = SimulationApp({"headless": True})

    try:
        local_report = scan_local_simready_and_mdl()
        raw_models = discover_and_download_nasa_models()
        textures = build_or_fetch_textures()

        converted_voyager_usd: Optional[Path] = None
        if "voyager" in raw_models:
            target = ASSETS_USD_DIR / "voyager_nasa_converted.usd"
            if target.exists() and target.stat().st_size > 1024:
                converted_voyager_usd = target
            elif _convert_mesh_with_kit(raw_models["voyager"], target):
                converted_voyager_usd = target

        converted_rocket_usd: Optional[Path] = None
        if "rocket" in raw_models:
            target_r = ASSETS_USD_DIR / "rocket_nasa_converted.usd"
            if target_r.exists() and target_r.stat().st_size > 1024:
                converted_rocket_usd = target_r
            elif _convert_mesh_with_kit(raw_models["rocket"], target_r):
                converted_rocket_usd = target_r

        voyager_usd = build_voyager_articulated_usd(ASSETS_USD_DIR / "voyager_articulated.usd", converted_voyager_usd)
        titan_usd = build_titan_iiie_centaur_usd(ASSETS_USD_DIR / "titan_iiie_centaur.usd")
        asteroids_usd = build_asteroid_prototypes_usd(ASSETS_USD_DIR / "asteroid_prototypes.usd")

        manifest = {
            "local_scan": local_report,
            "raw_downloads": {k: v.as_posix() for k, v in raw_models.items()},
            "converted_nasa_voyager_usd": converted_voyager_usd.as_posix() if converted_voyager_usd else None,
            "converted_nasa_rocket_usd": converted_rocket_usd.as_posix() if converted_rocket_usd else None,
            "voyager_articulated_usd": voyager_usd.as_posix(),
            "titan_iiie_centaur_usd": titan_usd.as_posix(),
            "asteroid_prototypes_usd": asteroids_usd.as_posix(),
            "textures": textures,
        }
        manifest_path = ASSETS_USD_DIR / "asset_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[AssetFetcher] Complete! Manifest written to {manifest_path}")
        return manifest
    finally:
        if sim_app is not None:
            sim_app.close()


if __name__ == "__main__":
    ensure_all_assets_ready(app_already_running=False)
