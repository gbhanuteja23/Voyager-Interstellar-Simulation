# Voyager Interstellar Journey Simulation — Implementation Plan

## 1. Environment & Local Asset Audit Findings

1. **NVIDIA Isaac Sim Runtime (`D:\0-NVIDIA-Omniverse\isaac-sim`):**
   - Detected **Isaac Sim `5.1.0-rc.19+release.26219.9c81211b.gl`** bundling **Python `3.11.13`** (`python.bat`) and `omni.kit.asset_converter` (`5.0.17`).
   - Compatible with modern `isaacsim.simulation_app.SimulationApp` (with graceful fallback to `omni.isaac.kit.SimulationApp`).
2. **SimReady Central Repository Scan (`D:\0-NVIDIA-Omniverse\1-Isaac-sim-assets\simready-assets-hf\simready-central`):**
   - Direct filesystem inspection confirmed `simready-central` is a clone of the **SimReady Central USD Search** HuggingFace Space backend (`usdsearch-dev`), not an extracted local USD asset volume. No `.usd`, `.usda`, `.usdc`, or `.mdl` asset files reside inside `simready-central` on disk, and its remote catalog (`services/explorer/dist/package_sizes.json`, 2,597 industrial/medical/automotive assets) contains zero natural rock, boulder, asteroid, terrain, or astronomical assets. No Docker container mount is required.
   - **Local Omniverse Core MDL Library (`D:\0-NVIDIA-Omniverse\isaac-sim\kit\mdl\core\Base\`):** Verified directly accessible on disk and indexed for scene materials:
     - `OmniPBR.mdl` — Planetary surfaces, displaced rocky/icy asteroid & Oort prototypes, gold Kapton MLI foil, brushed titanium/aluminum bus, launchpad structure.
     - `OmniPBR_Opacity.mdl` — Saturn's radial ring system (Cassini/Encke alpha), Earth cloud layer, teardrop Termination Shock & Heliopause fresnel-rim translucent shells.
     - `OmniPBR_ClearCoat.mdl` — Voyager Golden Record copper-gold anodized cover, Titan IIIE / Centaur painted stages & payload fairing.
     - `OmniEmissive.mdl` — Solar photosphere & multi-shell volumetric-style corona, solid rocket booster & Centaur RL-10 plumes, Heliopause crossing markers (2012 & 2018), solar-wind and cosmic-ray particle streams.
     - `OmniGlass.mdl` — ISS narrow/wide-angle camera optics on Voyager's Science Scan Platform.

---

## 2. Modular Project Structure (`./voyager_interstellar_sim/`)

```text
voyager_interstellar_sim/
├── __init__.py
├── config.py                    # Constants, mission epochs, piecewise log radial math, planet & spline tables
├── fetch_and_convert_assets.py  # Automated NASA 3D / planetary texture downloader, USD converter & procedural fallbacks
├── materials_vfx.py             # OmniPBR / OmniPBR_Opacity / OmniEmissive builders, Corona, Rings & Heliopause VFX
├── kinematics.py                # Keplerian orbits, Catmull-Rom V1/V2 splines, HGA Earth-tracking & live telemetry
├── scene_builder.py             # Full OpenUSD stage assembly, PointInstancers (Belts & Oort), Rocket & Camera rig
├── ui_mission_control.py        # Interactive omni.ui HUD (phase jumps, timeline scrubber, speed warp, cameras, telemetry)
├── main_voyager_sim.py          # Standalone Isaac Sim 5.x launcher + --headless --verify autonomous test suite
├── assets_raw/                  # Downloaded public-domain NASA GLB/OBJ models & planetary/starfield textures
└── assets_usd/                  # Converted & procedurally articulated OpenUSD assets + embedded/local textures
```

---

## 3. OpenUSD Stage Prim Hierarchy

Stage metadata: `upAxis = "Z"`, `metersPerUnit = 1.0`.

```text
/World (Xform)
├── Environment (Xform)
│   ├── MilkyWayDomeLight (DomeLight — 4K deep-space / galactic plane panorama)
│   └── FillDistantLight (DistantLight — subtle interstellar ambient fill)
├── Materials (Scope)
│   ├── Mat_SunPhotosphere, Mat_SunCoronaInner, Mat_SunCoronaOuter
│   ├── Mat_Mercury .. Mat_Neptune, Mat_EarthClouds, Mat_SaturnRings, Mat_Titan
│   ├── Mat_GoldKapton, Mat_SpacecraftBus, Mat_AntennaDish, Mat_RTGCarbon, Mat_GoldenRecord
│   ├── Mat_RocketWhite, Mat_RocketDarkMetal, Mat_ThrusterPlumeSRB, Mat_ThrusterPlumeHydrolox
│   ├── Mat_AsteroidSilicate_0..2, Mat_KuiperIce_0..2, Mat_OortCometaryIce
│   └── Mat_TerminationShock, Mat_Heliopause, Mat_SolarWindIon, Mat_CosmicRayIon, Mat_WaypointBeacon
├── SolarSystem (Xform)
│   ├── Sun (Xform)
│   │   ├── Photosphere (Mesh)
│   │   ├── CoronaInner (Mesh) & CoronaOuter (Mesh)
│   │   └── SolarSphereLight (SphereLight)
│   ├── Orbits (Scope — UsdGeom.BasisCurves for Mercury..Neptune)
│   └── Planets (Xform)
│       ├── Mercury, Venus, Mars, Jupiter, Uranus, Neptune (OrbitXform -> TiltXform -> SpinMesh)
│       ├── Earth (OrbitXform -> TiltXform -> SpinMesh + CloudsMesh + LC41_LaunchAnchor)
│       └── Saturn (OrbitXform -> TiltXform -> SpinMesh + RingsAnnulusMesh + TitanMoon)
├── Belts (Xform)
│   ├── MainAsteroidBelt (PointInstancer — 6 displaced rock prototypes, 2,500 orbiting instances)
│   └── KuiperBelt (PointInstancer — 6 icy/rocky prototypes, 3,000 orbiting instances)
├── Interstellar (Xform)
│   ├── TerminationShock (Mesh — Rankine-deformed solar wind shock shell ~84–94 AU)
│   ├── Heliopause (Mesh — Teardrop/comet-tail ISM-deformed boundary ~119–122 AU + heliotail)
│   ├── CrossingMarkers (Xform)
│   │   ├── Voyager1_Heliopause_2012 (Xform — glowing beacon + ring halo at 121.6 AU, +34.5° ecliptic lat)
│   │   └── Voyager2_Heliopause_2018 (Xform — glowing beacon + ring halo at 119.0 AU, -32.2° ecliptic lat)
│   ├── ParticleStreams (Xform)
│   │   ├── SolarWindStream (PointInstancer — radial outward plasma streaks inside Heliopause)
│   │   └── CosmicRayStream (PointInstancer — directional VLISM inflow streaks outside Heliopause)
│   └── OortCloud (PointInstancer — spherical shell of 3,500 icy bodies spanning 1,000–25,000 AU)
├── LaunchComplex (Xform)
│   └── TitanIIIE_Centaur (Xform — CoreStage, SRB_Left, SRB_Right, CentaurStage, FairingL, FairingR, Plumes)
├── Spacecraft (Xform)
│   ├── TrajectoryCurves (Scope — Voyager1_Curve, Voyager2_Curve, Future_Interstellar_Vectors)
│   ├── Voyager1 (Xform — Bus + NASA_ReferenceMesh + HighGainDish + MagBoom + RTGArm + ScanPlatform + GoldenRecord + LocatorHalo)
│   └── Voyager2 (Xform — Bus + NASA_ReferenceMesh + HighGainDish + MagBoom + RTGArm + ScanPlatform + GoldenRecord + LocatorHalo)
└── Cameras (Xform)
    ├── VoyagerChaseCam (Camera)
    ├── RocketStagingCam (Camera)
    ├── EclipticOverviewCam (Camera)
    ├── HeliopauseCrossSectionCam (Camera)
    └── FreeOrbitCam (Camera)
```

---

## 4. Asset Mapping & Acquisition Strategy

| Simulation Element | Primary Public-Domain Source (`fetch_and_convert_assets.py`) | OpenUSD Output & Articulated Fallback |
| :--- | :--- | :--- |
| **Voyager 1 & 2 Spacecraft** | NASA 3D Resources GitHub (`Voyager` `.glb`/`.stl`) converted via `omni.kit.asset_converter` | `./assets_usd/voyager_nasa.usd` composed with `./assets_usd/voyager_articulated.usd` (3.7 m parabolic HGA, decagonal bus, Golden Record, unfolding 13 m magnetometer boom, unfolding 3-cylinder MHW-RTG boom, 2-axis science scan platform) |
| **Titan IIIE / Centaur** | NASA 3D Resources GitHub launch vehicle search + procedural aerospace USD assembly | `./assets_usd/titan_iiie_centaur.usd` with independent stage Xforms (`CoreStage`, `SRB_L/R`, `Centaur`, `Fairing_L/R`, animated emissive thruster cones) |
| **Planetary & Ring Maps** | Public-domain NASA JPL / SVS / Wikimedia Commons CC0/CC-BY equirectangular 2K albedo & bump maps | Cached in `./assets_usd/textures/` with deterministic high-resolution multi-band procedural synthesis if any endpoint is unreachable |
| **Asteroids, Kuiper & Oort** | Displaced multi-octave 3D FBM icosphere USD prototypes (`asteroid_prototypes.usd`) bound to local `OmniPBR.mdl` | 6 distinct low-poly (~160 faces each) rock/ice prototypes instanced via `UsdGeom.PointInstancer` (< 150 MB VRAM total) |

---

## 5. Piecewise Logarithmic Radial Distance & Apparent Scale Math

To span $10^{-4}\text{ AU}$ (Earth ascent) to $25,000\text{ AU}$ (Oort Cloud) inside single-precision `float32` coordinates without camera jitter while preserving true heliocentric telemetry on the `omni.ui` HUD, true heliocentric distance $r = \|\mathbf{x}_{\text{AU}}\|$ maps monotonically to simulation radius $R_{\text{sim}}(r)$ (in meters):

1. **Regime I — Inner Solar System ($0 \le r \le 2.0\text{ AU}$):**
   $$R_{\text{sim}}(r) = R_0 \frac{\ln(1 + \alpha_1 r)}{\ln(1 + \alpha_1)}, \quad R_0 = 75.0\text{ m},\; \alpha_1 = 2.5$$
   - Sun center: $0\text{ m}$ (photosphere $14.0\text{ m}$, outer corona $24.0\text{ m}$); Mercury ($0.39\text{ AU}$): $40.6\text{ m}$; Earth ($1.0\text{ AU}$): $75.0\text{ m}$; Mars ($1.52\text{ AU}$): $93.7\text{ m}$; $R_1 = R_{\text{sim}}(2.0) = 107.22\text{ m}$.
2. **Regime II — Asteroid Belt, Gas/Ice Giants & Kuiper Belt ($2.0 < r \le 55.0\text{ AU}$):**
   $$R_{\text{sim}}(r) = R_1 + S_2 \ln\!\left(\frac{r}{2.0}\right), \quad S_2 = 115.0\text{ m}$$
   - Main Belt ($2.2\text{–}3.3\text{ AU}$): $118\text{–}165\text{ m}$; Jupiter ($5.20\text{ AU}$): $217.2\text{ m}$; Saturn ($9.58\text{ AU}$): $287.4\text{ m}$; Uranus ($19.22\text{ AU}$): $367.4\text{ m}$; Neptune ($30.11\text{ AU}$): $419.0\text{ m}$; Kuiper Belt ($33\text{–}52\text{ AU}$): $430\text{–}482\text{ m}$; $R_2 = R_{\text{sim}}(55.0) = 488.32\text{ m}$.
3. **Regime III — Heliosheath, Termination Shock, Heliopause & VLISM ($55.0 < r \le 250.0\text{ AU}$):**
   $$R_{\text{sim}}(r) = R_2 + S_3 \ln\!\left(\frac{r}{55.0}\right), \quad S_3 = 210.0\text{ m}$$
   - Termination Shock ($84\text{–}94\text{ AU}$): $577\text{–}601\text{ m}$; Heliopause ($119\text{–}122\text{ AU}$): $650\text{–}655\text{ m}$; Voyager 2 (2026, $139.5\text{ AU}$): $684\text{ m}$; Voyager 1 (2026, $166.4\text{ AU}$): $721\text{ m}$; $R_3 = R_{\text{sim}}(250.0) = 806.38\text{ m}$.
4. **Regime IV — Oort Cloud & Deep Interstellar Space ($r > 250.0\text{ AU}$):**
   $$R_{\text{sim}}(r) = R_3 + S_4 \ln\!\left(\frac{r}{250.0}\right), \quad S_4 = 110.0\text{ m}$$
   - Oort Cloud ($1,000\text{–}25,000\text{ AU}$): $959\text{–}1,313\text{ m}$. Maximum stage extent stays within $1,350\text{ m}$ (`float32` precision $< 0.15\text{ mm}$).

**Apparent Visual Radii:** Compressed via $a_{\text{vis}} = 2.60\text{ m} \cdot (R_{\text{body}} / R_{\oplus})^{0.42}$ (clamped so moons/rings never intersect parent orbits), while Voyager 1 & 2 maintain true 1:1 physical geometry ($3.7\text{ m}$ dish) in Chase/Staging views with an adaptive distant telemetry beacon for Solar-System-wide camera modes.
