"""Interactive `omni.ui` Mission Control Window, In-Viewport 3D AU & Location HUD Overlay,
and Headless-Compatible Mission Controller.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

from voyager_interstellar_sim.config import (
    CAMERA_MODES,
    MISSION_PHASES,
    PLANETS,
    PhasePreset,
)
from voyager_interstellar_sim.kinematics import (
    SimulationFrameState,
    evaluate_simulation_frame,
    get_trajectory_au_milestones,
)
from voyager_interstellar_sim.scene_builder import (
    VoyagerSceneManager,
    set_active_viewport_camera,
)


class MissionController:
    """State controller driving timeline scrubbing, phase jumps, speed warp, and camera selection."""

    BASE_SCRUB_RATE_PER_SEC: float = 0.0075  # 1x cinematic speed (~18.5s for full Launch+Deployment sequence)

    def __init__(self, scene_mgr: VoyagerSceneManager, headless: bool = False) -> None:
        self.scene_mgr: VoyagerSceneManager = scene_mgr
        self.headless: bool = headless

        self.active_spacecraft: int = 1
        self.scrub_s: float = 0.010
        self.speed_multiplier: float = 1.0  # 0.0=Pause, 1.0, 10.0, 100.0, 1000.0
        self.active_camera: str = "RocketStagingCam"
        self.auto_camera_on_phase: bool = True
        self.current_phase_label: str = MISSION_PHASES[0].label

        self.last_state: SimulationFrameState = evaluate_simulation_frame(
            self.scrub_s, active_spacecraft=self.active_spacecraft
        )
        self._ui_sync_callbacks: List[Callable[[SimulationFrameState], None]] = []

    def register_ui_callback(self, cb: Callable[[SimulationFrameState], None]) -> None:
        self._ui_sync_callbacks.append(cb)

    def set_spacecraft(self, sc_id: int) -> None:
        self.active_spacecraft = 1 if sc_id == 1 else 2
        self.step(0.0)

    def set_speed(self, mult: float) -> None:
        if mult > 0.0 and self.scrub_s >= 0.999:
            self.scrub_s = 0.0
        self.speed_multiplier = float(mult)

    def set_camera(self, cam_name: str) -> None:
        self.active_camera = cam_name
        if not self.headless:
            set_active_viewport_camera(f"/World/Cameras/{cam_name}")
        self.scene_mgr.update_cameras(
            self.last_state,
            active_spacecraft=self.active_spacecraft,
            active_camera=self.active_camera,
        )
        for cb in self._ui_sync_callbacks:
            cb(self.last_state)

    def set_scrub(self, s_val: float) -> None:
        self.scrub_s = max(0.0, min(1.0, float(s_val)))
        self.step(0.0)

    def jump_to_phase(self, phase_index_1based: int) -> PhasePreset:
        preset = MISSION_PHASES[max(0, min(len(MISSION_PHASES) - 1, phase_index_1based - 1))]
        self.current_phase_label = preset.label
        self.scrub_s = preset.s_voyager1 if self.active_spacecraft == 1 else preset.s_voyager2
        if self.auto_camera_on_phase:
            self.active_camera = preset.recommended_camera
            if not self.headless:
                set_active_viewport_camera(f"/World/Cameras/{self.active_camera}")
        self.step(0.0)
        return preset

    def step(self, dt_wall: float) -> SimulationFrameState:
        if dt_wall > 0.0 and self.speed_multiplier > 0.0:
            rate = self.BASE_SCRUB_RATE_PER_SEC * (self.speed_multiplier ** 0.58)
            self.scrub_s = min(1.0, self.scrub_s + rate * min(dt_wall, 0.1))
            if self.scrub_s >= 1.0:
                self.speed_multiplier = 0.0

        self.last_state = evaluate_simulation_frame(
            self.scrub_s, active_spacecraft=self.active_spacecraft
        )
        self.scene_mgr.apply_frame_state(
            self.last_state,
            active_spacecraft=self.active_spacecraft,
            active_camera=self.active_camera,
            dt_wall=dt_wall,
        )
        for cb in self._ui_sync_callbacks:
            cb(self.last_state)
        return self.last_state


class VoyagerViewportHUD:
    """In-Viewport HUD Overlay displaying live AU distance, Space Location banner,
    AU progress bar, and 3D-projected floating trajectory & spacecraft callout markers.
    """

    MAX_CALLOUT_SLOTS: int = 9

    def __init__(self, controller: MissionController) -> None:
        import omni.kit.viewport.utility as vp_util
        import omni.ui as ui

        self.ui = ui
        self.ctrl = controller
        self._vp_win = vp_util.get_active_viewport_window()
        self._vp_frame = None

        self._banner_probe_loc: Optional[ui.Label] = None
        self._banner_au_metrics: Optional[ui.Label] = None
        self._banner_regime: Optional[ui.Label] = None
        self._au_progress_model = ui.SimpleFloatModel(0.0)

        # 3D-tracked floating callout widgets: (placer, frame, title_lbl, sub_lbl)
        self._sc_callout: Optional[Tuple[object, object, object, object]] = None
        self._wp_callouts: List[Tuple[object, object, object]] = []

        if self._vp_win is not None:
            self._vp_frame = self._vp_win.get_frame("voyager.interstellar.viewport.hud.v2")
            self._build_viewport_overlay()
            self.ctrl.register_ui_callback(self._on_frame_updated)
            self._on_frame_updated(self.ctrl.last_state)

    def _build_viewport_overlay(self) -> None:
        ui = self.ui
        if self._vp_frame is None:
            return

        with self._vp_frame:
            with ui.ZStack():
                # 1. Floating 3D-Projected Trajectory Waypoint & Spacecraft Callout Layer
                for _ in range(self.MAX_CALLOUT_SLOTS):
                    pl = ui.Placer(offset_x=0, offset_y=0)
                    with pl:
                        fr = ui.Frame(
                            visible=False,
                            width=0,
                            height=0,
                            style={
                                "background_color": 0xAA0D1520,
                                "border_color": 0xAA4499CC,
                                "border_width": 1.0,
                                "border_radius": 4.0,
                                "padding": 4,
                            },
                        )
                        with fr:
                            lbl = ui.Label(
                                "",
                                width=0,
                                height=0,
                                style={"color": 0xFFD8F4FF, "font_size": 12},
                            )
                    self._wp_callouts.append((pl, fr, lbl))

                # Primary Active Spacecraft 3D Tracking Tag
                sc_pl = ui.Placer(offset_x=0, offset_y=0)
                with sc_pl:
                    sc_fr = ui.Frame(
                        visible=False,
                        width=0,
                        height=0,
                        style={
                            "background_color": 0xCC121A24,
                            "border_color": 0xFFFFCC44,
                            "border_width": 1.5,
                            "border_radius": 5.0,
                            "padding": 5,
                        },
                    )
                    with sc_fr:
                        with ui.VStack(spacing=2, width=0, height=0):
                            sc_t = ui.Label(
                                "",
                                width=0,
                                height=0,
                                style={"color": 0xFFFFDD66, "font_size": 13},
                            )
                            sc_s = ui.Label(
                                "",
                                width=0,
                                height=0,
                                style={"color": 0xFFA8E8FF, "font_size": 11},
                            )
                self._sc_callout = (sc_pl, sc_fr, sc_t, sc_s)

                # 2. Top-Center Fixed In-Viewport Telemetry & Space Location HUD Ribbon
                with ui.VStack(spacing=0):
                    ui.Spacer(height=36)
                    with ui.HStack(height=86):
                        ui.Spacer()
                        with ui.Frame(
                            width=780,
                            height=84,
                            style={
                                "background_color": 0xCC0B111A,
                                "border_color": 0xCC3B88C4,
                                "border_width": 1.5,
                                "border_radius": 8.0,
                                "padding": 8,
                            },
                        ):
                            with ui.VStack(spacing=4):
                                self._banner_probe_loc = ui.Label(
                                    "VOYAGER 1  |  LOCATION: Earth Launch Ascent",
                                    height=20,
                                    alignment=ui.Alignment.CENTER,
                                    style={"color": 0xFFFFD866, "font_size": 15},
                                )
                                self._banner_au_metrics = ui.Label(
                                    "DISTANCE: 1.000 AU (0.150 B km)   |   EARTH: 0.001 AU   |   SPEED: 38.6 km/s   |   YEAR: 1977.68",
                                    height=18,
                                    alignment=ui.Alignment.CENTER,
                                    style={"color": 0xFFFFFFFF, "font_size": 13},
                                )
                                with ui.HStack(height=14, spacing=6):
                                    ui.Label("1 AU Earth", width=68, style={"color": 0xFF88CCFF, "font_size": 11})
                                    ui.ProgressBar(
                                        self._au_progress_model,
                                        height=10,
                                        style={
                                            "color": 0xFF38D4FF,
                                            "background_color": 0x881E2B3A,
                                            "border_radius": 3.0,
                                        },
                                    )
                                    ui.Label("Oort (10k AU)", width=82, style={"color": 0xFFFFCC77, "font_size": 11})
                                self._banner_regime = ui.Label(
                                    "1 AU Earth  |  5.2 AU Jupiter  |  9.6 AU Saturn  |  30.1 AU Neptune  |  94 AU Term.Shock  |  121 AU Heliopause  |  2000+ AU Oort",
                                    height=14,
                                    alignment=ui.Alignment.CENTER,
                                    style={"color": 0xFF99BBD8, "font_size": 11},
                                )
                        ui.Spacer()
                    ui.Spacer()

    def _project_world_to_viewport(
        self,
        world_xyz: Tuple[float, float, float],
        world_to_cam,
        tan_half_fov_x: float,
        tan_half_fov_y: float,
        vp_w: float,
        vp_h: float,
    ) -> Optional[Tuple[float, float]]:
        from pxr import Gf

        p_cam = world_to_cam.Transform(Gf.Vec3d(float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])))
        z_depth = -float(p_cam[2])
        if z_depth <= 0.08:
            return None
        ndc_x = (float(p_cam[0]) / z_depth) / max(tan_half_fov_x, 1e-6)
        ndc_y = (float(p_cam[1]) / z_depth) / max(tan_half_fov_y, 1e-6)
        if abs(ndc_x) > 0.93 or abs(ndc_y) > 0.90:
            return None
        sx = (0.5 + 0.5 * ndc_x) * vp_w
        sy = (0.5 - 0.5 * ndc_y) * vp_h
        # Avoid overlapping top HUD banner
        if sy < 128.0 and abs(ndc_x) < 0.62:
            return None
        return (sx, sy)

    def _on_frame_updated(self, state: SimulationFrameState) -> None:
        if self._vp_frame is None:
            return
        sc_id = self.ctrl.active_spacecraft
        sc = state.voyager1 if sc_id == 1 else state.voyager2
        probe_tag = f"VOYAGER {sc_id}"

        yr_int = int(state.year)
        month = max(1, min(12, int((state.year - yr_int) * 12.0) + 1))

        if self._banner_probe_loc is not None:
            self._banner_probe_loc.text = f">> {probe_tag}   |   SPACE LOCATION: {sc.location_label}"
        if self._banner_au_metrics is not None:
            self._banner_au_metrics.text = (
                f"SUN DISTANCE: {sc.heliocentric_au:,.2f} AU ({sc.heliocentric_billion_km:,.2f} B km)   |   "
                f"EARTH DIST: {sc.earth_distance_au:,.2f} AU   |   "
                f"VEL: {sc.velocity_km_s:.2f} km/s   |   EPOCH: {yr_int}-{month:02d}"
            )
        if self._banner_regime is not None:
            self._banner_regime.text = (
                f"REGIME: {sc.plasma_regime}   |   "
                "Scale: [1 AU Earth -> 5.2 AU Jup -> 9.6 AU Sat -> 30 AU Nep -> 94 AU Shock -> 121 AU Heliopause -> Oort]"
            )
            self._banner_regime.style = {"color": sc.plasma_color_hex, "font_size": 11}

        # Log-AU progress in [0.0, 1.0] from 1.0 AU to 10,100 AU
        log_prog = math.log10(max(1.0, sc.heliocentric_au)) / math.log10(10100.0)
        self._au_progress_model.set_value(max(0.0, min(1.0, log_prog)))

        # Project 3D Spacecraft & Trajectory AU Milestones onto Viewport Canvas
        try:
            from pxr import Usd, UsdGeom

            vp_w = float(self._vp_frame.computed_width)
            vp_h = float(self._vp_frame.computed_height)
            if vp_w < 160.0 or vp_h < 120.0:
                return

            stage = self.ctrl.scene_mgr.stage
            cam_path_str = f"/World/Cameras/{self.ctrl.active_camera}"
            if self._vp_win is not None and getattr(self._vp_win, "viewport_api", None) is not None:
                vp_cam = str(self._vp_win.viewport_api.camera_path)
                if vp_cam:
                    cam_path_str = vp_cam

            cam_prim = stage.GetPrimAtPath(cam_path_str)
            if not cam_prim.IsValid():
                return

            usd_cam = UsdGeom.Camera(cam_prim)
            focal_mm = float(usd_cam.GetFocalLengthAttr().Get() or 24.0)
            h_ap_mm = float(usd_cam.GetHorizontalApertureAttr().Get() or 36.0)
            aspect = vp_w / max(vp_h, 1.0)
            tan_half_x = (0.5 * h_ap_mm) / max(focal_mm, 1.0)
            tan_half_y = tan_half_x / max(aspect, 0.2)

            xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
            world_to_cam = xcache.GetLocalToWorldTransform(cam_prim).GetInverse()

            occupied_pts: List[Tuple[float, float]] = []

            # A) Active Voyager 3D Callout Tag
            if self._sc_callout is not None:
                sc_pl, sc_fr, sc_t, sc_s = self._sc_callout
                sc_proj = self._project_world_to_viewport(
                    sc.pos_sim, world_to_cam, tan_half_x, tan_half_y, vp_w, vp_h
                )
                if sc_proj is not None:
                    px = min(vp_w - 235.0, max(12.0, sc_proj[0] + 22.0))
                    py = min(vp_h - 56.0, max(106.0, sc_proj[1] - 42.0))
                    sc_pl.offset_x = px
                    sc_pl.offset_y = py
                    sc_t.text = f"[+] {probe_tag}  |  {sc.heliocentric_au:,.2f} AU"
                    sc_s.text = sc.location_label
                    sc_fr.visible = True
                    occupied_pts.append((px, py))
                else:
                    sc_fr.visible = False

            # B) Candidate 3D AU Milestones (Active Trajectory Waypoints + Key Planets)
            candidates: List[Tuple[str, Tuple[float, float, float]]] = []
            for p_spec in PLANETS:
                if p_spec.name in ("Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"):
                    p_pos = state.planet_positions_sim[p_spec.name]
                    candidates.append((f"* {p_spec.name} ({p_spec.a_au:.2f} AU)", p_pos))

            for m_label, m_au, m_pos in get_trajectory_au_milestones(sc_id):
                if m_au >= 2.0:
                    candidates.append((f"+ {m_label} | {m_au:,.1f} AU", m_pos))

            slot_idx = 0
            for label_txt, w_pos in candidates:
                if slot_idx >= len(self._wp_callouts):
                    break
                proj = self._project_world_to_viewport(
                    w_pos, world_to_cam, tan_half_x, tan_half_y, vp_w, vp_h
                )
                if proj is None:
                    continue
                lx = min(vp_w - 195.0, max(10.0, proj[0] + 10.0))
                ly = min(vp_h - 34.0, max(106.0, proj[1] - 12.0))
                # Screen-space de-clutter check (minimum 92 px separation)
                too_close = any(((lx - ox) ** 2 + (ly - oy) ** 2) < (92.0 ** 2) for ox, oy in occupied_pts)
                if too_close:
                    continue
                pl_w, fr_w, lbl_w = self._wp_callouts[slot_idx]
                pl_w.offset_x = lx
                pl_w.offset_y = ly
                lbl_w.text = label_txt
                fr_w.visible = True
                occupied_pts.append((lx, ly))
                slot_idx += 1

            # Hide unused callout slots
            for rem_idx in range(slot_idx, len(self._wp_callouts)):
                self._wp_callouts[rem_idx][1].visible = False
        except Exception:
            pass


class VoyagerMissionControlWindow:
    """Rich NASA/JPL-styled `omni.ui` Mission Control Window with pinned live telemetry."""

    def __init__(self, controller: MissionController) -> None:
        import omni.ui as ui

        self.ui = ui
        self.ctrl = controller
        self._updating_slider = False

        self.window = ui.Window(
            "VOYAGER INTERSTELLAR MISSION CONTROL",
            width=455,
            height=690,
            flags=ui.WINDOW_FLAGS_NO_SCROLLBAR,
        )
        self.window.position_x = 20
        self.window.position_y = 112

        self._lbl_sc: Optional[ui.Label] = None
        self._lbl_loc: Optional[ui.Label] = None
        self._lbl_year: Optional[ui.Label] = None
        self._lbl_dist: Optional[ui.Label] = None
        self._lbl_vel: Optional[ui.Label] = None
        self._lbl_rtlt: Optional[ui.Label] = None
        self._lbl_regime: Optional[ui.Label] = None
        self._lbl_hga: Optional[ui.Label] = None
        self._lbl_cam_active: Optional[ui.Label] = None
        self._lbl_speed_active: Optional[ui.Label] = None
        self._slider_model = ui.SimpleFloatModel(self.ctrl.scrub_s)

        self._build_ui()
        self.ctrl.register_ui_callback(self._on_frame_updated)
        set_active_viewport_camera(f"/World/Cameras/{self.ctrl.active_camera}")
        self._on_frame_updated(self.ctrl.last_state)

    def _build_ui(self) -> None:
        ui = self.ui
        btn_style = {
            "Button": {
                "background_color": 0xFF2B241C,
                "border_color": 0xFF886633,
                "border_width": 1.0,
                "border_radius": 4.0,
                "padding": 5,
            },
            "Button:hovered": {"background_color": 0xFF4A3B28},
            "Button:pressed": {"background_color": 0xFF705832},
        }
        cam_btn_style = {
            "Button": {
                "background_color": 0xFF222A35,
                "border_color": 0xFF557799,
                "border_width": 1.0,
                "border_radius": 4.0,
                "padding": 4,
            },
            "Button:hovered": {"background_color": 0xFF34455A},
        }

        with self.window.frame:
            with ui.VStack(spacing=6, style={"margin": 8}):
                # Pinned Header + Live Telemetry Card (NEVER scrolls out of view!)
                ui.Label(
                    "NASA / JPL DEEP SPACE NETWORK -- VOYAGER 1 & 2",
                    height=18,
                    style={"color": 0xFF55C8FF, "font_size": 14},
                )
                with ui.Frame(
                    height=168,
                    style={
                        "background_color": 0xEE111720,
                        "border_color": 0xFF4477AA,
                        "border_width": 1.2,
                        "border_radius": 6.0,
                        "padding": 7,
                    },
                ):
                    with ui.VStack(spacing=3):
                        self._lbl_sc = ui.Label("PROBE: VOYAGER 1", height=17, style={"color": 0xFF44D8FF, "font_size": 13})
                        self._lbl_loc = ui.Label("LOCATION: Earth Launch Ascent", height=18, style={"color": 0xFFFFD866, "font_size": 13})
                        self._lbl_year = ui.Label("MISSION YEAR: 1977.68", height=16, style={"color": 0xFFFFFFFF})
                        self._lbl_dist = ui.Label("SUN DISTANCE: 1.00 AU (0.150 B km)", height=16, style={"color": 0xFFE8E8E8})
                        self._lbl_vel = ui.Label("HELIOCENTRIC VEL: 38.60 km/s", height=16, style={"color": 0xFFE8E8E8})
                        self._lbl_rtlt = ui.Label("ROUND-TRIP LIGHT: 00h 00m 01s", height=16, style={"color": 0xFFE8E8E8})
                        self._lbl_hga = ui.Label("HGA EARTH ALIGN: 100.0% | BOOMS: 100%", height=16, style={"color": 0xFFA8E6A0})
                        self._lbl_regime = ui.Label("REGIME: Supersonic Solar Wind", height=17, style={"color": 0xFF45C8FF, "font_size": 12})

                # Scrollable Interactive Controls Below Pinned Telemetry
                with ui.ScrollingFrame(
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF
                ):
                    with ui.VStack(spacing=7, style={"margin": 2}):
                        # 1. Spacecraft Selector
                        with ui.HStack(height=28, spacing=6):
                            ui.Label("ACTIVE PROBE:", width=98, style={"color": 0xFFCCCCCC})
                            ui.Button(
                                "VOYAGER 1 (North / Titan)",
                                clicked_fn=lambda: self.ctrl.set_spacecraft(1),
                                style=btn_style,
                            )
                            ui.Button(
                                "VOYAGER 2 (Grand Tour)",
                                clicked_fn=lambda: self.ctrl.set_spacecraft(2),
                                style=btn_style,
                            )

                        # 2. Continuous Timeline Scrubber
                        ui.Label("CONTINUOUS MISSION TIMELINE SCRUBBER (1977 -> OORT CLOUD):", height=16, style={"color": 0xFFBBBBBB})
                        ui.FloatSlider(self._slider_model, min=0.0, max=1.0, step=0.0005, height=22)
                        self._slider_model.add_value_changed_fn(self._on_slider_changed)

                        # 3. Playback Speed Multiplier
                        with ui.HStack(height=18):
                            ui.Label("TIME-WARP MULTIPLIER:", style={"color": 0xFFBBBBBB})
                            self._lbl_speed_active = ui.Label("ACTIVE: 1x", style={"color": 0xFF55FF99})
                        with ui.HStack(height=27, spacing=5):
                            for label_txt, mult in [("Pause", 0.0), ("1x", 1.0), ("10x", 10.0), ("100x", 100.0), ("1000x", 1000.0)]:
                                ui.Button(
                                    label_txt,
                                    clicked_fn=lambda m=mult: self._on_speed_clicked(m),
                                    style=btn_style,
                                )

                        ui.Separator(height=4)

                        # 4. Mission Phase Jump Buttons (6 Presets)
                        ui.Label("MISSION PHASE JUMP CHECKPOINTS:", height=17, style={"color": 0xFF55C8FF})
                        for row in range(0, len(MISSION_PHASES), 2):
                            with ui.HStack(height=28, spacing=6):
                                for ph in MISSION_PHASES[row : row + 2]:
                                    ui.Button(
                                        ph.short_label,
                                        tooltip=ph.description,
                                        clicked_fn=lambda idx=ph.index: self.ctrl.jump_to_phase(idx),
                                        style=btn_style,
                                    )

                        ui.Separator(height=4)

                        # 5. Multi-Camera Selector (5 Modes)
                        with ui.HStack(height=18):
                            ui.Label("MULTI-CAMERA RIG SELECTOR:", style={"color": 0xFF55C8FF})
                            self._lbl_cam_active = ui.Label(f"[{self.ctrl.active_camera}]", style={"color": 0xFFFFCC55})

                        for cam_id, cam_title in CAMERA_MODES:
                            ui.Button(
                                cam_title,
                                height=25,
                                clicked_fn=lambda cid=cam_id: self.ctrl.set_camera(cid),
                                style=cam_btn_style,
                            )

    def _on_slider_changed(self, model) -> None:
        if self._updating_slider:
            return
        val = model.get_value_as_float()
        self.ctrl.set_scrub(val)

    def _on_speed_clicked(self, mult: float) -> None:
        self.ctrl.set_speed(mult)
        if self._lbl_speed_active is not None:
            txt = "PAUSED" if mult == 0.0 else f"ACTIVE: {int(mult)}x"
            self._lbl_speed_active.text = txt

    def _on_frame_updated(self, state: SimulationFrameState) -> None:
        sc = state.voyager1 if self.ctrl.active_spacecraft == 1 else state.voyager2
        sc_title = "VOYAGER 1 (SSN-1977-084A)" if self.ctrl.active_spacecraft == 1 else "VOYAGER 2 (SSN-1977-076A)"

        self._updating_slider = True
        try:
            self._slider_model.set_value(state.scrub_s)
        finally:
            self._updating_slider = False

        if self._lbl_sc is not None:
            self._lbl_sc.text = f"PROBE: {sc_title}"
        if self._lbl_loc is not None:
            self._lbl_loc.text = f"LOCATION: {sc.location_label}"
        if self._lbl_year is not None:
            yr_int = int(state.year)
            month = max(1, min(12, int((state.year - yr_int) * 12.0) + 1))
            self._lbl_year.text = f"MISSION YEAR:      {state.year:7.2f}  ({yr_int}-{month:02d})"
        if self._lbl_dist is not None:
            self._lbl_dist.text = (
                f"HELIOCENTRIC DIST: {sc.heliocentric_au:7.2f} AU  ({sc.heliocentric_billion_km:6.3f} B km)"
            )
        if self._lbl_vel is not None:
            self._lbl_vel.text = f"HELIOCENTRIC VEL:  {sc.velocity_km_s:6.2f} km/s"
        if self._lbl_rtlt is not None:
            self._lbl_rtlt.text = f"ROUND-TRIP LIGHT:  {sc.round_trip_light_time_str}"
        if self._lbl_hga is not None:
            boom_pct = int(round(sc.boom_deploy_01 * 100.0))
            lock_pct = max(0.0, sc.earth_align_dot * 100.0)
            self._lbl_hga.text = f"HGA EARTH ALIGN:   {lock_pct:5.1f}%  |  BOOMS: {boom_pct}%"
        if self._lbl_regime is not None:
            self._lbl_regime.text = f"PLASMA REGIME: {sc.plasma_regime}"
            self._lbl_regime.style = {"color": sc.plasma_color_hex, "font_size": 12}
        if self._lbl_cam_active is not None:
            self._lbl_cam_active.text = f"[{self.ctrl.active_camera}]"
        if self._lbl_speed_active is not None:
            m = self.ctrl.speed_multiplier
            self._lbl_speed_active.text = "PAUSED" if m == 0.0 else f"ACTIVE: {int(m)}x"
