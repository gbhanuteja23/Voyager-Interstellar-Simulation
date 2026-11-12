"""Interactive `omni.ui` Mission Control Window & Headless-Compatible Mission Controller."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from voyager_interstellar_sim.config import (
    CAMERA_MODES,
    MISSION_PHASES,
    PhasePreset,
)
from voyager_interstellar_sim.kinematics import (
    SimulationFrameState,
    evaluate_simulation_frame,
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
            # Cap effective step so 1000x warp remains smooth rather than teleporting in 1 frame
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


class VoyagerMissionControlWindow:
    """Rich NASA/JPL-styled `omni.ui` HUD window for interactive mission control."""

    def __init__(self, controller: MissionController) -> None:
        import omni.ui as ui

        self.ui = ui
        self.ctrl = controller
        self._updating_slider = False

        self.window = ui.Window(
            "VOYAGER INTERSTELLAR MISSION CONTROL",
            width=450,
            height=680,
            flags=ui.WINDOW_FLAGS_NO_SCROLLBAR,
        )
        self.window.position_x = 24
        self.window.position_y = 48

        self._lbl_sc: Optional[ui.Label] = None
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
            with ui.ScrollingFrame(
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF
            ):
                with ui.VStack(spacing=8, style={"margin": 10}):
                    # Header Banner
                    ui.Label(
                        "NASA / JPL DEEP SPACE NETWORK — VOYAGER 1 & 2",
                        height=20,
                        style={"color": 0xFF55C8FF, "font_size": 15},
                    )
                    ui.Separator(height=2)

                    # 1. Spacecraft Selector
                    with ui.HStack(height=28, spacing=8):
                        ui.Label("ACTIVE PROBE:", width=110, style={"color": 0xFFCCCCCC})
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

                    # 2. Live Telemetry Readout Card
                    with ui.Frame(
                        style={
                            "background_color": 0xDD14181F,
                            "border_color": 0xFF446688,
                            "border_width": 1.0,
                            "border_radius": 6.0,
                            "padding": 8,
                        }
                    ):
                        with ui.VStack(spacing=4):
                            self._lbl_sc = ui.Label("PROBE: VOYAGER 1", height=18, style={"color": 0xFF44D8FF, "font_size": 14})
                            self._lbl_year = ui.Label("MISSION EPOCH: 1977.68", height=18, style={"color": 0xFFFFFFFF})
                            self._lbl_dist = ui.Label("HELIOCENTRIC DIST: 1.00 AU (0.150 B km)", height=18, style={"color": 0xFFE8E8E8})
                            self._lbl_vel = ui.Label("HELIOCENTRIC VEL:  38.60 km/s", height=18, style={"color": 0xFFE8E8E8})
                            self._lbl_rtlt = ui.Label("ROUND-TRIP LIGHT:  00h 00m 01s", height=18, style={"color": 0xFFE8E8E8})
                            self._lbl_hga = ui.Label("HGA EARTH LOCK:    99.9% | BOOMS: DEPLOYED", height=18, style={"color": 0xFFA8E6A0})
                            self._lbl_regime = ui.Label("REGIME: Supersonic Solar Wind", height=20, style={"color": 0xFF45C8FF, "font_size": 13})

                    # 3. Continuous Timeline Scrubber
                    ui.Label("CONTINUOUS MISSION TIMELINE SCRUBBER (1977 -> OORT CLOUD):", height=18, style={"color": 0xFFBBBBBB})
                    slider = ui.FloatSlider(self._slider_model, min=0.0, max=1.0, step=0.0005, height=22)
                    self._slider_model.add_value_changed_fn(self._on_slider_changed)
                    _ = slider

                    # 4. Playback Speed Multiplier
                    with ui.HStack(height=20):
                        ui.Label("TIME-WARP MULTIPLIER:", style={"color": 0xFFBBBBBB})
                        self._lbl_speed_active = ui.Label("ACTIVE: 1x", style={"color": 0xFF55FF99})
                    with ui.HStack(height=28, spacing=5):
                        for label_txt, mult in [("Pause", 0.0), ("1x", 1.0), ("10x", 10.0), ("100x", 100.0), ("1000x", 1000.0)]:
                            ui.Button(
                                label_txt,
                                clicked_fn=lambda m=mult: self._on_speed_clicked(m),
                                style=btn_style,
                            )

                    ui.Separator(height=4)

                    # 5. Mission Phase Jump Buttons (6 Presets)
                    ui.Label("MISSION PHASE JUMP CHECKPOINTS:", height=18, style={"color": 0xFF55C8FF})
                    for row in range(0, len(MISSION_PHASES), 2):
                        with ui.HStack(height=30, spacing=6):
                            for ph in MISSION_PHASES[row : row + 2]:
                                ui.Button(
                                    ph.short_label,
                                    tooltip=ph.description,
                                    clicked_fn=lambda idx=ph.index: self.ctrl.jump_to_phase(idx),
                                    style=btn_style,
                                )

                    ui.Separator(height=4)

                    # 6. Multi-Camera Selector (5 Modes)
                    with ui.HStack(height=20):
                        ui.Label("MULTI-CAMERA RIG SELECTOR:", style={"color": 0xFF55C8FF})
                        self._lbl_cam_active = ui.Label(f"[{self.ctrl.active_camera}]", style={"color": 0xFFFFCC55})

                    for cam_id, cam_title in CAMERA_MODES:
                        ui.Button(
                            cam_title,
                            height=26,
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
            self._lbl_regime.style = {"color": sc.plasma_color_hex, "font_size": 13}
        if self._lbl_cam_active is not None:
            self._lbl_cam_active.text = f"[{self.ctrl.active_camera}]"
        if self._lbl_speed_active is not None:
            m = self.ctrl.speed_multiplier
            self._lbl_speed_active.text = "PAUSED" if m == 0.0 else f"ACTIVE: {int(m)}x"
