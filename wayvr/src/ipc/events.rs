use crate::backend::wayvr::{self, WayVRTask, WvrServerState};
use wayvr_ipc::packet_client::{HandsfreeMode, HandsfreeParams};
use wlx_common::config::HandsfreePointer;

use crate::{
    backend::{
        self,
        task::{InputTask, OverlayTask, TaskType},
    },
    ipc::signal::WayVRSignal,
    state::AppState,
    windowing::manager::OverlayWindowManager,
};

fn process_tick_tasks(
    tick_tasks: Vec<backend::wayvr::TickTask>,
    server_state: &mut WvrServerState,
) {
    for tick_task in tick_tasks {
        match tick_task {
            backend::wayvr::TickTask::NewExternalProcess(request) => {
                log::info!("Registering external process with PID {}", request.pid);
                server_state.add_external_process(request.pid);
                server_state.manager.add_client(wayvr::client::WayVRClient {
                    client: request.client,
                    pid: request.pid,
                });
            }
        }
    }
}

/// True while a backdrop set over IPC (`WlxSetSkybox`) is drawn even though the main app (passthrough) is visible.
pub static SKYBOX_OVER_MAIN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `WlxDeskLayout`: the first `screens` screens (desktop captures first, then WayVR app panels and mirrors; 0 = all)
/// are shown in an arc in front of the head with the given curvature and size, the others are hidden (not closed).
fn desk_layout<O>(
    app: &mut AppState,
    overlays: &mut OverlayWindowManager<O>,
    p: wayvr_ipc::packet_client::WlxDeskLayoutParams,
) where
    O: Default,
{
    use crate::backend::task::ToggleMode;
    use crate::windowing::{OverlaySelector, grabmath, window::OverlayCategory};

    let mut ids: Vec<_> = overlays
        .iter()
        .filter_map(|(id, o)| {
            let rank = match o.config.category {
                OverlayCategory::Screen => 0,
                OverlayCategory::WayVR => 1,
                OverlayCategory::Mirror => 2,
                _ => return None,
            };
            Some((rank, id))
        })
        .collect();
    ids.sort_by_key(|(rank, _)| *rank); // stable: keeps the creation order inside a category
    let n = if p.screens == 0 { ids.len() } else { (p.screens as usize).min(ids.len()) };
    let hmd = app.input_state.hmd;
    let curvature = p.curvature.clamp(0.0, 1.0);
    let width = if p.width.is_finite() { p.width.clamp(0.3, 6.0) } else { 2.0 };
    let distance = if p.distance.is_finite() { p.distance.clamp(0.4, 8.0) } else { 2.5 };
    log::info!("desk layout: {n} of {} screens, curvature {curvature}, width {width} m, distance {distance} m", ids.len());
    for (i, (_, id)) in ids.into_iter().enumerate() {
        if i >= n {
            app.tasks.enqueue(TaskType::Overlay(OverlayTask::ToggleOverlay(
                OverlaySelector::Id(id),
                ToggleMode::EnsureOff,
            )));
            continue;
        }
        app.tasks.enqueue(TaskType::Overlay(OverlayTask::ToggleOverlay(
            OverlaySelector::Id(id),
            ToggleMode::EnsureOn,
        )));
        let t = grabmath::desk_arc(&hmd, i, n, width, distance);
        app.tasks.enqueue(TaskType::Overlay(OverlayTask::Modify(
            OverlaySelector::Id(id),
            Box::new(move |app, cfg| {
                if let Some(state) = cfg.active_state.as_mut() {
                    state.positioning = wlx_common::windowing::Positioning::Static;
                    state.curvature = (curvature > 0.0).then_some(curvature);
                    state.transform = t;
                    crate::windowing::window::save_transform(state, app);
                    cfg.dirty = true;
                }
            }),
        )));
    }
}

/// `WlxDeskScreens`: place/show/hide named screens: main in front of the head, left/right/up edge to edge around it.
fn desk_screens<O>(
    app: &mut AppState,
    overlays: &mut OverlayWindowManager<O>,
    p: wayvr_ipc::packet_client::WlxDeskScreensParams,
) where
    O: Default,
{
    use crate::backend::task::ToggleMode;
    use crate::windowing::{OverlayID, OverlaySelector, grabmath, window::OverlayCategory};
    use glam::{Affine3A, Vec3A};
    use wayvr_ipc::packet_client::WlxDeskSlot;

    // displayed (width, height) in metres of a quad, as the OpenXR backend computes it
    fn quad(t: &Affine3A, extent: [u32; 2]) -> Option<(f32, f32)> {
        if extent[0] == 0 || extent[1] == 0 {
            return None;
        }
        let aspect = extent[0] as f32 / extent[1] as f32;
        Some(if aspect < 1.0 {
            let major = t.matrix3.col(0).length();
            (major * aspect, major)
        } else {
            let major = t.matrix3.col(1).length();
            (major, major / aspect)
        })
    }
    fn is_screenish(c: OverlayCategory) -> bool {
        matches!(c, OverlayCategory::Screen | OverlayCategory::WayVR | OverlayCategory::Mirror)
    }

    crate::overlays::screen_numbers::note_desk_screens(
        &p.screens.iter().map(|s| (s.name.as_str().into(), s.slot, s.visible)).collect::<Vec<_>>(),
        p.place,
    );

    let hmd = app.input_state.hmd;
    let pivot = Vec3A::from(hmd.translation);
    let curvature = if p.curvature.is_finite() { p.curvature.clamp(0.0, 1.0) } else { 0.0 };
    let width = if p.width.is_finite() { p.width.clamp(0.3, 6.0) } else { 2.0 };
    let distance = if p.distance.is_finite() { p.distance.clamp(0.4, 8.0) } else { 2.5 };
    let gap = if p.gap.is_finite() { p.gap.clamp(0.0, 1.0) } else { 0.05 };

    // names -> (id, slot, visible, was active before this call)
    let mut listed: Vec<(OverlayID, WlxDeskSlot, bool, bool)> = Vec::new();
    for s in &p.screens {
        let found = overlays
            .iter()
            .find(|(_, o)| is_screenish(o.config.category) && o.config.name.as_ref() == s.name)
            .map(|(id, o)| (id, o.config.is_active()));
        match found {
            Some((id, active)) => listed.push((id, s.slot, s.visible, active)),
            None => log::warn!("desk screens: no screen/panel named '{}', skipped", s.name),
        }
    }

    let set_pose = |app: &mut AppState, id: OverlayID, t: Affine3A| {
        app.tasks.enqueue(TaskType::Overlay(OverlayTask::Modify(
            OverlaySelector::Id(id),
            Box::new(move |app, cfg| {
                if let Some(state) = cfg.active_state.as_mut() {
                    state.positioning = wlx_common::windowing::Positioning::Static;
                    state.curvature = (curvature > 0.0).then_some(curvature);
                    state.transform = t;
                    crate::windowing::window::save_transform(state, app);
                    cfg.dirty = true;
                }
            }),
        )));
    };
    let toggle = |app: &mut AppState, id: OverlayID, mode: ToggleMode| {
        app.tasks.enqueue(TaskType::Overlay(OverlayTask::ToggleOverlay(OverlaySelector::Id(id), mode)));
    };

    // main first: its (resulting) pose and displayed size drive the neighbours
    let mut main_pose: Option<(Affine3A, (f32, f32))> = None;
    let mut shown = 0;
    let mut hidden = 0;
    if let Some(&(id, _, visible, was_active)) = listed.iter().find(|(_, s, ..)| *s == WlxDeskSlot::Main) {
        if visible {
            let place_it = p.place || !was_active;
            let current = overlays
                .mut_by_id(id)
                .and_then(|o| o.config.active_state.as_ref().map(|s| s.transform));
            let t = if place_it {
                Some(grabmath::desk_arc(&hmd, 0, 1, width, distance))
            } else {
                current
            };
            toggle(app, id, ToggleMode::EnsureOn);
            if let Some(t) = t {
                if place_it {
                    set_pose(app, id, t);
                }
                let size = overlays
                    .mut_by_id(id)
                    .and_then(|o| o.config.backend.frame_meta())
                    .and_then(|m| quad(&(t * m.transform), m.extent))
                    .unwrap_or_else(|| {
                        let w = t.matrix3.col(0).length();
                        (w, w * 9.0 / 16.0)
                    });
                main_pose = Some((t, size));
            }
            shown += 1;
        } else {
            toggle(app, id, ToggleMode::EnsureOff);
            hidden += 1;
        }
    }

    for &(id, slot, visible, was_active) in &listed {
        let nslot = match slot {
            WlxDeskSlot::Main => continue,
            WlxDeskSlot::Left => grabmath::Slot::Left,
            WlxDeskSlot::Right => grabmath::Slot::Right,
            WlxDeskSlot::Up => grabmath::Slot::Up,
            WlxDeskSlot::UpLeft => grabmath::Slot::UpLeft,
            WlxDeskSlot::UpRight => grabmath::Slot::UpRight,
        };
        if !visible {
            toggle(app, id, ToggleMode::EnsureOff);
            hidden += 1;
            continue;
        }
        toggle(app, id, ToggleMode::EnsureOn);
        shown += 1;
        if p.place || !was_active {
            match main_pose {
                Some((m, size)) => set_pose(app, id, grabmath::desk_neighbour(&m, size, pivot, nslot, gap)),
                None => log::warn!("desk screens: no visible main screen, {slot:?} not placed"),
            }
        }
    }

    if p.hide_others {
        // only other desktop captures (a mirrored output, a stale screen): never a WayVR app window or panel, which
        // would make an app vanish from VR on a mode switch (pimax desk preset, user report 2026-10-15)
        let others: Vec<OverlayID> = overlays
            .iter()
            .filter(|(id, o)| hide_other_ok(o.config.category) && !listed.iter().any(|(l, ..)| l == id))
            .map(|(id, _)| id)
            .collect();
        for id in others {
            toggle(app, id, ToggleMode::EnsureOff);
            hidden += 1;
        }
    }
    log::info!(
        "desk screens: {shown} shown, {hidden} hidden of {} listed (place {}, hide_others {}, width {width} m, distance {distance} m, curvature {curvature}, gap {gap} m)",
        p.screens.len(),
        p.place,
        p.hide_others
    );
}

/// `desk-screens --hide-others` hides only desktop captures that are not listed. WayVR app windows and panels and
/// mirrors keep their state: a mode switch only changes which screens show, it never hides an app.
fn hide_other_ok(c: crate::windowing::window::OverlayCategory) -> bool {
    matches!(c, crate::windowing::window::OverlayCategory::Screen)
}

#[cfg(test)]
mod desk_screens_tests {
    use super::hide_other_ok;
    use crate::windowing::window::OverlayCategory;

    #[test]
    fn hide_others_never_hides_app_windows() {
        assert!(hide_other_ok(OverlayCategory::Screen));
        for c in [OverlayCategory::WayVR, OverlayCategory::Mirror, OverlayCategory::Panel, OverlayCategory::Keyboard, OverlayCategory::Dashboard] {
            assert!(!hide_other_ok(c), "{c:?}");
        }
    }
}

/// `WlxScreenNudge`: move/scale a screen itself in space (left/right/up/down around the head, closer/farther, bigger/smaller).
fn screen_nudge<O>(
    app: &mut AppState,
    overlays: &mut OverlayWindowManager<O>,
    p: wayvr_ipc::packet_client::WlxScreenNudgeParams,
) where
    O: Default,
{
    use crate::windowing::{OverlaySelector, grabmath, window::OverlayCategory};
    use glam::{Affine3A, Vec3A};
    use wayvr_ipc::packet_client::WlxNudgeDir;

    fn quad(t: &Affine3A, extent: [u32; 2]) -> Option<(f32, f32)> {
        if extent[0] == 0 || extent[1] == 0 {
            return None;
        }
        let aspect = extent[0] as f32 / extent[1] as f32;
        Some(if aspect < 1.0 {
            let major = t.matrix3.col(0).length();
            (major * aspect, major)
        } else {
            let major = t.matrix3.col(1).length();
            (major, major / aspect)
        })
    }

    let found = overlays
        .iter()
        .find(|(_, o)| {
            matches!(o.config.category, OverlayCategory::Screen | OverlayCategory::WayVR | OverlayCategory::Mirror)
                && o.config.name.as_ref() == p.name
        })
        .map(|(id, _)| id);
    let Some(id) = found else {
        log::warn!("screen nudge: no screen/panel named '{}'", p.name);
        return;
    };
    let Some(o) = overlays.mut_by_id(id) else {
        return;
    };
    let Some(t) = o.config.active_state.as_ref().map(|s| s.transform) else {
        log::info!("screen nudge: '{}' is not shown, nothing to move", p.name);
        return;
    };
    let size = o
        .config
        .backend
        .frame_meta()
        .and_then(|m| quad(&(t * m.transform), m.extent))
        .unwrap_or_else(|| {
            let w = t.matrix3.col(0).length();
            (w, w * 9.0 / 16.0)
        });
    let dir = match p.dir {
        WlxNudgeDir::Left => grabmath::NudgeDir::Left,
        WlxNudgeDir::Right => grabmath::NudgeDir::Right,
        WlxNudgeDir::Up => grabmath::NudgeDir::Up,
        WlxNudgeDir::Down => grabmath::NudgeDir::Down,
        WlxNudgeDir::Closer => grabmath::NudgeDir::Closer,
        WlxNudgeDir::Farther => grabmath::NudgeDir::Farther,
        WlxNudgeDir::Bigger => grabmath::NudgeDir::Bigger,
        WlxNudgeDir::Smaller => grabmath::NudgeDir::Smaller,
    };
    let pivot = Vec3A::from(app.input_state.hmd.translation);
    let nt = grabmath::nudge(&t, size, pivot, dir, p.step);
    log::info!("screen nudge: '{}' {:?} step {}", p.name, p.dir, p.step);
    crate::overlays::screen_numbers::note_screen_moved(&p.name);
    app.tasks.enqueue(TaskType::Overlay(OverlayTask::Modify(
        OverlaySelector::Id(id),
        Box::new(move |app, cfg| {
            if let Some(state) = cfg.active_state.as_mut() {
                state.positioning = wlx_common::windowing::Positioning::Static;
                state.transform = nt;
                crate::windowing::window::save_transform(state, app);
                cfg.dirty = true;
            }
        }),
    )));
}

/// Name of the custom panel (`custom_panels: [captions]`, `gui/captions.xml`) kept under the main screen.
pub const CAPTIONS_NAME: &str = "captions";

/// Desk-mode housekeeping every tick: keep the keyboard hidden when `enable_keyboard: false`, and put the
/// "captions" panel just below the main screen (same orientation, 90% of its width) so it follows when it is moved.
/// The main screen is the first active screen, or the overlay named in `PIMAX_CAPTIONS_SCREEN`.
fn desk_housekeeping<O>(app: &mut AppState, overlays: &mut OverlayWindowManager<O>)
where
    O: Default,
{
    use crate::windowing::window::OverlayCategory;
    use glam::{Affine3A, Mat3A, Vec3A};

    if !app.session.config.enable_keyboard {
        for o in overlays.values_mut() {
            if o.config.category == OverlayCategory::Keyboard && o.config.is_active() {
                o.config.deactivate();
            }
        }
    }

    let Some(cap_id) = overlays.lookup(CAPTIONS_NAME) else {
        return;
    };
    if !overlays.mut_by_id(cap_id).is_some_and(|c| c.config.is_active()) {
        return;
    }
    let want = std::env::var("PIMAX_CAPTIONS_SCREEN").ok();
    let screen_id = overlays
        .iter()
        .find(|(_, o)| {
            o.config.is_active()
                && matches!(o.config.category, OverlayCategory::Screen | OverlayCategory::WayVR)
                && want.as_deref().is_none_or(|w| o.config.name.as_ref() == w)
        })
        .map(|(id, _)| id);
    let Some(screen_id) = screen_id else {
        return;
    };

    // displayed (width, height) in metres of a quad, as the OpenXR backend computes it
    fn quad(t: &Affine3A, extent: [u32; 2]) -> Option<(f32, f32)> {
        if extent[0] == 0 || extent[1] == 0 {
            return None;
        }
        let aspect = extent[0] as f32 / extent[1] as f32;
        Some(if aspect < 1.0 {
            let major = t.matrix3.col(0).length();
            (major * aspect, major)
        } else {
            let major = t.matrix3.col(1).length();
            (major, major / aspect)
        })
    }

    let Some((st, s_w, s_h)) = overlays.mut_by_id(screen_id).and_then(|s| {
        let st = s.config.active_state.as_ref()?.transform;
        let meta = s.config.backend.frame_meta()?;
        let (w, h) = quad(&(st * meta.transform), meta.extent)?;
        Some((st, w, h))
    }) else {
        return;
    };

    let Some(cap) = overlays.mut_by_id(cap_id) else {
        return;
    };
    let Some(meta) = cap.config.backend.frame_meta() else {
        return;
    };
    let aspect = meta.extent[0] as f32 / meta.extent[1].max(1) as f32;
    let k = meta.transform.matrix3.col(1).length();
    if aspect < 1.0 || k <= 0.0 || s_w <= 0.0 {
        return;
    }
    let c_w = s_w * 0.9;
    let c_h = c_w / aspect;
    let scale = c_w / k;
    let right = st.matrix3.x_axis.normalize_or_zero();
    let up = st.matrix3.y_axis.normalize_or_zero();
    let fwd = st.matrix3.z_axis.normalize_or_zero();
    let centre = st.translation - up * (s_h * 0.5 + 0.03 + c_h * 0.5);
    if let Some(state) = cap.config.active_state.as_mut() {
        state.transform = Affine3A {
            matrix3: Mat3A::from_cols(right * scale, up * scale, fwd * scale),
            translation: Vec3A::from(centre),
        };
        cap.config.dirty = true;
    }
}

pub fn tick_events<O>(
    app: &mut AppState,
    _overlays: &mut OverlayWindowManager<O>,
) -> anyhow::Result<()>
where
    O: Default,
{
    desk_housekeeping(app, _overlays);
    crate::overlays::screen_numbers::tick(app, _overlays);

    while let Some(signal) = app.wayvr_signals.read() {
        match signal {
            WayVRSignal::WindowVisibilityChanged(window_handle, visible) => {
                if let Some(server) = &mut app.wvr_server {
                    server
                        .tasks
                        .send(WayVRTask::VisibilityChange(window_handle, visible));
                }
            }
            WayVRSignal::DeviceHaptics(device, haptics) => {
                app.tasks
                    .enqueue(TaskType::Input(InputTask::Haptics { device, haptics }));
            }
            WayVRSignal::ShowHide => {
                app.tasks.enqueue(TaskType::Overlay(OverlayTask::ShowHide));
            }
            WayVRSignal::ResetScreens => {
                app.tasks
                    .enqueue(TaskType::Overlay(OverlayTask::ResetScreens));
            }
            WayVRSignal::Recenter => {
                app.tasks.enqueue(TaskType::Playspace(
                    crate::backend::task::PlayspaceTask::Recenter,
                ));
            }
            WayVRSignal::DeskLayout(p) => desk_layout(app, _overlays, p),
            WayVRSignal::DeskScreens(p) => desk_screens(app, _overlays, p),
            WayVRSignal::ScreenNumbers(p) => crate::overlays::screen_numbers::handle(app, p),
            WayVRSignal::ScreenNudge(p) => screen_nudge(app, _overlays, p),
            WayVRSignal::SetSkybox(s) => {
                log::info!("backdrop: {s:?}");
                SKYBOX_OVER_MAIN.store(s.is_some(), std::sync::atomic::Ordering::Relaxed);
                app.session.config.use_skybox = s.is_some();
                if let Some(tex) = s {
                    app.session.config.skybox_texture = tex.into();
                }
                #[cfg(feature = "openxr")]
                app.tasks.enqueue(TaskType::OpenXR(
                    crate::backend::task::OpenXrTask::EnvironmentChanged,
                ));
            }
            WayVRSignal::GazeLaser(on) => {
                let cfg = &mut app.session.config;
                cfg.gaze_laser = on.unwrap_or(!cfg.gaze_laser);
                log::info!("gaze laser: {}", if cfg.gaze_laser { "on" } else { "off" });
            }
            WayVRSignal::PreferHands(on) => {
                let cfg = &mut app.session.config;
                cfg.controllers_in_work = on.map_or(!cfg.controllers_in_work, |hands_only| !hands_only);
                log::info!(
                    "controllers in Work: {}",
                    if cfg.controllers_in_work {
                        "on (a tracked controller wins, hands fill in)"
                    } else {
                        "off (hands only, controllers ignored entirely)"
                    }
                );
            }
            WayVRSignal::SwitchSet(set) => {
                app.tasks
                    .enqueue(TaskType::Overlay(OverlayTask::SwitchSet(set)));
            }
            WayVRSignal::CustomTask(custom_task) => {
                app.tasks
                    .enqueue(TaskType::Overlay(OverlayTask::ModifyPanel(custom_task)));
            }
            WayVRSignal::Handsfree(params) => match params {
                HandsfreeParams::SetMode(mode) => {
                    let mode = match mode {
                        HandsfreeMode::None => HandsfreePointer::None,
                        HandsfreeMode::Hmd => HandsfreePointer::HmdOnly,
                        HandsfreeMode::HmdPinch => HandsfreePointer::Hmd,
                        HandsfreeMode::EyeTracking => HandsfreePointer::EyeTrackingOnly,
                        HandsfreeMode::EyeTrackingPinch => HandsfreePointer::EyeTracking,
                    };
                    app.session.config.handsfree_pointer = mode;
                }
                _ => app.input_state.apply_handsfree_action(params),
            },
        }
    }

    let tick_tasks = WvrServerState::tick_events(app)?;
    if let Some(wayvr_server) = app.wvr_server.as_mut() {
        process_tick_tasks(tick_tasks, wayvr_server);
    }

    Ok(())
}
