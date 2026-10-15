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
