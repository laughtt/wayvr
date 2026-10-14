use std::{sync::Arc, time::Duration};

use glam::{Affine3A, Quat, Vec3, vec3};
use wgui::globals::expand_env_vars;
use wlx_common::{
    common::LeftRight,
    overlays::{BackendAttrib, BackendAttribValue},
    windowing::{OverlayWindowState, Positioning},
};

use crate::{
    gui::{
        panel::{GuiPanel, NewGuiPanelParams, apply_custom_command},
        timer::GuiTimer,
    },
    state::AppState,
    windowing::{
        backend::OverlayEventData,
        window::{OverlayCategory, OverlayWindowConfig},
    },
};

struct CustomPanelState;

/// Name of the custom panel (`custom_panels: [perf]`, `gui/perf.xml`) that follows the right hand like the watch.
pub const PERF_NAME: &str = "perf";
/// Width of the wrist panel in metres.
const PERF_WIDTH_M: f32 = 0.10;
/// The watch pose (watch.rs `WATCH_POS`/`WATCH_ROT`) mirrored for the right hand: x flipped, and the panel turned so the
/// text still reads left to right (along the forearm towards the elbow).
const PERF_POS: Vec3 = vec3(0.03, -0.01, 0.125);
const PERF_ROT: Quat = Quat::from_xyzw(0.707_106_8, 0.0, 0.707_106_8, 0.0);

pub fn create_custom(app: &mut AppState, name: Arc<str>) -> Option<OverlayWindowConfig> {
    let params = NewGuiPanelParams {
        external_xml: true,
        ..NewGuiPanelParams::default()
    };

    let mut panel =
        GuiPanel::new_from_template(app, &format!("gui/{name}.xml"), CustomPanelState, params)
            .inspect_err(|e| log::warn!("Error creating '{name}': {e:?}"))
            .ok()?;

    if let Some(icon) = panel.parser_state.data.var_map.get("_panel_icon") {
        let icon = expand_env_vars(icon);
        panel
            .extra_attribs
            .insert(BackendAttrib::Icon, BackendAttribValue::Icon(icon.into()));
    }

    panel
        .update_layout(app)
        .inspect_err(|e| log::warn!("Error layouting '{name}': {e:?}"))
        .ok()?;

    panel
        .timers
        .push(GuiTimer::new(Duration::from_millis(100), 0));

    let mut scale = panel.layout.content_size.x / 40.0 * 0.05;

    panel.on_notify = Some(Box::new({
        let name = name.clone();
        move |panel, app, event_data| {
            let OverlayEventData::CustomCommand { element, command } = event_data else {
                return Ok(());
            };

            if let Err(e) = apply_custom_command(panel, app, &element, &command) {
                log::warn!("Could not apply {command:?} on {name}/{element}: {e:?}");
            }

            Ok(())
        }
    }));

    // "captions": voice caption strip, placed under the main screen by ipc::events (not pointable, not grabbable)
    let captions = name.as_ref() == crate::ipc::events::CAPTIONS_NAME;

    // "perf": performance readout on the RIGHT wrist (mirror of the watch's pose), 10 cm wide, not pointable/grabbable
    let perf = name.as_ref() == PERF_NAME;
    let mut positioning = Positioning::default();
    let mut rot = Quat::IDENTITY;
    let mut pos = vec3(0.0, 0.0, -0.5);
    if perf {
        scale = PERF_WIDTH_M;
        positioning = Positioning::FollowHand {
            hand: LeftRight::Right,
            lerp: 1.0,
        };
        rot = PERF_ROT;
        pos = PERF_POS;
    }
    let fixed = captions || perf;

    Some(OverlayWindowConfig {
        name,
        category: OverlayCategory::Panel,
        show_on_spawn: fixed,
        default_state: OverlayWindowState {
            interactable: !fixed,
            grabbable: !fixed,
            positioning,
            angle_fade: perf,
            transform: Affine3A::from_scale_rotation_translation(Vec3::ONE * scale, rot, pos),
            ..OverlayWindowState::default()
        },
        ..OverlayWindowConfig::from_backend(Box::new(panel))
    })
}
