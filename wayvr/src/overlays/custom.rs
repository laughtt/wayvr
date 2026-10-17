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

/// Width of the Work menu panel in metres (the custom panel named by the `menu_panel` setting): the longer side of a
/// landscape panel, like `PERF_WIDTH_M`.
pub const MENU_WIDTH_M: f32 = 0.45;
/// Where the Work menu appears relative to the head when it is shown: 0.6 m ahead along the line of sight, 8 cm
/// below it (eye height or slightly below). The head's rotation is part of the parent pose: the panel faces the head.
const MENU_POS: Vec3 = vec3(0.0, -0.08, -0.6);

/// True when the custom panel `name` is the Work menu, i.e. the panel the `menu_panel` setting names.
pub fn is_menu_panel(menu_panel: Option<&str>, name: &str) -> bool {
    menu_panel.is_some_and(|m| !m.is_empty() && m == name)
}

/// The overlay the `menu` binding toggles: the one named by `menu_panel` if `lookup` finds it. None = no menu panel
/// is set or it does not exist, and the binding toggles the dashboard instead.
pub fn menu_overlay<I>(
    menu_panel: Option<&str>,
    lookup: impl FnOnce(&str) -> Option<I>,
) -> Option<I> {
    menu_panel.filter(|m| !m.is_empty()).and_then(lookup)
}

/// Default state of the Work menu: pointable, NOT grabbable (a menu must not be dragged by accident), `Floating`
/// so it is parented to the head. Every fresh activation (the `menu` binding, the watch's overlay buttons) therefore
/// puts it `MENU_POS` in front of the head, facing it, and never where it was last.
pub fn menu_state(width: f32) -> OverlayWindowState {
    OverlayWindowState {
        interactable: true,
        grabbable: false,
        positioning: Positioning::Floating,
        transform: Affine3A::from_scale_rotation_translation(
            Vec3::ONE * width,
            Quat::IDENTITY,
            MENU_POS,
        ),
        ..OverlayWindowState::default()
    }
}

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

    let mut default_state = OverlayWindowState {
        interactable: !fixed,
        grabbable: !fixed,
        positioning,
        angle_fade: perf,
        transform: Affine3A::from_scale_rotation_translation(Vec3::ONE * scale, rot, pos),
        ..OverlayWindowState::default()
    };

    // the Work menu (`menu_panel`): hidden at spawn, shown by the `menu` binding (Touch: left menu button) or the
    // watch's overlay buttons, 0.45 m wide, 0.6 m in front of the head, pointable, not grabbable. It is global: it
    // stays when sets are switched or hidden, and a menu left open at exit is not reopened at the next start (a
    // global overlay only comes back shown if it spawns shown).
    let menu = is_menu_panel(app.session.config.menu_panel.as_deref(), &name);
    if menu {
        default_state = menu_state(MENU_WIDTH_M);
    }

    Some(OverlayWindowConfig {
        name,
        category: OverlayCategory::Panel,
        show_on_spawn: fixed,
        global: menu,
        default_state,
        ..OverlayWindowConfig::from_backend(Box::new(panel))
    })
}

#[cfg(test)]
mod tests {
    use glam::Mat3A;

    use super::*;

    /// A head 1.6 m up, turned and tilted (looking down a little and to the side).
    fn tilted_head() -> Affine3A {
        Affine3A::from_rotation_translation(
            Quat::from_rotation_y(2.0) * Quat::from_rotation_x(-0.4),
            vec3(1.0, 1.6, -3.0),
        )
    }

    #[test]
    fn menu_is_pointable_and_not_grabbable() {
        let s = menu_state(MENU_WIDTH_M);
        assert!(
            s.interactable,
            "the controller ray and the hand pinch must reach its buttons"
        );
        assert!(!s.grabbable, "a menu must not be dragged by accident");
        assert!(!s.angle_fade && !s.align_to_hmd && s.saved_transform.is_none());
        // parented to the head: every fresh activation places it in front of the head again
        assert!(matches!(s.positioning, Positioning::Floating));
    }

    /// `OverlayWindowConfig::reset` puts a `Floating` overlay at `hmd * state.transform`.
    #[test]
    fn menu_appears_in_front_of_the_head_facing_it() {
        for hmd in [Affine3A::IDENTITY, tilted_head()] {
            let world = hmd * menu_state(MENU_WIDTH_M).transform;
            let to_menu = world.translation - hmd.translation;

            // about 0.6 m ahead along the line of sight
            assert!(
                (to_menu.length() - 0.6).abs() < 0.02,
                "{} m away",
                to_menu.length()
            );
            let gaze = -hmd.matrix3.z_axis.normalize();
            assert!(
                to_menu.normalize().dot(gaze) > 0.98,
                "not in front of the eyes"
            );

            // at eye height or slightly below, never above
            let up = hmd.matrix3.y_axis.normalize();
            let below = -to_menu.dot(up);
            assert!(
                below > 0.0 && below < 0.15,
                "{below} m below the line of sight"
            );

            // facing the head: the front of the quad (+Z) points back at the head
            let front = world.matrix3.z_axis.normalize();
            assert!(
                front.dot(-to_menu.normalize()) > 0.98,
                "not facing the head"
            );
            // upright relative to the head (no roll of its own)
            assert!(world.matrix3.y_axis.normalize().dot(up) > 0.999);
        }
    }

    #[test]
    fn menu_is_about_45_cm_wide_and_unskewed() {
        let m: Mat3A = menu_state(MENU_WIDTH_M).transform.matrix3;
        assert!((MENU_WIDTH_M - 0.45).abs() < 1e-6);
        for axis in [m.x_axis, m.y_axis, m.z_axis] {
            assert!((axis.length() - MENU_WIDTH_M).abs() < 1e-5);
        }
    }

    #[test]
    fn only_the_panel_named_by_menu_panel_is_the_menu() {
        assert!(is_menu_panel(Some("work_menu"), "work_menu"));
        assert!(!is_menu_panel(Some("work_menu"), "perf"));
        assert!(!is_menu_panel(None, "work_menu"));
        assert!(!is_menu_panel(Some(""), ""));
    }

    #[test]
    fn menu_binding_toggles_the_named_overlay_else_the_dashboard() {
        let lookup = |name: &str| match name {
            "work_menu" => Some(7),
            "kbd" => Some(1),
            _ => None,
        };
        assert_eq!(menu_overlay(Some("work_menu"), lookup), Some(7));
        assert_eq!(menu_overlay(Some("kbd"), lookup), Some(1));
        // unset, empty, or no overlay of that name: None = the dashboard
        assert_eq!(menu_overlay(None, lookup), None);
        assert_eq!(menu_overlay(Some(""), lookup), None);
        assert_eq!(menu_overlay(Some("missing"), lookup), None);
    }
}
