use glam::{Affine3A, EulerRot, Mat3A, Quat, Vec3, Vec3A};
use idmap_derive::IntegerId;
use std::{f32::consts::PI, sync::Arc};
use wlx_common::windowing::{OverlayWindowState, Positioning};

use crate::{
    state::AppState,
    subsystem::input::InputFocus,
    windowing::{
        backend::{FrameMeta, OverlayBackend, RenderResources, ShouldRender},
        snap_upright,
    },
};

pub struct OverlayWindowData<T> {
    pub config: OverlayWindowConfig,
    pub data: T,
    pub birthframe: usize,
    pub primary_pointer: Option<usize>,
    pub hover_pointers: [bool; 2],
}

impl<T> OverlayWindowData<T>
where
    T: Default,
{
    pub fn from_config(config: OverlayWindowConfig) -> Self {
        Self {
            data: T::default(),
            config,
            primary_pointer: None,
            hover_pointers: [false, false],
            birthframe: 0,
        }
    }
}

impl<T> OverlayWindowData<T> {
    pub fn init(&mut self, app: &mut AppState) -> anyhow::Result<()> {
        //TODO: load state?

        self.config.backend.init(app)
    }
    pub fn should_render(&mut self, app: &mut AppState) -> anyhow::Result<ShouldRender> {
        self.config.backend.should_render(app)
    }
    pub fn render(&mut self, app: &mut AppState, rdr: &mut RenderResources) -> anyhow::Result<()> {
        self.config.backend.render(app, rdr)
    }
    pub fn frame_meta(&mut self) -> Option<FrameMeta> {
        self.config.backend.frame_meta()
    }
}

#[derive(Debug, Clone, Copy, IntegerId, PartialEq, Eq)]
pub enum OverlayCategory {
    Internal,
    Keyboard,
    Dashboard,
    /// User-provided custom panels
    Panel,
    /// Same as Panel but does not support Modify and Reload
    BuiltInPanel,
    Screen,
    Mirror,
    WayVR,
    Passthru,
}

pub struct OverlayWindowConfig {
    pub name: Arc<str>,
    pub backend: Box<dyn OverlayBackend>,
    /// The default state to show when the overlay is newly spawned.
    pub default_state: OverlayWindowState,
    /// The current state to show. None if the overlay is hidden.
    pub active_state: Option<OverlayWindowState>,
    /// Order to draw overlays in. Overlays with higher numbers will be drawn over ones with lower numbers.
    pub z_order: u32,
    /// If set, hovering this overlay will cause the HID provider to switch focus.
    pub input_focus: Option<InputFocus>,
    /// Category of the overlay, used by toolbox on the watch.
    pub category: OverlayCategory,
    /// Should the overlay be displayed on the next frame?
    pub show_on_spawn: bool,
    /// Does not belong to any set; switching sets does not affect this overlay.
    pub global: bool,
    /// True if transform, curvature, alpha has changed. Only used by OpenVR.
    pub dirty: bool,
    /// True if the window is showing the edit overlay
    pub editing: bool,
    /// True if the window is being resized. Captures pointer hits even if not directly hit.
    pub resizing: bool,
    /// Used by grab to pause following of HMD or other devices
    pub pause_movement: bool,
    /// seconds a world-locked screen has been out of view (desk mode auto-recovery)
    pub out_of_view_s: f32,
    /// seconds since this window was created; a screen left out of view at start-up is brought back sooner
    pub age_s: f32,
}

impl OverlayWindowConfig {
    pub fn from_backend(backend: Box<dyn OverlayBackend>) -> Self {
        Self {
            name: "".into(),
            backend,
            default_state: OverlayWindowState {
                transform: Affine3A::from_translation(Vec3::NEG_Z),
                ..OverlayWindowState::default()
            },
            active_state: None,
            z_order: 0,
            input_focus: None,
            category: OverlayCategory::Internal,
            show_on_spawn: false,
            global: false,
            dirty: true,
            editing: false,
            resizing: false,
            pause_movement: false,
            out_of_view_s: 0.0,
            age_s: 0.0,
        }
    }

    pub const fn is_active(&self) -> bool {
        self.active_state.is_some()
    }

    /// only call this directly for `OverlayCategory::Internal`
    /// for anything else, use `OverlayTask::ToggleOverlay` instead
    pub fn activate(&mut self, app: &mut AppState, realign: bool) {
        log::debug!("activate {}", self.name.as_ref());
        self.dirty = true;
        self.active_state = Some(self.default_state.clone());
        self.reset(app, realign);
    }

    /// only call this directly for `OverlayCategory::Internal`
    /// for anything else, use `OverlayTask::ToggleOverlay` instead
    pub fn deactivate(&mut self) {
        log::debug!("deactivate {}", self.name.as_ref());
        self.active_state = None;
    }

    pub fn tick(&mut self, app: &mut AppState) {
        if app.session.config.world_lock_grab {
            self.auto_recover(app);
        }
        self.auto_movement(app);
        self.angle_fade(app);
    }

    /// Desk mode: a world-locked screen that is unusable (bad transform, no pose, too close/far, scale out of limits,
    /// or out of view for 10 s with no grab) goes back to the default position in front of the head.
    fn auto_recover(&mut self, app: &mut AppState) {
        use crate::windowing::grabmath;
        if self.pause_movement
            || self.editing
            || !matches!(
                self.category,
                OverlayCategory::Screen | OverlayCategory::Mirror | OverlayCategory::WayVR
            )
        {
            return;
        }
        let Some(state) = self.active_state.as_mut() else {
            return;
        };
        if !matches!(state.positioning, Positioning::Static) {
            return;
        }
        let cfg = &app.session.config;
        let lim = grabmath::Limits {
            size_min: cfg.grab_size_min,
            size_max: cfg.grab_size_max,
            dist_min: cfg.grab_dist_min,
            dist_max: cfg.grab_dist_max,
        };
        let hmd = app.input_state.hmd;
        let mut reason = if state.saved_transform.is_none() {
            Some("no saved pose")
        } else {
            grabmath::invalid_reason(&state.transform, &hmd, &lim)
        };
        self.age_s += app.delta_time;
        if reason.is_none() {
            // start-up: a pose saved in another session's world is often behind the user; don't make them wait.
            // After that a screen the user can place is never moved because they looked away (opt-in only).
            if let Some(limit) = out_of_view_limit(self.age_s, cfg.world_lock_out_of_view_reset) {
                if grabmath::view_angle_deg(&state.transform, &hmd) > 80.0 {
                    self.out_of_view_s += app.delta_time;
                    if self.out_of_view_s > limit {
                        reason = Some(if limit < 10.0 {
                            "out of view for more than 2 s at start-up"
                        } else {
                            "out of view for more than 10 s (world_lock_out_of_view_reset is on)"
                        });
                    }
                } else {
                    self.out_of_view_s = 0.0;
                }
            } else {
                self.out_of_view_s = 0.0;
            }
        }
        if let Some(reason) = reason {
            log::warn!("auto-reset {}: {reason}", self.name);
            state.transform = grabmath::default_front(&hmd, 0);
            save_transform(state, app);
            self.out_of_view_s = 0.0;
            self.dirty = true;
        }
    }

    fn auto_movement(&mut self, app: &mut AppState) {
        if self.pause_movement {
            return;
        }

        let Some(state) = self.active_state.as_mut() else {
            return;
        };

        let cur_transform = state
            .saved_transform
            .unwrap_or(self.default_state.transform);
        let scale = scalar_scale(&cur_transform);

        let (target_transform, lerp) = match state.positioning {
            Positioning::FollowHead { lerp } => (app.input_state.hmd * cur_transform, lerp),
            Positioning::FollowHand { hand, lerp } => (
                app.input_state.pointers[hand as usize].pose * cur_transform,
                lerp,
            ),
            Positioning::Anchored => (app.anchor * cur_transform, 1.0),
            Positioning::Static | Positioning::Floating => (state.transform, 1.0), // STAGE space
        };

        state.transform = match lerp {
            1.0 => target_transform,
            lerp => {
                let rot_from = Quat::from_mat3a(&state.transform.matrix3.div_scalar(scale));
                let rot_to = Quat::from_mat3a(&target_transform.matrix3.div_scalar(scale));

                let rotation = rot_from.slerp(rot_to, lerp);
                let translation = state
                    .transform
                    .translation
                    .slerp(target_transform.translation, lerp);

                Affine3A::from_scale_rotation_translation(
                    Vec3::ONE * scale,
                    rotation,
                    translation.into(),
                )
            }
        };

        if state.align_to_hmd {
            realign(
                &mut state.transform,
                &app.input_state.hmd,
                scale,
                app.session.config.snap_angle_deg,
            );
        }

        self.dirty = true;
    }

    fn angle_fade(&mut self, app: &mut AppState) {
        let Some(state) = self.active_state.as_mut() else {
            return;
        };

        if !state.angle_fade {
            return;
        }

        let to_hmd = (state.transform.translation - app.input_state.hmd.translation).normalize();
        let watch_normal = state.transform.transform_vector3a(Vec3A::NEG_Z).normalize();
        let dot = to_hmd.dot(watch_normal);

        state.alpha = (dot - app.session.config.watch_view_angle_min)
            / (app.session.config.watch_view_angle_max - app.session.config.watch_view_angle_min);
        state.alpha += 0.1;
        state.alpha = state.alpha.clamp(0., 1.);
    }

    /// Returns true if changes were saved.
    pub fn reset(&mut self, app: &mut AppState, hard_reset: bool) {
        let Some(state) = self.active_state.as_mut() else {
            return;
        };

        let cur_transform = state
            .saved_transform
            .unwrap_or(self.default_state.transform);

        let parent_transform = match state.positioning {
            Positioning::Floating | Positioning::FollowHead { .. } => app.input_state.hmd,
            Positioning::FollowHand { hand, .. } => app.input_state.pointers[hand as usize].pose,
            Positioning::Anchored => app.anchor,
            Positioning::Static => {
                if hard_reset {
                    app.input_state.hmd
                } else {
                    Affine3A::IDENTITY
                }
            }
        };

        if hard_reset {
            state.saved_transform = None;
        }

        state.transform = parent_transform * cur_transform;

        if state.align_to_hmd || (state.grabbable && hard_reset) {
            let scale = scalar_scale(&cur_transform);
            realign(
                &mut state.transform,
                &app.input_state.hmd,
                scale,
                app.session.config.snap_angle_deg,
            );
        }
        self.dirty = true;
    }
}

pub fn realign(transform: &mut Affine3A, hmd: &Affine3A, scale: f32, snap_angle: f32) {
    let to_hmd = hmd.translation - transform.translation;
    let up_dir: Vec3A;

    if hmd.x_axis.dot(Vec3A::Y).abs() > 0.2 {
        // Snap upright
        up_dir = hmd.y_axis;
    } else {
        let dot = to_hmd.normalize().dot(hmd.z_axis);
        let z_dist = to_hmd.length();
        let y_dist = (transform.translation.y - hmd.translation.y).abs();
        let x_angle = (y_dist / z_dist).asin();

        if dot < -f32::EPSILON {
            // facing down
            let up_point = hmd.translation + z_dist / x_angle.cos() * Vec3A::Y;
            up_dir = (up_point - transform.translation).normalize();
        } else if dot > f32::EPSILON {
            // facing up
            let dn_point = hmd.translation + z_dist / x_angle.cos() * Vec3A::NEG_Y;
            up_dir = (transform.translation - dn_point).normalize();
        } else {
            // perfectly upright
            up_dir = Vec3A::Y;
        }
    }

    let col_z = (transform.translation - hmd.translation).normalize();
    let col_y = up_dir;
    let col_x = col_y.cross(col_z);
    let col_y = col_z.cross(col_x).normalize();
    let col_x = col_x.normalize();

    let flip = Mat3A::from_quat(Quat::from_axis_angle(Vec3::Y, PI));
    let mut rotation = Mat3A::from_cols(col_x, col_y, col_z) * flip;

    if snap_angle > f32::EPSILON {
        let snap_step = snap_angle.to_radians();
        let (yaw, pitch, roll) = rotation.to_euler(EulerRot::YXZ);

        let pitch = (pitch / snap_step).round() * snap_step;
        let roll = (roll / snap_step).round() * snap_step;

        rotation = Mat3A::from_euler(EulerRot::YXZ, yaw, pitch, roll);
    }

    transform.matrix3 = rotation.mul_scalar(scale);
}

pub fn scalar_scale(a: &Affine3A) -> f32 {
    let det = a.matrix3.determinant();
    (a.matrix3.x_axis.length() * det.signum()
        + a.matrix3.y_axis.length()
        + a.matrix3.z_axis.length())
        / 3.0
}

pub fn save_transform(state: &mut OverlayWindowState, app: &mut AppState) {
    let parent_transform = match state.positioning {
        Positioning::Floating => snap_upright(app.input_state.hmd, Vec3A::Y),
        Positioning::FollowHead { .. } => app.input_state.hmd,
        Positioning::FollowHand { hand, .. } => app.input_state.pointers[hand as usize].pose,
        Positioning::Anchored => snap_upright(app.anchor, Vec3A::Y),
        Positioning::Static => Affine3A::IDENTITY,
    };

    state.saved_transform = Some(parent_transform.inverse() * state.transform);
}

pub fn spawn_transform_from_parent(
    parent: &Affine3A,
    hmd: &Affine3A,
    child_scale: f32,
) -> Affine3A {
    const LEFT: f32 = 0.08;
    const DOWN: f32 = 0.08;
    const CLOSER_TO_HMD: f32 = 0.06;

    let x_axis = parent.x_axis.normalize();
    let y_axis = parent.y_axis.normalize();
    let z_axis = parent.z_axis.normalize();
    let to_hmd = hmd.translation - parent.translation;
    let toward_hmd = if z_axis.dot(to_hmd) >= 0.0 {
        z_axis
    } else {
        -z_axis
    };

    let translation =
        parent.translation - x_axis * LEFT - y_axis * DOWN + toward_hmd * CLOSER_TO_HMD;

    let rotation = Quat::from_mat3a(&Mat3A::from_cols(x_axis, y_axis, z_axis));

    Affine3A::from_scale_rotation_translation(Vec3::ONE * child_scale, rotation, translation.into())
}

/// Seconds a world-locked screen may stay out of view before it is reset: 2 s in the first 30 s after start-up,
/// 10 s later only when `world_lock_out_of_view_reset` is on, otherwise never.
fn out_of_view_limit(age_s: f32, opt_in: bool) -> Option<f32> {
    if age_s < 30.0 {
        Some(2.0)
    } else if opt_in {
        Some(10.0)
    } else {
        None
    }
}

#[cfg(test)]
mod out_of_view_tests {
    use super::out_of_view_limit;

    #[test]
    fn startup_always_later_only_opt_in() {
        assert_eq!(out_of_view_limit(5.0, false), Some(2.0));
        assert_eq!(out_of_view_limit(100.0, false), None);
        assert_eq!(out_of_view_limit(100.0, true), Some(10.0));
    }
}
