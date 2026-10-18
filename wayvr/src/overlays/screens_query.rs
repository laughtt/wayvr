//! Read-only `wayvrctl screens`: the desk screens' poses and the head pose, with distance / bearing / elevation from
//! the head. The IPC handler has no access to the overlays, so `tick` (every frame, next to the screen number badges)
//! keeps a small snapshot that the handler copies. Stage (world) space, metres and degrees.
//!
//! Conventions: the head looks along its -Z axis. Yaw 0 = looking along -Z, positive = turning right (towards +X);
//! pitch positive = up. Bearing: horizontal angle from the head's forward to the screen centre, + = right, -180..180.
//! Elevation: vertical angle of the screen centre above the head's own pitch (a screen straight ahead of the gaze = 0).

use std::sync::{LazyLock, Mutex};

use glam::{Affine3A, Vec3A};
use wayvr_ipc::packet_server::{WlxScreens, WlxScreensEntry, WlxScreensHead};

use crate::{
    state::AppState,
    windowing::{manager::OverlayWindowManager, window::OverlayCategory},
};

static SNAP: LazyLock<Mutex<WlxScreens>> = LazyLock::new(|| {
    Mutex::new(WlxScreens {
        head_valid: false,
        head: WlxScreensHead { pos: [0.0; 3], yaw_deg: 0.0, pitch_deg: 0.0 },
        screens: Vec::new(),
    })
});

/// Yaw (+ = right) and pitch (+ = up) in degrees of a direction.
pub fn yaw_pitch_deg(dir: Vec3A) -> (f32, f32) {
    let h = (dir.x * dir.x + dir.z * dir.z).sqrt();
    (dir.x.atan2(-dir.z).to_degrees(), dir.y.atan2(h).to_degrees())
}

/// Distance (m), bearing (deg, + = right) and elevation (deg, + = above the head's pitch) of `target` seen from `head`.
pub fn relative(head: &Affine3A, target: Vec3A) -> (f32, f32, f32) {
    let d = target - Vec3A::from(head.translation);
    let dist = d.length();
    let fwd = -Vec3A::from(head.matrix3.z_axis);
    let (_, head_pitch) = yaw_pitch_deg(fwd);
    let fh = Vec3A::new(fwd.x, 0.0, fwd.z);
    let fh = if fh.length() < 1e-3 { Vec3A::NEG_Z } else { fh.normalize() };
    let right = Vec3A::new(-fh.z, 0.0, fh.x);
    let bearing = d.dot(right).atan2(d.dot(fh)).to_degrees();
    let elevation = if dist < 1e-6 { 0.0 } else { yaw_pitch_deg(d).1 - head_pitch };
    (dist, bearing, elevation)
}

/// Displayed width in metres of a quad (as the OpenXR backend computes it).
fn width_of(t: &Affine3A, extent: [u32; 2]) -> f32 {
    if extent[0] == 0 || extent[1] == 0 {
        return t.matrix3.col(0).length();
    }
    let aspect = extent[0] as f32 / extent[1] as f32;
    if aspect < 1.0 { t.matrix3.col(0).length() * aspect } else { t.matrix3.col(1).length() }
}

/// Every frame: refresh the snapshot (at most 6 screens; the vector is reused).
pub fn tick<O: Default>(app: &mut AppState, overlays: &mut OverlayWindowManager<O>) {
    let hmd = app.input_state.hmd;
    let shown = crate::overlays::screen_numbers::shown_screens();
    let mut snap = SNAP.lock().unwrap();
    snap.head_valid = hmd != Affine3A::IDENTITY;
    let (yaw, pitch) = yaw_pitch_deg(-Vec3A::from(hmd.matrix3.z_axis));
    snap.head = WlxScreensHead { pos: hmd.translation.into(), yaw_deg: yaw, pitch_deg: pitch };
    snap.screens.clear();
    for (name, n) in shown {
        let Some(id) = overlays.iter().find_map(|(id, o)| {
            (matches!(o.config.category, OverlayCategory::Screen | OverlayCategory::WayVR | OverlayCategory::Mirror)
                && o.config.name.as_ref() == name.as_ref())
            .then_some(id)
        }) else {
            continue;
        };
        let Some(o) = overlays.mut_by_id(id) else { continue };
        let on = o.config.is_active();
        let Some(st) = o.config.active_state.as_ref() else { continue };
        let t = st.transform;
        let curvature = st.curvature.unwrap_or(0.0);
        let width_m = match o.config.backend.frame_meta() {
            Some(m) => width_of(&(t * m.transform), m.extent),
            None => t.matrix3.col(0).length(),
        };
        let pos = Vec3A::from(t.translation);
        let (distance_m, bearing_deg, elevation_deg) = relative(&hmd, pos);
        let (yaw_deg, pitch_deg) = yaw_pitch_deg(-Vec3A::from(t.matrix3.z_axis).normalize_or_zero());
        snap.screens.push(WlxScreensEntry {
            n,
            output: name.to_string(),
            shown: on,
            pos: pos.into(),
            yaw_deg,
            pitch_deg,
            width_m,
            curvature,
            distance_m,
            bearing_deg,
            elevation_deg,
        });
    }
}

/// The last snapshot, for the IPC answer.
pub fn report() -> WlxScreens {
    SNAP.lock().unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    fn near(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    #[test]
    fn straight_ahead_is_zero() {
        let (d, b, e) = relative(&Affine3A::IDENTITY, Vec3A::new(0.0, 0.0, -2.5));
        assert!(near(d, 2.5) && near(b, 0.0) && near(e, 0.0), "{d} {b} {e}");
    }

    #[test]
    fn right_is_positive_left_negative() {
        let (_, r, _) = relative(&Affine3A::IDENTITY, Vec3A::new(2.0, 0.0, -2.0));
        let (_, l, _) = relative(&Affine3A::IDENTITY, Vec3A::new(-2.0, 0.0, -2.0));
        assert!(near(r, 45.0) && near(l, -45.0), "{r} {l}");
    }

    #[test]
    fn behind_is_180() {
        let (_, b, _) = relative(&Affine3A::IDENTITY, Vec3A::new(0.0, 0.0, 3.0));
        assert!(near(b.abs(), 180.0), "{b}");
    }

    #[test]
    fn above_is_positive_elevation() {
        let (d, b, e) = relative(&Affine3A::IDENTITY, Vec3A::new(0.0, 2.0, -2.0));
        assert!(near(b, 0.0) && near(e, 45.0) && near(d, 8f32.sqrt()), "{d} {b} {e}");
        let (_, _, below) = relative(&Affine3A::IDENTITY, Vec3A::new(0.0, -2.0, -2.0));
        assert!(near(below, -45.0));
    }

    #[test]
    fn follows_the_head_pose() {
        // head turned 90 deg to the left (facing -X) at (1,0,0): a screen at (-2,0,0) is straight ahead
        let head = Affine3A::from_rotation_translation(Quat::from_rotation_y(90f32.to_radians()), glam::Vec3::new(1.0, 0.0, 0.0));
        let (d, b, e) = relative(&head, Vec3A::new(-2.0, 0.0, 0.0));
        assert!(near(d, 3.0) && near(b, 0.0) && near(e, 0.0), "{d} {b} {e}");
        // and the old straight-ahead screen (1,0,-2) is now to the right
        let (_, b2, _) = relative(&head, Vec3A::new(1.0, 0.0, -2.0));
        assert!(near(b2, 90.0), "{b2}");
        let (yaw, pitch) = yaw_pitch_deg(-Vec3A::from(head.matrix3.z_axis));
        assert!(near(yaw, -90.0) && near(pitch, 0.0), "{yaw} {pitch}");
    }

    #[test]
    fn elevation_is_relative_to_head_pitch() {
        // head pitched up 30 deg: a screen 30 deg above the horizon is straight ahead of the gaze
        let head = Affine3A::from_rotation_translation(Quat::from_rotation_x(30f32.to_radians()), glam::Vec3::ZERO);
        let t = 30f32.to_radians();
        let (_, _, e) = relative(&head, Vec3A::new(0.0, t.sin(), -t.cos()) * 2.0);
        assert!(near(e, 0.0), "{e}");
    }
}
