//! Camera hands as the pointer (`prefer_hands`, on by default).
//!
//! When the runtime tracks a bare hand with its cameras (XR_EXT_hand_tracking, unobstructed data source), that hand
//! is the pointer: its ray and its pinch replace the controllers. The controller rays and the controller buttons are
//! ignored until no camera hand has been seen for [`HOLD`]. `wayvrctl prefer-hands off` (voice: "prioritize
//! controllers") gives the controllers the pointer back; `on` ("prioritize hands") returns it to the hands.
//!
//! The math here has no OpenXR in it (tests run without a runtime); `backend/openxr/camera_hands.rs` feeds it.

use std::time::{Duration, Instant};

use glam::{Affine3A, Vec3, Vec3A};

/// Thumb tip to finger tip distance (m) below which a pinch starts.
pub const PINCH_ON_M: f32 = 0.022;
/// ... and above which it ends (hysteresis: camera joints jitter by a few mm).
pub const PINCH_OFF_M: f32 = 0.040;
/// The hands keep the pointer this long after the last tracked camera hand, so a few missed camera frames do not
/// flick the pointer back to a controller that lies on the desk.
pub const HOLD: Duration = Duration::from_millis(500);

/// Shoulder estimate from the head: this far down and this far to the side (m).
const SHOULDER_DOWN_M: f32 = 0.17;
const SHOULDER_SIDE_M: f32 = 0.16;

/// The joints the pointer needs (stage space).
#[derive(Debug, Clone, Copy)]
pub struct HandJoints {
    pub index_proximal: Vec3,
    pub thumb_tip: Vec3,
    pub index_tip: Vec3,
    pub middle_tip: Vec3,
}

/// One tracked hand as a pointer.
#[derive(Debug, Clone, Copy)]
pub struct HandSample {
    /// Ray pose: -Z along the ray, like an OpenXR aim pose.
    pub pose: Affine3A,
    /// Index pinch.
    pub click: bool,
    /// Middle-finger pinch.
    pub grab: bool,
}

/// The ray of a hand: from an estimated shoulder through the index knuckle (steadier than the wrist angle and does
/// not move when the fingers pinch), starting at the knuckle.
pub fn aim_pose(j: &HandJoints, left: bool, hmd: &Affine3A) -> Option<Affine3A> {
    let head = Vec3::from(hmd.translation);
    let mut right = Vec3::from(hmd.matrix3 * Vec3A::X);
    right.y = 0.0;
    let right = right.try_normalize().unwrap_or(Vec3::X);
    let side = if left {
        -SHOULDER_SIDE_M
    } else {
        SHOULDER_SIDE_M
    };
    let shoulder = head - Vec3::Y * SHOULDER_DOWN_M + right * side;
    let origin = j.index_proximal;
    let dir = (origin - shoulder).try_normalize()?;
    // up hint: world up, or the head's forward when the ray points straight up or down
    let up = if dir.dot(Vec3::Y).abs() > 0.98 {
        Vec3::from(hmd.matrix3 * Vec3A::NEG_Z)
    } else {
        Vec3::Y
    };
    let view = Affine3A::look_to_rh(origin, dir, up);
    let pose = view.inverse();
    pose.is_finite().then_some(pose)
}

/// A pinch with hysteresis.
#[derive(Debug, Default, Clone, Copy)]
pub struct Pinch {
    pub on: bool,
}

impl Pinch {
    pub fn update(&mut self, a: Vec3, b: Vec3) -> bool {
        let d = a.distance(b);
        if !d.is_finite() {
            self.on = false;
        } else if self.on {
            self.on = d < PINCH_OFF_M;
        } else {
            self.on = d < PINCH_ON_M;
        }
        self.on
    }
}

/// Per-hand pinch state.
#[derive(Debug, Default, Clone, Copy)]
pub struct HandGestures {
    index: Pinch,
    middle: Pinch,
}

impl HandGestures {
    /// Click = index pinch, grab = middle-finger pinch; when both fingers touch the thumb it is a grab only.
    pub fn sample(&mut self, j: &HandJoints, left: bool, hmd: &Affine3A) -> Option<HandSample> {
        let index = self.index.update(j.thumb_tip, j.index_tip);
        let middle = self.middle.update(j.thumb_tip, j.middle_tip);
        let pose = aim_pose(j, left, hmd)?;
        Some(HandSample {
            pose,
            click: index && !middle,
            grab: middle,
        })
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Who has the pointer.
#[derive(Debug, Default)]
pub struct HandPriority {
    last_seen: Option<Instant>,
    active: bool,
}

impl HandPriority {
    /// `prefer`: the `prefer_hands` setting; `any_hand`: a camera hand is tracked now. True while the hands have the
    /// pointer. The second value is Some(new state) when it changed (for the log).
    pub fn update(&mut self, prefer: bool, any_hand: bool, now: Instant) -> (bool, Option<bool>) {
        if any_hand {
            self.last_seen = Some(now);
        }
        let active = prefer
            && self
                .last_seen
                .is_some_and(|t| now.saturating_duration_since(t) <= HOLD);
        let changed = (active != self.active).then_some(active);
        self.active = active;
        (active, changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    fn joints(knuckle: Vec3, pinch_index: bool, pinch_middle: bool) -> HandJoints {
        let thumb = knuckle + Vec3::new(0.0, -0.03, -0.05);
        HandJoints {
            index_proximal: knuckle,
            thumb_tip: thumb,
            index_tip: if pinch_index {
                thumb + Vec3::new(0.01, 0.0, 0.0)
            } else {
                thumb + Vec3::new(0.0, 0.06, 0.0)
            },
            middle_tip: if pinch_middle {
                thumb + Vec3::new(0.012, 0.0, 0.0)
            } else {
                thumb + Vec3::new(0.0, 0.07, 0.02)
            },
        }
    }

    #[test]
    fn ray_points_from_the_shoulder_through_the_knuckle() {
        let hmd = Affine3A::from_translation(Vec3::new(0.0, 1.6, 0.0));
        // right hand in front of the right shoulder, at shoulder height
        let knuckle = Vec3::new(0.16, 1.43, -0.5);
        let p = aim_pose(&joints(knuckle, false, false), false, &hmd).unwrap();
        let fwd = Vec3::from(p.matrix3 * Vec3A::NEG_Z);
        assert!(fwd.distance(Vec3::NEG_Z) < 1e-4, "{fwd}");
        assert!(Vec3::from(p.translation).distance(knuckle) < 1e-5);
        // up stays up
        assert!(Vec3::from(p.matrix3 * Vec3A::Y).y > 0.99);
    }

    #[test]
    fn ray_follows_the_head_yaw_and_side() {
        // head turned 90 deg left: its right is world -Z; the left shoulder is then at +Z
        let hmd = Affine3A::from_rotation_translation(
            Quat::from_rotation_y(90f32.to_radians()),
            Vec3::new(0.0, 1.6, 0.0),
        );
        let knuckle = Vec3::new(-0.5, 1.43, 0.16);
        let p = aim_pose(&joints(knuckle, false, false), true, &hmd).unwrap();
        let fwd = Vec3::from(p.matrix3 * Vec3A::NEG_Z);
        assert!(fwd.distance(Vec3::NEG_X) < 1e-4, "{fwd}");
    }

    #[test]
    fn straight_up_does_not_break() {
        let hmd = Affine3A::from_translation(Vec3::new(0.0, 1.6, 0.0));
        let knuckle = Vec3::new(0.16, 2.2, 0.0);
        let p = aim_pose(&joints(knuckle, false, false), false, &hmd).unwrap();
        assert!(p.is_finite());
        assert!(Vec3::from(p.matrix3 * Vec3A::NEG_Z).y > 0.99);
    }

    #[test]
    fn pinch_has_hysteresis() {
        let mut p = Pinch::default();
        let o = Vec3::ZERO;
        assert!(
            !p.update(o, Vec3::new(0.03, 0.0, 0.0)),
            "3 cm is not a pinch yet"
        );
        assert!(p.update(o, Vec3::new(0.015, 0.0, 0.0)));
        assert!(p.update(o, Vec3::new(0.03, 0.0, 0.0)), "3 cm keeps a pinch");
        assert!(!p.update(o, Vec3::new(0.05, 0.0, 0.0)));
        assert!(!p.update(o, Vec3::new(f32::NAN, 0.0, 0.0)));
    }

    #[test]
    fn index_pinch_clicks_middle_pinch_grabs() {
        let hmd = Affine3A::from_translation(Vec3::new(0.0, 1.6, 0.0));
        let k = Vec3::new(0.16, 1.43, -0.5);
        let mut g = HandGestures::default();
        let s = g.sample(&joints(k, false, false), false, &hmd).unwrap();
        assert!(!s.click && !s.grab);
        let s = g.sample(&joints(k, true, false), false, &hmd).unwrap();
        assert!(s.click && !s.grab);
        let mut g = HandGestures::default();
        let s = g.sample(&joints(k, true, true), false, &hmd).unwrap();
        assert!(!s.click && s.grab, "both fingers: grab only");
    }

    #[test]
    fn hands_keep_the_pointer_for_the_hold_time() {
        let t0 = Instant::now();
        let mut p = HandPriority::default();
        assert_eq!(
            p.update(true, false, t0),
            (false, None),
            "no hand yet: controllers"
        );
        assert_eq!(p.update(true, true, t0), (true, Some(true)));
        assert_eq!(
            p.update(true, false, t0 + HOLD / 2),
            (true, None),
            "short loss: still the hands"
        );
        assert_eq!(
            p.update(true, false, t0 + HOLD * 2),
            (false, Some(false)),
            "long loss: controllers again"
        );
        assert_eq!(
            p.update(false, true, t0 + HOLD * 3),
            (false, None),
            "prefer_hands off: controllers"
        );
        assert_eq!(p.update(true, true, t0 + HOLD * 3), (true, Some(true)));
    }
}
