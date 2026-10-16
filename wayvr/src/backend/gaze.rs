//! Eye-gaze pointer: smoothing, fixation dead zone and dwell-click logic.
//!
//! The OpenXR backend feeds the raw gaze pose (stage space) into [`GazeState::feed`] every frame.
//! `backend::input::interact` then borrows a free controller slot to raycast the (filtered) gaze
//! against the overlays, so a gaze ray behaves like a controller ray that can only hover, plus click
//! from a controller trigger or a dwell timer.

use std::{
    f32::consts::PI,
    time::{Duration, Instant},
};

use glam::{Affine3A, Quat, Vec3, Vec3A};

use crate::{backend::input::Pointer, windowing::OverlayID};

/// Gaze is considered lost if no valid sample arrived for this long.
pub const GAZE_STALE: Duration = Duration::from_millis(500);
/// Raw gaze samples that are bit-identical this many frames in a row are treated as a frozen tracker.
const FROZEN_FRAMES: u32 = 45;
/// How long a dwell click is held down.
const DWELL_PRESS: Duration = Duration::from_millis(90);
/// Gaze must stay within this distance (meters, on the hit surface) to count as "same spot".
const DWELL_RADIUS: f32 = 0.03;

/// One Euro filter (Casiez et al.) on a 3D vector.
#[derive(Debug, Clone, Default)]
pub struct OneEuro3 {
    prev: Option<Vec3>,
    dprev: Vec3,
}

fn alpha(cutoff: f32, dt: f32) -> f32 {
    let tau = 1.0 / (2.0 * PI * cutoff.max(1e-3));
    1.0 / (1.0 + tau / dt.max(1e-4))
}

impl OneEuro3 {
    pub const fn reset(&mut self) {
        self.prev = None;
        self.dprev = Vec3::ZERO;
    }

    pub fn filter(&mut self, x: Vec3, dt: f32, min_cutoff: f32, beta: f32) -> Vec3 {
        let Some(prev) = self.prev else {
            self.prev = Some(x);
            self.dprev = Vec3::ZERO;
            return x;
        };
        let dt = dt.clamp(1e-4, 0.25);
        let dx = (x - prev) / dt;
        let a_d = alpha(1.0, dt);
        let dhat = self.dprev.lerp(dx, a_d);
        self.dprev = dhat;
        let cutoff = min_cutoff + beta * dhat.length();
        let out = prev.lerp(x, alpha(cutoff, dt));
        self.prev = Some(out);
        out
    }
}

/// Map the user-facing smoothing value (0..1) to One Euro parameters `(min_cutoff_hz, beta)`.
pub fn smoothing_params(smoothing: f32) -> Option<(f32, f32)> {
    let s = smoothing.clamp(0.0, 1.0);
    if s <= 0.001 {
        return None;
    }
    // 0.5 -> ~1.9 Hz at rest, 1.0 -> 0.3 Hz. Beta makes fast saccades pass through nearly unfiltered.
    let min_cutoff = 8.0f32 * (0.3f32 / 8.0).powf(s);
    Some((min_cutoff, 2.5))
}

/// Hold `current` still while `target` is within `deadzone_deg` of it. Outside the dead zone the output
/// follows `target`, trailing it by exactly the dead zone angle (no jump when leaving).
pub fn apply_deadzone(current: Vec3, target: Vec3, deadzone_deg: f32) -> Vec3 {
    let dz = deadzone_deg.max(0.0).to_radians();
    let c = current.normalize_or_zero();
    let t = target.normalize_or_zero();
    if c == Vec3::ZERO {
        return t;
    }
    let angle = c.dot(t).clamp(-1.0, 1.0).acos();
    if angle <= dz || angle < 1e-6 {
        return c;
    }
    let f = (angle - dz) / angle;
    Quat::IDENTITY
        .slerp(Quat::from_rotation_arc(c, t), f)
        .mul_vec3(c)
        .normalize()
}

#[derive(Debug, Default)]
pub struct Dwell {
    anchor: Option<(OverlayID, Vec3A)>,
    since: Option<Instant>,
    press_until: Option<Instant>,
    /// after a click, gaze has to leave the spot before another one can fire
    spent: bool,
}

impl Dwell {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Returns whether a dwell click is currently held down.
    pub fn update(&mut self, hit: Option<(OverlayID, Vec3A)>, now: Instant, dwell_ms: u32) -> bool {
        if dwell_ms == 0 {
            self.reset();
            return false;
        }

        if let Some(until) = self.press_until {
            if now < until {
                return true;
            }
            self.press_until = None;
        }

        let Some((id, pos)) = hit else {
            self.anchor = None;
            self.since = None;
            self.spent = false;
            return false;
        };

        let same = self
            .anchor
            .is_some_and(|(aid, apos)| aid == id && apos.distance(pos) <= DWELL_RADIUS);
        if !same {
            self.anchor = Some((id, pos));
            self.since = Some(now);
            self.spent = false;
            return false;
        }

        if !self.spent
            && self.since.is_some_and(|s| {
                now.duration_since(s) >= Duration::from_millis(u64::from(dwell_ms))
            })
        {
            self.spent = true;
            self.press_until = Some(now + DWELL_PRESS);
            return true;
        }
        false
    }
}

/// Counters for the periodic diagnostics line.
#[derive(Debug, Default)]
pub struct GazeStats {
    pub frames: u32,
    /// located with a valid orientation
    pub located: u32,
    /// accepted by `feed` (finite, not frozen)
    pub fed_ok: u32,
    pub last_flags: u32,
    pub since: Option<Instant>,
}

pub struct GazeState {
    pub stats: GazeStats,
    /// Virtual pointer holding gaze pose / click / hover state. Swapped into a free controller slot
    /// while interacting.
    pub ptr: Pointer,
    /// Controller slot the gaze pointer last interacted through.
    pub slot: Option<usize>,
    /// Filtered gaze ray is usable this frame.
    pub active: bool,
    pub dwell: Dwell,
    pub last_hit: Option<(OverlayID, Vec3A)>,
    filter: OneEuro3,
    out_dir: Option<Vec3>,
    last_feed: Option<Instant>,
    last_valid: Option<Instant>,
    last_raw: Option<(Quat, Vec3)>,
    frozen: u32,
    pub dwell_ms: u32,
    /// A sample was accepted this frame (false during blinks / invalid frames, even within the grace period).
    pub fresh: bool,
    /// Silent gaze tracking runs (mirrors the config for the IPC answer).
    pub tracking: bool,
    /// `wayvrctl gaze` answers with fixations (config `gaze_target` or `gaze_pointer`). With only the visual marker
    /// (`gaze_laser`) the answer says `tracking: false`, so nothing can target by gaze.
    pub reporting: bool,
    /// The laser is on (mirrors the config).
    pub laser_on: bool,
    /// This frame's laser stroke `(a, b, width)`, if it is drawn.
    pub laser: Option<(Vec3A, Vec3A, f32)>,
    /// Fixations for `wayvrctl gaze`.
    pub target: crate::backend::gaze_target::GazeTarget,
}

impl GazeState {
    pub fn new() -> Self {
        Self {
            stats: GazeStats::default(),
            ptr: Pointer::new(0),
            slot: None,
            active: false,
            dwell: Dwell::default(),
            last_hit: None,
            filter: OneEuro3::default(),
            out_dir: None,
            last_feed: None,
            last_valid: None,
            last_raw: None,
            frozen: 0,
            dwell_ms: 0,
            fresh: false,
            tracking: false,
            reporting: false,
            laser_on: false,
            laser: None,
            target: crate::backend::gaze_target::GazeTarget::default(),
        }
    }

    fn lose(&mut self) {
        self.active = false;
        self.filter.reset();
        self.out_dir = None;
        self.last_feed = None;
        self.last_raw = None;
        self.frozen = 0;
        self.ptr.tracked = false;
        self.dwell.reset();
        self.last_hit = None;
    }

    /// Feed this frame's raw gaze pose (stage space, -Z forward), or `None` if unavailable/invalid.
    pub fn feed(
        &mut self,
        raw: Option<Affine3A>,
        smoothing: f32,
        deadzone_deg: f32,
        dwell_ms: u32,
        now: Instant,
    ) {
        self.dwell_ms = dwell_ms;
        self.fresh = false;

        let valid = raw.filter(|p| {
            let t = p.translation;
            let ok = p.to_cols_array().iter().all(|f| f.is_finite())
                && p.matrix3.x_axis.length_squared() > 0.5;
            ok && t.is_finite()
        });

        let Some(raw) = valid else {
            // keep the last filtered ray for a short grace period (blinks), then give up
            if self
                .last_valid
                .is_none_or(|t| now.duration_since(t) > GAZE_STALE)
            {
                self.lose();
            }
            return;
        };

        let (rot, pos) = {
            let (_, r, t) = raw.to_scale_rotation_translation();
            (r, t)
        };

        // frozen tracker detection
        if self.last_raw == Some((rot, pos)) {
            self.frozen += 1;
            if self.frozen >= FROZEN_FRAMES {
                self.lose();
                self.last_raw = Some((rot, pos));
                self.frozen = FROZEN_FRAMES;
                return;
            }
        } else {
            self.frozen = 0;
        }
        self.last_raw = Some((rot, pos));
        self.last_valid = Some(now);
        self.stats.fed_ok += 1;

        let dt = self
            .last_feed
            .map_or(1.0 / 90.0, |t| now.duration_since(t).as_secs_f32());
        self.last_feed = Some(now);

        let raw_dir = rot * Vec3::NEG_Z;
        let smoothed = match smoothing_params(smoothing) {
            Some((min_cutoff, beta)) => self
                .filter
                .filter(raw_dir, dt, min_cutoff, beta)
                .normalize_or_zero(),
            None => raw_dir,
        };
        let smoothed = if smoothed == Vec3::ZERO {
            raw_dir
        } else {
            smoothed
        };

        let out = match self.out_dir {
            Some(cur) => apply_deadzone(cur, smoothed, deadzone_deg),
            None => smoothed,
        };
        self.out_dir = Some(out);

        let delta = Quat::from_rotation_arc(raw_dir, out);
        let pose = Affine3A::from_rotation_translation(delta * rot, pos);
        self.ptr.raw_pose = raw;
        self.ptr.pose = pose;
        self.ptr.tracked = true;
        self.active = true;
        self.fresh = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn angle(a: Vec3, b: Vec3) -> f32 {
        a.normalize()
            .dot(b.normalize())
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
    }

    #[test]
    fn one_euro_first_sample_passthrough_and_converges() {
        let mut f = OneEuro3::default();
        let a = Vec3::new(0.0, 0.0, -1.0);
        assert_eq!(f.filter(a, 0.011, 1.0, 2.5), a);
        let b = Vec3::new(0.1, 0.0, -1.0);
        let mut out = a;
        for _ in 0..400 {
            out = f.filter(b, 0.011, 1.0, 2.5);
        }
        assert!((out - b).length() < 1e-3);
    }

    #[test]
    fn one_euro_reduces_jitter() {
        let mut f = OneEuro3::default();
        let (mc, beta) = smoothing_params(0.5).unwrap();
        let mut max_dev = 0.0f32;
        for i in 0..200 {
            let j = if i % 2 == 0 { 0.003 } else { -0.003 };
            let o = f.filter(Vec3::new(j, 0.0, -1.0), 1.0 / 90.0, mc, beta);
            if i > 20 {
                max_dev = max_dev.max(o.x.abs());
            }
        }
        assert!(max_dev < 0.0015, "dev {max_dev}");
    }

    #[test]
    fn smoothing_zero_is_off() {
        assert!(smoothing_params(0.0).is_none());
        assert!(smoothing_params(1.0).unwrap().0 < smoothing_params(0.5).unwrap().0);
    }

    #[test]
    fn deadzone_holds_then_trails() {
        let c = Vec3::NEG_Z;
        let near = Quat::from_rotation_y(0.3f32.to_radians()) * c;
        assert_eq!(apply_deadzone(c, near, 0.6), c);
        let far = Quat::from_rotation_y(3.0f32.to_radians()) * c;
        let out = apply_deadzone(c, far, 0.6);
        assert!((angle(out, far) - 0.6).abs() < 0.01, "{}", angle(out, far));
        assert!(angle(out, c) > 2.0);
    }

    #[test]
    fn dwell_fires_once_then_rearms() {
        let id = OverlayID::default();
        let mut d = Dwell::default();
        let t0 = Instant::now();
        let hit = Some((id, Vec3A::new(0.0, 0.0, -1.0)));
        assert!(!d.update(hit, t0, 500));
        assert!(!d.update(hit, t0 + Duration::from_millis(400), 500));
        assert!(d.update(hit, t0 + Duration::from_millis(510), 500));
        assert!(d.update(hit, t0 + Duration::from_millis(550), 500)); // still pressed
        assert!(!d.update(hit, t0 + Duration::from_millis(700), 500)); // released, spent
        assert!(!d.update(hit, t0 + Duration::from_millis(2000), 500));
        // look away, then back
        assert!(!d.update(None, t0 + Duration::from_millis(2100), 500));
        assert!(!d.update(hit, t0 + Duration::from_millis(2200), 500));
        assert!(d.update(hit, t0 + Duration::from_millis(2750), 500));
    }

    #[test]
    fn dwell_resets_on_move_and_disabled() {
        let id = OverlayID::default();
        let mut d = Dwell::default();
        let t0 = Instant::now();
        assert!(!d.update(Some((id, Vec3A::ZERO)), t0, 300));
        assert!(!d.update(
            Some((id, Vec3A::new(0.2, 0.0, 0.0))),
            t0 + Duration::from_millis(250),
            300
        ));
        assert!(!d.update(
            Some((id, Vec3A::new(0.2, 0.0, 0.0))),
            t0 + Duration::from_millis(400),
            300
        ));
        assert!(!d.update(
            Some((id, Vec3A::new(0.2, 0.0, 0.0))),
            t0 + Duration::from_millis(900),
            0
        ));
    }

    #[test]
    fn feed_invalid_and_stale() {
        let mut g = GazeState::new();
        let t0 = Instant::now();
        g.feed(None, 0.5, 0.6, 0, t0);
        assert!(!g.active);
        let p = Affine3A::from_rotation_translation(Quat::IDENTITY, Vec3::new(0.0, 1.6, 0.0));
        g.feed(Some(p), 0.5, 0.6, 0, t0);
        assert!(g.active && g.ptr.tracked);
        // blink: keeps ray for grace period
        g.feed(None, 0.5, 0.6, 0, t0 + Duration::from_millis(100));
        assert!(g.active);
        g.feed(None, 0.5, 0.6, 0, t0 + Duration::from_millis(700));
        assert!(!g.active && !g.ptr.tracked);
        // NaN pose rejected
        let bad = Affine3A::from_translation(Vec3::NAN);
        g.feed(Some(bad), 0.5, 0.6, 0, t0 + Duration::from_millis(800));
        assert!(!g.active);
    }

    #[test]
    fn frozen_tracker_goes_inactive() {
        let mut g = GazeState::new();
        let t0 = Instant::now();
        let p = Affine3A::from_rotation_translation(Quat::IDENTITY, Vec3::new(0.0, 1.6, 0.0));
        for i in 0..(FROZEN_FRAMES + 2) {
            g.feed(
                Some(p),
                0.5,
                0.6,
                0,
                t0 + Duration::from_millis(11 * u64::from(i)),
            );
        }
        assert!(!g.active);
    }
}
