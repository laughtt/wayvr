//! Silent gaze tracking: where the eyes land on the overlays, as fixations.
//!
//! Every frame the (filtered) gaze ray is ray-cast against the overlays without hovering, clicking or moving
//! the mouse (`backend::input::track_gaze_silent`), and the hit is fed into [`GazeTarget::update`].
//! A fixation is a run of hits on the same overlay that stays inside a [`DISPERSION_DEG`] cone around its
//! running mean direction; short invalid stretches (blinks, < [`BLINK_MAX_MS`]) do not end it.
//! [`GazeTarget::report`] builds the `wayvrctl gaze` answer: the fixation in progress, the last stable one
//! (≥ `min_fixation_ms`, ended in the last [`STABLE_WINDOW_MS`]) and a short history, so a voice command
//! spoken after looking away still resolves.
//!
//! The module also places the laser marker ([`laser_segment`]). Pure logic, no GPU or IPC: unit tested.

use std::{collections::VecDeque, sync::Arc};

use glam::{Affine3A, Vec2, Vec3, Vec3A};
use wayvr_ipc::packet_server::{WlxGaze, WlxGazeFixation};
use wlx_common::config::GazeLaserStyle;

/// A new fixation starts when the gaze leaves this cone (degrees) around the current fixation's mean.
/// About two thirds of the measured calibration error (3.08 deg), so jitter stays inside, saccades do not.
pub const DISPERSION_DEG: f32 = 2.0;
/// Invalid stretches shorter than this (blinks) do not end a fixation.
pub const BLINK_MAX_MS: u64 = 300;
/// `last_stable` falls back to a finished fixation that ended at most this long ago.
pub const STABLE_WINDOW_MS: u64 = 2000;
/// Finished fixations are kept this long.
pub const HISTORY_MS: u64 = 10_000;
const HISTORY_MAX: usize = 32;

/// One frame's gaze hit.
#[derive(Clone, Debug)]
pub struct GazeSample {
    pub overlay: Arc<str>,
    /// Overlay UV (0..1, origin top-left)
    pub uv: Vec2,
    /// Desktop (compositor logical) pixel, for screen overlays
    pub px: Option<Vec2>,
    /// Gaze direction (stage space)
    pub dir: Vec3,
    /// Eyes to hit point, metres
    pub dist: f32,
}

#[derive(Clone, Debug)]
struct Acc {
    overlay: Arc<str>,
    start_ms: u64,
    last_ms: u64,
    hits: u32,
    invalid: u32,
    sum_uv: Vec2,
    sum_px: Vec2,
    px_n: u32,
    sum_dir: Vec3,
    sum_dist: f32,
    sum_sq_dev: f32,
}

impl Acc {
    fn new(s: &GazeSample, now_ms: u64) -> Self {
        let mut a = Self {
            overlay: s.overlay.clone(),
            start_ms: now_ms,
            last_ms: now_ms,
            hits: 0,
            invalid: 0,
            sum_uv: Vec2::ZERO,
            sum_px: Vec2::ZERO,
            px_n: 0,
            sum_dir: Vec3::ZERO,
            sum_dist: 0.0,
            sum_sq_dev: 0.0,
        };
        a.add(s, now_ms);
        a
    }

    fn mean_dir(&self) -> Vec3 {
        self.sum_dir.normalize_or_zero()
    }

    fn add(&mut self, s: &GazeSample, now_ms: u64) {
        if self.hits > 0 {
            let d = angle_deg(self.mean_dir(), s.dir);
            self.sum_sq_dev += d * d;
        }
        self.hits += 1;
        self.last_ms = now_ms;
        self.sum_uv += s.uv;
        if let Some(px) = s.px {
            self.sum_px += px;
            self.px_n += 1;
        }
        self.sum_dir += s.dir.normalize_or_zero();
        self.sum_dist += s.dist;
    }

    fn accepts(&self, s: &GazeSample) -> bool {
        self.overlay == s.overlay && angle_deg(self.mean_dir(), s.dir) <= DISPERSION_DEG
    }

    fn duration_ms(&self) -> u32 {
        u32::try_from(self.last_ms.saturating_sub(self.start_ms)).unwrap_or(u32::MAX)
    }

    fn fixation(&self, min_fixation_ms: u32) -> WlxGazeFixation {
        let n = self.hits.max(1) as f32;
        let dispersion = if self.hits > 1 {
            (self.sum_sq_dev / (self.hits - 1) as f32).sqrt()
        } else {
            0.0
        };
        let duration_ms = self.duration_ms();
        // JSON has no NaN (it would come back as null and fail to decode)
        let fin = |x: f32| if x.is_finite() { x } else { 0.0 };
        let fin2 = |v: Vec2| [fin(v.x), fin(v.y)];
        WlxGazeFixation {
            overlay: self.overlay.to_string(),
            uv: fin2(self.sum_uv / n),
            px: (self.px_n > 0).then(|| fin2(self.sum_px / self.px_n as f32)),
            start_ms: self.start_ms,
            end_ms: self.last_ms,
            duration_ms,
            confidence: confidence(self.hits, self.invalid, dispersion, duration_ms, min_fixation_ms),
            dispersion_deg: fin(dispersion),
            dist_m: fin(self.sum_dist / n),
            samples: self.hits,
        }
    }
}

fn angle_deg(a: Vec3, b: Vec3) -> f32 {
    let (a, b) = (a.normalize_or_zero(), b.normalize_or_zero());
    if a == Vec3::ZERO || b == Vec3::ZERO {
        return 180.0;
    }
    a.dot(b).clamp(-1.0, 1.0).acos().to_degrees()
}

/// 0..1 from the share of valid frames, the angular spread and the length (relative to the minimum).
pub fn confidence(hits: u32, invalid: u32, dispersion_deg: f32, duration_ms: u32, min_fixation_ms: u32) -> f32 {
    let frames = (hits + invalid).max(1) as f32;
    let valid_share = hits as f32 / frames;
    let spread = 1.0 - 0.5 * (dispersion_deg / DISPERSION_DEG).clamp(0.0, 1.0);
    let length = (duration_ms as f32 / (2.0 * min_fixation_ms.max(1) as f32)).clamp(0.0, 1.0);
    let c = valid_share * spread * (0.5 + 0.5 * length);
    (c * 100.0).round() / 100.0
}

/// Fixation tracker behind `wayvrctl gaze`.
#[derive(Debug, Default)]
pub struct GazeTarget {
    cur: Option<Acc>,
    history: VecDeque<WlxGazeFixation>,
    valid: bool,
    min_fixation_ms: u32,
}

impl GazeTarget {
    pub fn reset(&mut self) {
        self.cur = None;
        self.history.clear();
        self.valid = false;
    }

    fn close(&mut self) {
        let Some(acc) = self.cur.take() else {
            return;
        };
        if acc.duration_ms() >= self.min_fixation_ms {
            self.history.push_back(acc.fixation(self.min_fixation_ms));
            while self.history.len() > HISTORY_MAX {
                self.history.pop_front();
            }
        }
    }

    /// Feed one frame. `valid`: a fresh gaze sample arrived this frame; `sample`: what it hit (None = nothing).
    pub fn update(&mut self, now_ms: u64, sample: Option<GazeSample>, valid: bool, min_fixation_ms: u32) {
        self.min_fixation_ms = min_fixation_ms;
        self.valid = valid;
        while self
            .history
            .front()
            .is_some_and(|f| f.end_ms + HISTORY_MS < now_ms)
        {
            self.history.pop_front();
        }

        // a gap longer than a blink ends the fixation in progress
        if self
            .cur
            .as_ref()
            .is_some_and(|c| now_ms.saturating_sub(c.last_ms) > BLINK_MAX_MS)
        {
            self.close();
        }

        if !valid {
            if let Some(c) = self.cur.as_mut() {
                c.invalid += 1;
            }
            return;
        }
        let Some(s) = sample else {
            // looking at nothing
            self.close();
            return;
        };
        if let Some(c) = self.cur.as_mut()
            && c.accepts(&s)
        {
            c.add(&s, now_ms);
            return;
        }
        self.close();
        self.cur = Some(Acc::new(&s, now_ms));
    }

    /// The `wayvrctl gaze` answer.
    pub fn report(&self, now_ms: u64, tracking: bool, laser: bool) -> WlxGaze {
        let min = self.min_fixation_ms;
        let current = self.cur.as_ref().map(|c| c.fixation(min));
        let history: Vec<_> = self
            .history
            .iter()
            .filter(|f| f.end_ms + HISTORY_MS >= now_ms)
            .cloned()
            .collect();
        let last_stable = current
            .clone()
            .filter(|c| c.duration_ms >= min)
            .or_else(|| {
                history
                    .iter()
                    .rev()
                    .find(|f| f.end_ms + STABLE_WINDOW_MS >= now_ms)
                    .cloned()
            });
        WlxGaze {
            ts_ms: now_ms,
            tracking,
            laser,
            valid: self.valid,
            min_fixation_ms: min,
            current,
            last_stable,
            history,
        }
    }
}

/// Unix epoch milliseconds.
pub fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// The laser stroke `(a, b, width)` for a gaze hit. `eye`: gaze origin; `hit`: hit point; `hmd`: head pose.
/// Dot: a square of about 0.6 deg (min 1 cm, max 4 cm) facing the head, pulled 5 mm toward the eyes.
/// Ray: the last 25 cm of a 1.5 mm line from below the chin to the hit point (a ray along the gaze itself
/// would be seen end-on).
pub fn laser_segment(style: GazeLaserStyle, eye: Vec3A, hit: Vec3A, hmd: &Affine3A) -> (Vec3A, Vec3A, f32) {
    let to_hit = hit - eye;
    let dist = to_hit.length();
    let dir = to_hit.normalize_or_zero();
    match style {
        GazeLaserStyle::Dot => {
            let s = (dist * 0.6f32.to_radians().tan()).clamp(0.0101, 0.04);
            let c = hit - dir * 0.005;
            let up = hmd.y_axis.normalize_or_zero();
            (c - up * (s * 0.5), c + up * (s * 0.5), s)
        }
        GazeLaserStyle::Ray => {
            let origin = eye - hmd.y_axis.normalize_or_zero() * 0.15;
            let d = (hit - origin).normalize_or_zero();
            let len = 0.25f32.min((hit - origin).length() * 0.5);
            (hit - d * len, hit - d * 0.003, 0.0015)
        }
    }
}

/// Premultiplied RGBA of the laser: a cool white that reads on passthrough and on the screens.
pub fn laser_rgba(alpha: f32) -> [f32; 4] {
    let a = alpha.clamp(0.05, 1.0);
    [0.85 * a, 0.95 * a, 1.0 * a, a]
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    fn s(overlay: &str, uv: (f32, f32), yaw_deg: f32) -> GazeSample {
        GazeSample {
            overlay: overlay.into(),
            uv: Vec2::new(uv.0, uv.1),
            px: Some(Vec2::new(uv.0 * 1920.0, uv.1 * 1080.0)),
            dir: Quat::from_rotation_y(yaw_deg.to_radians()) * Vec3::NEG_Z,
            dist: 1.2,
        }
    }

    /// 90 Hz frames from `t0` for `ms` milliseconds.
    fn run(g: &mut GazeTarget, t0: u64, ms: u64, f: impl Fn(u64) -> (Option<GazeSample>, bool)) -> u64 {
        let mut t = t0;
        while t < t0 + ms {
            let (smp, valid) = f(t);
            g.update(t, smp, valid, 200);
            t += 11;
        }
        t
    }

    #[test]
    fn steady_gaze_becomes_stable_fixation() {
        let mut g = GazeTarget::default();
        let t = run(&mut g, 1000, 150, |i| (Some(s("SUNSHINE", (0.5, 0.5), (i % 3) as f32 * 0.3)), true));
        let r = g.report(t, true, true);
        assert!(r.current.is_some() && r.last_stable.is_none(), "too short yet");
        let t = run(&mut g, t, 300, |_| (Some(s("SUNSHINE", (0.5, 0.5), 0.2)), true));
        let r = g.report(t, true, true);
        let st = r.last_stable.expect("stable after 200 ms");
        assert_eq!(st.overlay, "SUNSHINE");
        assert!(st.duration_ms >= 400, "{}", st.duration_ms);
        assert!((st.uv[0] - 0.5).abs() < 1e-4);
        assert_eq!(st.px.map(|p| p[0].round()), Some(960.0));
        assert!(st.confidence > 0.7, "{}", st.confidence);
        assert!(st.dispersion_deg < 0.5);
    }

    #[test]
    fn saccade_starts_new_fixation_and_keeps_history() {
        let mut g = GazeTarget::default();
        let t = run(&mut g, 0, 400, |_| (Some(s("SUNSHINE", (0.2, 0.3), -10.0)), true));
        let t = run(&mut g, t, 100, |_| (Some(s("SUNSHINE", (0.8, 0.3), 10.0)), true));
        let r = g.report(t, true, false);
        assert_eq!(r.history.len(), 1);
        assert!((r.current.as_ref().unwrap().uv[0] - 0.8).abs() < 1e-4);
        // current too short: last_stable falls back to the finished one
        let ls = r.last_stable.unwrap();
        assert!((ls.uv[0] - 0.2).abs() < 1e-4);
        // a short glance is not kept
        let t = run(&mut g, t, 50, |_| (Some(s("SUNSHINE", (0.5, 0.9), 0.0)), true));
        assert_eq!(g.report(t, true, false).history.len(), 1);
    }

    #[test]
    fn blink_does_not_end_fixation_but_long_loss_does() {
        let mut g = GazeTarget::default();
        let t = run(&mut g, 0, 300, |_| (Some(s("A", (0.5, 0.5), 0.0)), true));
        let t = run(&mut g, t, 150, |_| (None, false)); // blink
        let t = run(&mut g, t, 100, |_| (Some(s("A", (0.5, 0.5), 0.0)), true));
        let r = g.report(t, true, false);
        let c = r.current.unwrap();
        assert!(c.duration_ms >= 500 && r.history.is_empty());
        assert!(c.confidence < 1.0, "invalid frames lower it: {}", c.confidence);
        // tracking lost for longer than a blink
        let t = run(&mut g, t, 500, |_| (None, false));
        let r = g.report(t, true, false);
        assert!(r.current.is_none() && !r.valid);
        assert_eq!(r.history.len(), 1);
        assert!(r.last_stable.is_some(), "ended < 2 s ago");
        // after 2 s it is no longer `last_stable`, after 10 s it leaves the history
        let r = g.report(t + 2500, true, false);
        assert!(r.last_stable.is_none() && r.history.len() == 1);
        g.update(t + 11_000, None, false, 200);
        assert!(g.report(t + 11_000, true, false).history.is_empty());
    }

    #[test]
    fn other_overlay_or_nothing_ends_fixation() {
        let mut g = GazeTarget::default();
        let t = run(&mut g, 0, 300, |_| (Some(s("A", (0.5, 0.5), 0.0)), true));
        let t = run(&mut g, t, 50, |_| (Some(s("B", (0.5, 0.5), 0.0)), true));
        assert_eq!(g.report(t, true, false).current.unwrap().overlay, "B");
        let t = run(&mut g, t, 300, |_| (None, true));
        let r = g.report(t, true, false);
        assert!(r.current.is_none() && r.valid);
        assert_eq!(r.history.len(), 1, "B was too short");
    }

    #[test]
    fn confidence_bounds() {
        assert_eq!(confidence(40, 0, 0.0, 400, 200), 1.0);
        assert!(confidence(20, 20, 0.0, 400, 200) <= 0.5);
        assert!(confidence(40, 0, 2.0, 400, 200) <= 0.5);
        assert!(confidence(1, 0, 0.0, 0, 200) <= 0.5);
    }

    #[test]
    fn laser_dot_and_ray_geometry() {
        let hmd = Affine3A::from_translation(Vec3::new(0.0, 1.6, 0.0));
        let eye = Vec3A::new(0.0, 1.6, 0.0);
        let hit = Vec3A::new(0.0, 1.6, -2.0);
        let (a, b, w) = laser_segment(GazeLaserStyle::Dot, eye, hit, &hmd);
        assert!((a.distance(b) - w).abs() < 1e-5 && w >= 0.0101 && w <= 0.04);
        assert!(((a + b) * 0.5).z > -2.0, "pulled toward the eyes");
        let (a, b, w) = laser_segment(GazeLaserStyle::Ray, eye, hit, &hmd);
        assert!(w < 0.002 && a.distance(b) > 0.2 && a.distance(b) < 0.26);
        assert!(b.distance(hit) < 0.01 && a.y < b.y, "rises toward the hit");
        let c = laser_rgba(0.35);
        assert!(c[3] == 0.35 && c[0] <= c[3]);
    }
}
