//! Eye-gaze pointer: smoothing, fixation dead zone and dwell-click logic.
//!
//! The OpenXR backend feeds the raw gaze pose (stage space) into [`GazeState::feed`] every frame.
//! `backend::input::interact` then borrows a free controller slot to raycast the (filtered) gaze
//! against the overlays, so a gaze ray behaves like a controller ray that can only hover, plus click
//! from a controller trigger or a dwell timer.

use std::{
    collections::VecDeque,
    f32::consts::PI,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use glam::{Affine3A, DVec3, Quat, Vec3, Vec3A};

use crate::{
    backend::{
        gaze_lessons::{Lesson, LessonStore, unix_secs},
        input::Pointer,
    },
    windowing::OverlayID,
};

/// Raw head-relative gaze samples are kept this long (gaze click v2).
const HISTORY: Duration = Duration::from_millis(500);
/// The "look" at a moment is the per-axis median of the raw samples in this window before it (Frametop LOOK).
const LOOK: Duration = Duration::from_millis(300);
/// A refine lesson further than this (degrees) past the current correction is refused.
const REFINE_MAX_DEG: f64 = 20.0;
/// A laser lesson is taken only if the corrected gaze was within this many degrees of the laser hit.
const LASER_MAX_DEG: f64 = 4.0;
/// A gaze-check lesson is refused past this many degrees.
const CHECK_MAX_DEG: f64 = 15.0;
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

// ---------------------------------------------------------------------------------------------------------------
// Gaze click v2: angles, hold-to-refine click, gaze check

/// Head-relative direction (-Z forward, +X right, +Y up) to (yaw degrees, + left; pitch degrees, + up).
pub fn dir_to_angles(d: Vec3) -> (f64, f64) {
    let d = DVec3::new(f64::from(d.x), f64::from(d.y), f64::from(d.z)).normalize_or_zero();
    ((-d.x).atan2(-d.z).to_degrees(), d.y.clamp(-1.0, 1.0).asin().to_degrees())
}

/// The inverse of [`dir_to_angles`].
pub fn angles_to_dir(yaw: f64, pitch: f64) -> Vec3 {
    let (y, p) = (yaw.to_radians(), pitch.to_radians());
    Vec3::new(
        (-y.sin() * p.cos()) as f32,
        p.sin() as f32,
        (-y.cos() * p.cos()) as f32,
    )
}

fn angle_deg(a: Vec3, b: Vec3) -> f32 {
    a.normalize_or_zero()
        .dot(b.normalize_or_zero())
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        f64::midpoint(v[n / 2 - 1], v[n / 2])
    }
}

/// The error of the head-relative angles `a` against `b`, degrees.
fn angles_err(a: (f64, f64), b: (f64, f64)) -> f64 {
    f64::from(angle_deg(angles_to_dir(a.0, a.1), angles_to_dir(b.0, b.1)))
}

/// Hold the trigger still for this long and the click turns into a real press (drag).
const REFINE_PRESS_AFTER: Duration = Duration::from_millis(500);
/// A release sooner than this, without head movement, is a tap: it clicks at the frozen spot.
const REFINE_TAP: Duration = Duration::from_millis(250);
/// The head counts as still when it moved less than this (degrees) over the last [`REFINE_PRESS_AFTER`].
const REFINE_STILL_DEG: f32 = 0.5;
/// The refined head pointer holds still inside this dead zone and then trails the head.
const REFINE_DEADZONE_DEG: f32 = 0.5;
/// The head moved the dot at least this much (degrees): the click is a refined one.
const REFINE_MOVED_DEG: f32 = 0.2;
/// A refined / tapped click is held down this long.
const REFINE_CLICK: Duration = Duration::from_millis(60);

/// What the refine state machine tells `interact_gaze` this frame.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RefineOut {
    /// Replace the gaze pointer direction (stage space) with this; `None` = the normal gaze.
    pub dir: Option<Vec3>,
    /// The gaze pointer's button is down.
    pub click: bool,
    /// A refined click starts this frame: record a lesson from where the click lands.
    pub lesson: Option<RefineLesson>,
}

/// What a refine lesson needs from the moment the trigger went down.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefineLesson {
    /// Raw head-relative look (yaw, pitch degrees) at the press.
    pub look: Option<(f64, f64)>,
    pub hmd_rot0: Quat,
    pub hmd_pos0: Vec3,
}

#[derive(Debug, Clone)]
struct Hold {
    t0: Instant,
    anchor: Vec3,
    rot0: Quat,
    pos0: Vec3,
    look: Option<(f64, f64)>,
    moved: bool,
    ctrl: usize,
    dir: Vec3,
    rots: VecDeque<(Instant, Quat)>,
    lesson_given: bool,
}

#[derive(Debug, Clone, Default)]
enum RState {
    #[default]
    Idle,
    Hold(Box<Hold>),
    Pressed(Box<Hold>),
    Clicking {
        until: Instant,
        dir: Vec3,
    },
}

/// Hold-to-refine gaze click (only used through `gate::refine`).
///
/// Trigger down freezes the gaze at the anchor and presses nothing. While held, the head moves the pointer (gain 1,
/// with a small trailing dead zone). A tap clicks at the anchor, a release after head movement clicks at the refined
/// spot, a 0.5 s still hold presses for real and the head then drags until release.
#[derive(Debug, Default)]
pub struct Refine {
    state: RState,
    /// Raw controller buttons last frame: a Hold arms only on a rising edge seen after the button was low.
    prev_buttons: [bool; 2],
}

/// A refined click teaches only when the refined spot is at least this far (degrees) from the frozen one.
const REFINE_LESSON_DEG: f32 = 1.0;

impl Refine {
    /// Gaze unusable (or refine off): drop the state but keep the button memory, so a trigger held across the
    /// drop never makes a tap later. `buttons`: the raw controller buttons this frame.
    pub fn observe(&mut self, buttons: [bool; 2]) {
        self.state = RState::Idle;
        self.prev_buttons = buttons;
    }

    /// The controller that owns the Hold / press.
    pub const fn ctrl(&self) -> Option<usize> {
        match &self.state {
            RState::Hold(h) | RState::Pressed(h) => Some(h.ctrl),
            _ => None,
        }
    }

    /// The refine logic owns the pointer (frozen, refining, pressed or clicking).
    pub const fn active(&self) -> bool {
        !matches!(self.state, RState::Idle)
    }

    fn head_pointer(h: &mut Hold, rot: Quat) -> Vec3 {
        let target = (rot * h.rot0.inverse()) * h.anchor;
        if angle_deg(h.anchor, target) > REFINE_MOVED_DEG {
            h.moved = true;
        }
        h.dir = apply_deadzone(h.dir, target, REFINE_DEADZONE_DEG);
        h.dir
    }

    /// One frame. `buttons`: the RAW controller buttons (whether or not the ray hits something); `free`: the
    /// controller's ray hits nothing, so its trigger may start a Hold. Once a Hold runs, only the button of the
    /// controller that started it counts, wherever that controller points. `cur_dir`: the corrected and filtered
    /// gaze direction (stage space) if usable; `look`: the raw head-relative look now. `eye_pos`: the gaze ray origin.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        buttons: [bool; 2],
        free: [bool; 2],
        now: Instant,
        hmd_rot: Quat,
        eye_pos: Vec3,
        cur_dir: Option<Vec3>,
        look: Option<(f64, f64)>,
    ) -> RefineOut {
        let hmd_pos = eye_pos;
        let rising_ctrl = (0..2).find(|&i| buttons[i] && !self.prev_buttons[i] && free[i]);
        self.prev_buttons = buttons;
        let state = std::mem::take(&mut self.state);
        let owner_down = |h: &Hold| buttons[h.ctrl];
        match state {
            RState::Idle => {
                if let (Some(ctrl), Some(anchor)) = (rising_ctrl, cur_dir) {
                    let h = Hold {
                        t0: now,
                        anchor,
                        rot0: hmd_rot,
                        pos0: hmd_pos,
                        look,
                        moved: false,
                        ctrl,
                        dir: anchor,
                        rots: VecDeque::from([(now, hmd_rot)]),
                        lesson_given: false,
                    };
                    self.state = RState::Hold(Box::new(h));
                    return RefineOut {
                        dir: Some(anchor),
                        ..RefineOut::default()
                    };
                }
                RefineOut::default()
            }
            RState::Hold(mut h) => {
                h.rots.push_back((now, hmd_rot));
                while h
                    .rots
                    .front()
                    .is_some_and(|(t, _)| now.duration_since(*t) > REFINE_PRESS_AFTER)
                {
                    h.rots.pop_front();
                }
                let dir = Self::head_pointer(&mut h, hmd_rot);
                let lesson = |h: &Hold| RefineLesson {
                    look: h.look,
                    hmd_rot0: h.rot0,
                    hmd_pos0: h.pos0,
                };
                if !owner_down(&h) {
                    // release before a press: tap = at the anchor, else the refined spot
                    let tap = now.duration_since(h.t0) < REFINE_TAP && !h.moved;
                    let at = if tap { h.anchor } else { dir };
                    let out = RefineOut {
                        dir: Some(at),
                        click: true,
                        lesson: (!tap && angle_deg(h.anchor, at) >= REFINE_LESSON_DEG).then(|| lesson(&h)),
                    };
                    self.state = RState::Clicking {
                        until: now + REFINE_CLICK,
                        dir: at,
                    };
                    return out;
                }
                let held = now.duration_since(h.t0) >= REFINE_PRESS_AFTER;
                let still = h.rots.front().is_some_and(|(t, _)| {
                    now.duration_since(*t) + Duration::from_millis(60) >= REFINE_PRESS_AFTER
                }) && h
                    .rots
                    .iter()
                    .all(|(_, q)| q.angle_between(hmd_rot).to_degrees() < REFINE_STILL_DEG);
                if held && still {
                    let out = RefineOut {
                        dir: Some(dir),
                        click: true,
                        lesson: (angle_deg(h.anchor, dir) >= REFINE_LESSON_DEG && !h.lesson_given)
                            .then(|| lesson(&h)),
                    };
                    h.lesson_given = true;
                    self.state = RState::Pressed(h);
                    return out;
                }
                self.state = RState::Hold(h);
                RefineOut {
                    dir: Some(dir),
                    ..RefineOut::default()
                }
            }
            RState::Pressed(mut h) => {
                if !owner_down(&h) {
                    // released: the press ends this frame
                    return RefineOut::default();
                }
                let dir = Self::head_pointer(&mut h, hmd_rot);
                self.state = RState::Pressed(h);
                RefineOut {
                    dir: Some(dir),
                    click: true,
                    lesson: None,
                }
            }
            RState::Clicking { until, dir } => {
                if now < until {
                    self.state = RState::Clicking { until, dir };
                    RefineOut {
                        dir: Some(dir),
                        click: true,
                        lesson: None,
                    }
                } else {
                    RefineOut::default()
                }
            }
        }
    }
}

/// How long `wayvrctl gaze-check` waits for a steady look.
const CHECK_TIMEOUT: Duration = Duration::from_secs(8);
/// The corrected gaze must come within this many degrees of the target before the settle time starts.
const CHECK_NEAR_DEG: f32 = 6.0;
/// Settle time after the first near sample.
const CHECK_SETTLE: Duration = Duration::from_millis(450);
/// The steady window and its largest raw spread.
const CHECK_WINDOW: Duration = Duration::from_millis(600);
const CHECK_SPREAD_DEG: f32 = 1.0;
/// Distance of the check target straight ahead of the head.
const CHECK_DIST: f32 = 1.5;

/// One frame's gaze, as the check sees it.
#[derive(Debug, Clone, Copy)]
pub struct CheckSample {
    pub t: Instant,
    /// Raw head-relative angles (yaw, pitch degrees).
    pub raw: (f64, f64),
    /// Raw gaze direction, stage space.
    pub raw_dir: Vec3,
    /// Raw + learned correction, stage space.
    pub corr_dir: Vec3,
    pub hmd_rot: Quat,
    pub hmd_pos: Vec3,
}

/// A finished check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CheckOutcome {
    /// `look`: median raw head-relative look; `truth`: median head-relative direction of the target.
    Done {
        look: (f64, f64),
        truth: (f64, f64),
    },
    Failed(&'static str),
}

#[derive(Debug, Clone)]
struct CheckRun {
    target: Vec3,
    started: Instant,
    near_since: Option<Instant>,
    window: Vec<CheckSample>,
    saw_sample: bool,
    settled_once: bool,
}

/// `wayvrctl gaze-check`: a world-fixed target 1.5 m ahead; a steady look at it becomes a lesson.
#[derive(Debug, Default)]
pub struct GazeCheck {
    run: Option<CheckRun>,
}

impl GazeCheck {
    pub const fn active(&self) -> bool {
        self.run.is_some()
    }

    /// The world position of the target while a check runs.
    pub fn target(&self) -> Option<Vec3> {
        self.run.as_ref().map(|r| r.target)
    }

    pub fn cancel(&mut self) {
        self.run = None;
    }

    /// Start a check: the target is `CHECK_DIST` straight ahead of the head now.
    pub fn start(&mut self, now: Instant, hmd_rot: Quat, hmd_pos: Vec3) {
        self.run = Some(CheckRun {
            target: hmd_pos + hmd_rot * (Vec3::NEG_Z * CHECK_DIST),
            started: now,
            near_since: None,
            window: Vec::new(),
            saw_sample: false,
            settled_once: false,
        });
    }

    /// One frame; `sample` is `None` while the gaze is invalid (blinks). Returns the outcome once it is known.
    pub fn step(&mut self, now: Instant, sample: Option<CheckSample>) -> Option<CheckOutcome> {
        let run = self.run.as_mut()?;
        if let Some(s) = sample {
            run.saw_sample = true;
            let to_target = run.target - s.hmd_pos;
            if angle_deg(s.corr_dir, to_target) > CHECK_NEAR_DEG {
                run.near_since = None;
                run.window.clear();
            } else {
                let since = *run.near_since.get_or_insert(s.t);
                if s.t.duration_since(since) >= CHECK_SETTLE {
                    run.window.push(s);
                    let first = run.window[0].t;
                    if s.t.duration_since(first) >= CHECK_WINDOW {
                        run.settled_once = true;
                        let mean = run
                            .window
                            .iter()
                            .fold(Vec3::ZERO, |a, w| a + w.raw_dir.normalize_or_zero())
                            .normalize_or_zero();
                        let spread = run
                            .window
                            .iter()
                            .map(|w| angle_deg(w.raw_dir, mean))
                            .fold(0.0, f32::max);
                        if spread <= CHECK_SPREAD_DEG {
                            let mut ly: Vec<f64> = run.window.iter().map(|w| w.raw.0).collect();
                            let mut lp: Vec<f64> = run.window.iter().map(|w| w.raw.1).collect();
                            let truths: Vec<(f64, f64)> = run
                                .window
                                .iter()
                                .map(|w| {
                                    dir_to_angles(w.hmd_rot.inverse() * (run.target - w.hmd_pos))
                                })
                                .collect();
                            let mut ty: Vec<f64> = truths.iter().map(|t| t.0).collect();
                            let mut tp: Vec<f64> = truths.iter().map(|t| t.1).collect();
                            let out = CheckOutcome::Done {
                                look: (median(&mut ly), median(&mut lp)),
                                truth: (median(&mut ty), median(&mut tp)),
                            };
                            self.run = None;
                            return Some(out);
                        }
                        // not steady: settle again
                        run.near_since = Some(s.t);
                        run.window.clear();
                    }
                }
            }
        }
        if now.duration_since(run.started) > CHECK_TIMEOUT {
            let why = if !run.saw_sample {
                "no gaze"
            } else if run.settled_once {
                "not steady"
            } else {
                "timed out"
            };
            self.run = None;
            return Some(CheckOutcome::Failed(why));
        }
        None
    }
}

/// The answer to `wayvrctl gaze-check`.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckAnswer {
    pub ok: bool,
    pub error_before_deg: Option<f32>,
    pub error_after_deg: Option<f32>,
    pub reason: Option<String>,
}

impl CheckAnswer {
    pub fn failed(reason: &str) -> Self {
        Self {
            ok: false,
            error_before_deg: None,
            error_after_deg: None,
            reason: Some(reason.to_string()),
        }
    }
}

/// Mailbox between the IPC server (another call site, `&InputState` only) and the frame loop.
#[derive(Debug, Default)]
pub struct Shared {
    /// `wayvrctl gaze-check` asked for a check.
    pub check_request: bool,
    /// The finished check, taken by the waiting connection.
    pub check_result: Option<CheckAnswer>,
    /// `wayvrctl gaze-learn forget`.
    pub forget: bool,
}

/// `wayvrctl gaze-learn status`.
#[derive(Debug, Clone, PartialEq)]
pub struct LearnStatus {
    pub enabled: bool,
    pub lessons: u32,
    pub taken: u32,
    pub refused: u32,
    pub last_error_deg: Option<f32>,
    pub offset_deg: [f32; 2],
    pub key: String,
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
    /// HMD pose (stage space), set every frame before `feed`: gaze samples are made head-relative with it.
    pub hmd: Affine3A,
    /// Gaze click v2: apply and learn the correction (mirrors `gaze_learn`).
    pub learn: bool,
    /// The learned correction and its file.
    pub store: LessonStore,
    /// Origin of the gaze ray (eye pose translation, stage space) of the last accepted sample.
    pub eye_pos: Vec3,
    /// Hold-to-refine click state (only driven through `gate::refine`).
    pub refine: Refine,
    /// `wayvrctl gaze-check`.
    pub check: GazeCheck,
    /// Mailbox for the IPC server.
    pub shared: Arc<Mutex<Shared>>,
    /// Raw head-relative samples of the last [`HISTORY`]: (time, yaw, pitch).
    history: VecDeque<(Instant, f64, f64)>,
    /// This frame's sample for the check (cleared when the gaze is invalid).
    sample: Option<CheckSample>,
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
            hmd: Affine3A::IDENTITY,
            learn: false,
            // tests never touch the user's lessons file
            store: if cfg!(test) {
                LessonStore::in_memory()
            } else {
                LessonStore::load(crate::backend::gaze_lessons::default_path(), unix_secs())
            },
            eye_pos: Vec3::ZERO,
            refine: Refine::default(),
            check: GazeCheck::default(),
            shared: Arc::new(Mutex::new(Shared::default())),
            history: VecDeque::new(),
            sample: None,
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
        self.history.clear();
        self.sample = None;
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
        self.sample = None;
        self.feed_inner(raw, smoothing, deadzone_deg, dwell_ms, now);
        self.step_check(now);
    }

    fn feed_inner(
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

        // gaze click v2: raw direction -> head-relative angles -> + learned correction -> back to stage space
        let (_, hmd_rot, _) = self.hmd.to_scale_rotation_translation();
        self.eye_pos = pos;
        let (ry, rp) = dir_to_angles(hmd_rot.inverse() * raw_dir);
        self.history.push_back((now, ry, rp));
        while self
            .history
            .front()
            .is_some_and(|(t, ..)| now.duration_since(*t) > HISTORY)
        {
            self.history.pop_front();
        }
        let corr_dir = if self.learn && !self.store.correction.is_empty() {
            let (cy, cp) = self.store.correction.get(ry, rp);
            let d = hmd_rot * angles_to_dir(ry + cy, rp + cp);
            if d.is_finite() { d } else { raw_dir }
        } else {
            raw_dir
        };
        self.sample = Some(CheckSample {
            t: now,
            raw: (ry, rp),
            raw_dir,
            corr_dir,
            hmd_rot,
            // the truth is measured from the gaze ray origin (the eye), not the HMD origin
            hmd_pos: pos,
        });

        let smoothed = match smoothing_params(smoothing) {
            Some((min_cutoff, beta)) => self
                .filter
                .filter(corr_dir, dt, min_cutoff, beta)
                .normalize_or_zero(),
            None => corr_dir,
        };
        let smoothed = if smoothed == Vec3::ZERO {
            corr_dir
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

impl GazeState {
    /// The filtered gaze direction (stage space), if usable.
    pub fn current_dir(&self) -> Option<Vec3> {
        self.out_dir.filter(|_| self.active)
    }

    /// The raw head-relative look at `now`: per-axis median of the raw samples of the last [`LOOK`].
    pub fn look(&self, now: Instant) -> Option<(f64, f64)> {
        let (mut y, mut p): (Vec<f64>, Vec<f64>) = self
            .history
            .iter()
            .filter(|(t, ..)| *t <= now && now.duration_since(*t) <= LOOK)
            .map(|(_, y, p)| (*y, *p))
            .unzip();
        (!y.is_empty()).then(|| (median(&mut y), median(&mut p)))
    }

    /// The current learned correction at a raw head-relative look.
    fn correction_at(&self, look: (f64, f64)) -> (f64, f64) {
        self.store.correction.get(look.0, look.1)
    }

    /// Apply the settings that decide whether lessons are used: call every frame with the config.
    pub fn sync_config(&mut self, learn: bool, key: &str, hmd: Affine3A) {
        self.hmd = hmd;
        self.learn = learn;
        if learn {
            self.store.check_key(key);
        }
    }

    /// Head-relative angles of a world point as seen from the HMD pose `rot`/`pos`.
    fn truth_angles(rot: Quat, pos: Vec3, point: Vec3) -> (f64, f64) {
        dir_to_angles(rot.inverse() * (point - pos))
    }

    /// A "refine" lesson from a refined gaze click that landed on `hit` (stage space).
    pub fn record_refine(&mut self, lesson: &RefineLesson, hit: Vec3) {
        if !self.learn {
            return;
        }
        let Some(look) = lesson.look else {
            return;
        };
        let truth = Self::truth_angles(lesson.hmd_rot0, lesson.hmd_pos0, hit);
        let (dy, dp) = (truth.0 - look.0, truth.1 - look.1);
        let (cy, cp) = self.correction_at(look);
        let left = (dy - cy).hypot(dp - cp);
        if !left.is_finite() || left > REFINE_MAX_DEG {
            self.store.refuse();
            return;
        }
        self.store.add(
            Lesson {
                time: unix_secs(),
                hy: look.0,
                hp: look.1,
                dy,
                dp,
                wy: 1.0,
                wp: 1.0,
                how: "refine".into(),
            },
            left,
        );
    }

    /// A passive "laser" lesson: a controller laser clicked `hit` (stage space) on a screen while the gaze was fresh.
    /// Taken only if the corrected gaze was within [`LASER_MAX_DEG`] of it; never produces any input.
    pub fn record_laser(&mut self, hit: Vec3, now: Instant) {
        if !self.learn || !self.active || !self.fresh {
            return;
        }
        let Some(look) = self.look(now) else {
            return;
        };
        let (_, rot, _) = self.hmd.to_scale_rotation_translation();
        let truth = Self::truth_angles(rot, self.eye_pos, hit);
        let (cy, cp) = self.correction_at(look);
        let err = angles_err((look.0 + cy, look.1 + cp), truth);
        if !err.is_finite() || err > LASER_MAX_DEG {
            return;
        }
        self.store.add(
            Lesson {
                time: unix_secs(),
                hy: look.0,
                hp: look.1,
                dy: truth.0 - look.0,
                dp: truth.1 - look.1,
                wy: 0.5,
                wp: 0.5,
                how: "laser".into(),
            },
            err,
        );
    }

    /// Finish a gaze check: record the lesson and answer.
    fn finish_check(&mut self, out: CheckOutcome) -> CheckAnswer {
        match out {
            CheckOutcome::Failed(why) => CheckAnswer::failed(why),
            CheckOutcome::Done { look, truth } => {
                let (cy, cp) = self.correction_at(look);
                let before = angles_err((look.0 + cy, look.1 + cp), truth);
                let (dy, dp) = (truth.0 - look.0, truth.1 - look.1);
                if !before.is_finite() || (dy - cy).hypot(dp - cp) > CHECK_MAX_DEG {
                    self.store.refuse();
                    return CheckAnswer::failed("not steady");
                }
                self.store.add(
                    Lesson {
                        time: unix_secs(),
                        hy: look.0,
                        hp: look.1,
                        dy,
                        dp,
                        wy: 1.0,
                        wp: 1.0,
                        how: "check".into(),
                    },
                    before,
                );
                let (cy, cp) = self.correction_at(look);
                let after = angles_err((look.0 + cy, look.1 + cp), truth);
                CheckAnswer {
                    ok: true,
                    error_before_deg: Some(before as f32),
                    error_after_deg: Some(after as f32),
                    reason: None,
                }
            }
        }
    }

    fn put_check_result(&mut self, a: CheckAnswer) {
        if let Ok(mut sh) = self.shared.lock() {
            sh.check_result = Some(a);
        }
    }

    fn step_check(&mut self, now: Instant) {
        if !self.check.active() {
            return;
        }
        let sample = self.sample.filter(|_| self.fresh);
        if let Some(out) = self.check.step(now, sample) {
            let a = self.finish_check(out);
            self.put_check_result(a);
        }
    }

    /// Frame loop: serve what the IPC server left in the mailbox (forget, a new check). `tracking`: any of
    /// `gaze_laser` / `gaze_target` / `gaze_pointer` is on.
    pub fn poll_shared(&mut self, tracking: bool, now: Instant) {
        let (forget, request) = match self.shared.lock() {
            Ok(mut sh) => (
                std::mem::take(&mut sh.forget),
                std::mem::take(&mut sh.check_request),
            ),
            Err(_) => return,
        };
        if forget {
            self.store.forget();
        }
        if request {
            if !self.learn {
                self.put_check_result(CheckAnswer::failed("learning off"));
            } else if !tracking || !self.active {
                self.put_check_result(CheckAnswer::failed("no gaze"));
            } else {
                let (_, rot, pos) = self.hmd.to_scale_rotation_translation();
                self.check.start(now, rot, pos);
            }
        }
        if !tracking && self.check.active() {
            self.check.cancel();
            self.put_check_result(CheckAnswer::failed("no gaze"));
        }
    }

    /// `wayvrctl gaze-learn status`.
    pub fn learn_status(&self) -> LearnStatus {
        let (oy, op) = self.store.correction.offset();
        LearnStatus {
            enabled: self.learn,
            lessons: self.store.correction.len() as u32,
            taken: self.store.taken,
            refused: self.store.refused,
            last_error_deg: self.store.last_error_deg.map(|v| v as f32),
            offset_deg: [oy as f32, op as f32],
            key: self.store.key.clone(),
        }
    }
}

/// The ONE gate for gaze input (omarchyvr, user decision 2026-10-17): "by default the gaze is only a visual point,
/// never a mouse". Every path where the eyes could hover, click, focus, move the cursor or report a target for voice
/// asks this gate, and the gate is `gaze_pointer` (omarchyvr writes it from eye.conf `gaze_mouse`, default off; the
/// app switch "Eyes as a mouse"). Off: the gaze ray only places the visual dot (`gaze_laser`).
pub mod gate {
    use wlx_common::config::HandsfreePointer;

    /// Gaze input is allowed (the eyes may act as a mouse).
    pub const fn input_on(gaze_pointer: bool) -> bool {
        gaze_pointer
    }

    /// The gaze may hover / click through a free controller slot this frame.
    pub const fn may_point(gaze_pointer: bool, active: bool, picking_focus: bool, any_handsfree: bool) -> bool {
        input_on(gaze_pointer) && active && !picking_focus && !any_handsfree
    }

    /// Gaze has priority over the controller rays (`gaze_over_controllers`), only while gaze input is on.
    pub const fn over_controllers(gaze_pointer: bool, over: bool) -> bool {
        input_on(gaze_pointer) && over
    }

    /// The dwell-click time actually used: 0 (never clicks) while gaze input is off.
    pub const fn dwell_ms(gaze_pointer: bool, dwell_ms: u32) -> u32 {
        if input_on(gaze_pointer) { dwell_ms } else { 0 }
    }

    /// Hold-to-refine gaze click: only effective through the gaze pointer gate.
    pub const fn refine(gaze_pointer: bool, gaze_refine: bool) -> bool {
        input_on(gaze_pointer) && gaze_refine
    }

    /// Whether the gaze pointer's button is down this frame. The ONLY place a gaze click is decided: with
    /// `gaze_pointer` off it is `false` for every input; the learning (`gaze_learn`, laser lessons, `gaze-check`)
    /// has no input here at all.
    #[allow(clippy::fn_params_excessive_bools)]
    pub const fn pointer_click(
        gaze_pointer: bool,
        gaze_refine: bool,
        trigger: bool,
        dwell: bool,
        refine_click: bool,
    ) -> bool {
        input_on(gaze_pointer) && if refine(gaze_pointer, gaze_refine) { refine_click || dwell } else { trigger || dwell }
    }

    /// `wayvrctl gaze` answers with targets (what voice "this" reads) only while gaze input is on.
    pub const fn reporting(gaze_pointer: bool) -> bool {
        input_on(gaze_pointer)
    }

    /// WayVR's own handsfree modes (dashboard, keyboard bar, `wayvrctl handsfree`) can make the eyes the pointer
    /// (`EyeTracking`, `EyeTrackingOnly`): with gaze input off they fall back to the head pointer of the same kind.
    pub const fn handsfree(mode: HandsfreePointer, gaze_pointer: bool) -> HandsfreePointer {
        match mode {
            HandsfreePointer::EyeTracking if !input_on(gaze_pointer) => HandsfreePointer::Hmd,
            HandsfreePointer::EyeTrackingOnly if !input_on(gaze_pointer) => HandsfreePointer::HmdOnly,
            m => m,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::backend::gaze::{Dwell, GazeState};
        use glam::{Affine3A, Quat, Vec3};
        use crate::windowing::OverlayID;
        use glam::Vec3A;
        use std::time::{Duration, Instant};

        /// With the switch off, no gaze input path can produce a pointer, click, focus or target event, whatever
        /// else is configured (an old 20-gaze.yaml with dwell and gaze_over_controllers, a dashboard eye mode).
        #[test]
        fn switch_off_no_gaze_path_can_point_click_or_focus() {
            for active in [false, true] {
                for picking in [false, true] {
                    for hf in [false, true] {
                        assert!(!may_point(false, active, picking, hf));
                    }
                }
            }
            assert!(!over_controllers(false, true), "controllers keep hovering: a trigger never clicks where you look");
            assert_eq!(dwell_ms(false, 600), 0);
            assert!(!reporting(false), "wayvrctl gaze has no target for voice");
            for m in [HandsfreePointer::EyeTracking, HandsfreePointer::EyeTrackingOnly] {
                let e = handsfree(m, false);
                assert!(!matches!(e, HandsfreePointer::EyeTracking | HandsfreePointer::EyeTrackingOnly), "{m:?} -> {e:?}");
            }
            // the dwell timer fed with the gated time never clicks, however long the eyes rest on one spot
            let mut d = Dwell::default();
            let t0 = Instant::now();
            let hit = Some((OverlayID::default(), Vec3A::new(0.0, 0.0, -1.0)));
            for ms in (0..20_000).step_by(11) {
                assert!(!d.update(hit, t0 + Duration::from_millis(ms), dwell_ms(false, 600)));
            }
        }

        /// Gaze click v2: with the switch off, no combination of learn / refine / laser clicks / gaze-check can make
        /// the gaze pointer click. The learning has no input to `pointer_click` at all; this also drives the real
        /// learning paths and checks the pointer state stays unclicked.
        #[test]
        fn switch_off_learning_never_clicks() {
            for refine_cfg in [false, true] {
                for trigger in [false, true] {
                    for dwell in [false, true] {
                        for refine_click in [false, true] {
                            assert!(!pointer_click(false, refine_cfg, trigger, dwell, refine_click));
                        }
                    }
                }
                assert!(!refine(false, refine_cfg), "gaze_refine is only effective through gaze_pointer");
            }
            // switch on: the old behaviour without refine, the refine state with it
            assert!(pointer_click(true, false, true, false, false));
            assert!(pointer_click(true, false, false, true, false));
            assert!(!pointer_click(true, true, true, false, false), "refine: the trigger alone presses nothing");
            assert!(pointer_click(true, true, false, true, false), "refine idle: dwell still works");
            assert!(pointer_click(true, true, false, false, true));

            let mut g = GazeState::new();
            let t0 = Instant::now();
            g.sync_config(true, "k", Affine3A::IDENTITY);
            let p = Affine3A::from_rotation_translation(Quat::from_rotation_y(0.05), Vec3::new(0.0, 1.6, 0.0));
            for i in 0..80u64 {
                let q = Affine3A::from_rotation_translation(
                    Quat::from_rotation_y(0.05 + (i % 3) as f32 * 0.0001),
                    Vec3::new(0.0, 1.6, 0.0),
                );
                g.feed(Some(if i % 2 == 0 { p } else { q }), 0.5, 0.6, 0, t0 + Duration::from_millis(11 * i));
                // a laser lesson, a check request: nothing may click
                g.record_laser(Vec3::new(0.0, 1.6, -2.0), t0 + Duration::from_millis(11 * i));
                if i == 10 {
                    g.shared.lock().unwrap().check_request = true;
                }
                g.poll_shared(true, t0 + Duration::from_millis(11 * i));
                assert!(!g.ptr.now.click && !g.ptr.before.click);
            }
            // with the switch off the refine logic is never driven: it stays idle
            assert!(!g.refine.active());
        }

        #[test]
        fn switch_on_is_the_only_way_in() {
            assert!(may_point(true, true, false, false));
            assert!(!may_point(true, false, false, false), "no valid gaze, no pointer");
            assert!(over_controllers(true, true) && !over_controllers(true, false));
            assert_eq!(dwell_ms(true, 600), 600);
            assert!(reporting(true));
            assert!(matches!(handsfree(HandsfreePointer::EyeTracking, true), HandsfreePointer::EyeTracking));
            assert!(matches!(handsfree(HandsfreePointer::Hmd, false), HandsfreePointer::Hmd));
        }

        /// WayVR's own default is off too (an empty config).
        #[test]
        fn wayvr_default_config_has_no_gaze_input() {
            let c: wlx_common::config::GeneralConfig = serde_json::from_str("{}").unwrap();
            assert!(!input_on(c.gaze_pointer) && !c.gaze_over_controllers && c.gaze_dwell_ms == 0);
            assert!(!c.gaze_learn && !c.gaze_refine && c.gaze_learn_key.is_empty());
            assert!(c.click_steady_px == 0.0, "steady clicks are off unless the app switches them on");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::gaze_lessons::Lesson;

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

    #[test]
    fn angles_round_trip() {
        for (y, p) in [(0.0, 0.0), (10.0, -5.0), (-30.0, 20.0), (3.3, 44.0)] {
            let (y2, p2) = dir_to_angles(angles_to_dir(y, p));
            assert!((y - y2).abs() < 1e-4 && (p - p2).abs() < 1e-4, "{y},{p} -> {y2},{p2}");
        }
        // yaw is positive to the left (-X), pitch positive up
        assert!(dir_to_angles(Vec3::new(-0.2, 0.0, -1.0)).0 > 0.0);
        assert!(dir_to_angles(Vec3::new(0.0, 0.2, -1.0)).1 > 0.0);
    }

    fn head(deg_yaw: f32) -> Quat {
        Quat::from_rotation_y(deg_yaw.to_radians())
    }

    /// Run the refine machine at 90 Hz from `t0` for `ms`, the head following `yaw(t_ms)`; returns each frame's out.
    fn run_refine(
        r: &mut Refine,
        t0: Instant,
        from_ms: u64,
        to_ms: u64,
        trigger: impl Fn(u64) -> bool,
        yaw: impl Fn(u64) -> f32,
    ) -> Vec<(u64, RefineOut)> {
        run_refine_free(r, t0, from_ms, to_ms, trigger, |_| true, yaw)
    }

    /// Like `run_refine`, with the "controller 0's ray hits nothing" flag per frame.
    fn run_refine_free(
        r: &mut Refine,
        t0: Instant,
        from_ms: u64,
        to_ms: u64,
        trigger: impl Fn(u64) -> bool,
        free: impl Fn(u64) -> bool,
        yaw: impl Fn(u64) -> f32,
    ) -> Vec<(u64, RefineOut)> {
        let mut v = Vec::new();
        let mut ms = from_ms;
        while ms <= to_ms {
            let out = r.update(
                [trigger(ms), false],
                [free(ms), true],
                t0 + Duration::from_millis(ms),
                head(yaw(ms)),
                Vec3::ZERO,
                Some(Vec3::NEG_Z),
                Some((0.0, 0.0)),
            );
            v.push((ms, out));
            ms += 11;
        }
        v
    }

    #[test]
    fn refine_tap_clicks_at_the_anchor_once() {
        let mut r = Refine::default();
        let t0 = Instant::now();
        let outs = run_refine(&mut r, t0, 0, 400, |ms| ms < 150, |_| 0.0);
        let clicks: Vec<_> = outs.iter().filter(|(_, o)| o.click).collect();
        assert!(!clicks.is_empty());
        // nothing pressed while held, the dot frozen at the anchor
        assert!(outs.iter().filter(|(ms, _)| *ms < 150).all(|(_, o)| !o.click && o.dir == Some(Vec3::NEG_Z)));
        // ~60 ms, at the anchor, no lesson for a tap
        let span = clicks.last().unwrap().0 - clicks[0].0;
        assert!((40..=75).contains(&span), "{span}");
        assert!(clicks.iter().all(|(_, o)| o.dir == Some(Vec3::NEG_Z) && o.lesson.is_none()));
        assert!(!r.active());
    }

    #[test]
    fn refine_release_after_head_movement_clicks_refined_and_records_a_lesson() {
        let mut r = Refine::default();
        let t0 = Instant::now();
        // trigger held 400 ms, head turns 3 deg left meanwhile
        let outs = run_refine(&mut r, t0, 0, 700, |ms| ms < 400, |ms| (ms.min(300) as f32) / 100.0);
        let click: Vec<_> = outs.iter().filter(|(_, o)| o.click).collect();
        assert!(!click.is_empty());
        let first = click[0].1;
        let d = first.dir.unwrap();
        let moved = angle(d, Vec3::NEG_Z);
        // gain 1 with the 0.5 deg dead zone
        assert!((moved - 2.5).abs() < 0.2, "{moved}");
        assert!(first.lesson.is_some(), "a refined click records a lesson");
        assert_eq!(outs.iter().filter(|(_, o)| o.lesson.is_some()).count(), 1);
    }

    #[test]
    fn refine_still_hold_presses_drags_and_releases() {
        let mut r = Refine::default();
        let t0 = Instant::now();
        // trigger held 1.5 s; the head is still the first 0.8 s, then turns 2 deg
        let outs = run_refine(&mut r, t0, 0, 1800, |ms| ms < 1500, |ms| if ms < 800 { 0.0 } else { (ms - 800).min(200) as f32 / 100.0 });
        let first_press = outs.iter().find(|(_, o)| o.click).unwrap();
        assert!((500..=620).contains(&first_press.0), "press at {}", first_press.0);
        // pressed until the release at 1500
        assert!(outs.iter().filter(|(ms, _)| *ms > first_press.0 && *ms < 1500).all(|(_, o)| o.click));
        // dragged by the head
        let last = outs.iter().rev().find(|(ms, _)| *ms < 1500).unwrap();
        assert!(angle(last.1.dir.unwrap(), Vec3::NEG_Z) > 1.2);
        assert!(outs.iter().filter(|(ms, _)| *ms >= 1500).all(|(_, o)| !o.click));
        assert!(!r.active());
    }

    #[test]
    fn refine_ray_sweeping_onto_an_overlay_is_not_a_release() {
        let mut r = Refine::default();
        let t0 = Instant::now();
        // the controller ray hits an overlay from 100 ms on, the button stays down until 1500 ms
        let outs = run_refine_free(&mut r, t0, 0, 1800, |ms| ms < 1500, |ms| ms < 100, |_| 0.0);
        let first = outs.iter().find(|(_, o)| o.click).unwrap();
        assert!(first.0 >= 500, "no click before the 0.5 s press, got {}", first.0);
        assert!(outs.iter().filter(|(ms, _)| *ms > first.0 && *ms < 1500).all(|(_, o)| o.click));
        assert!(outs.iter().filter(|(ms, _)| *ms >= 1500).all(|(_, o)| !o.click));
    }

    #[test]
    fn refine_trigger_held_across_a_gaze_drop_makes_no_tap() {
        let mut r = Refine::default();
        let t0 = Instant::now();
        // the gaze drops while the trigger is held: observe() keeps the button memory
        r.observe([true, false]);
        let outs = run_refine(&mut r, t0, 0, 600, |ms| ms < 300, |_| 0.0);
        assert!(outs.iter().all(|(_, o)| !o.click && o.dir.is_none()), "no Hold, no tap");
        assert!(!r.active());
        // a fresh press after the button was low arms normally
        let outs = run_refine(&mut r, t0, 700, 1000, |ms| ms < 800, |_| 0.0);
        assert!(outs.iter().any(|(_, o)| o.click));
    }

    #[test]
    fn refine_head_jitter_below_one_degree_teaches_nothing() {
        let mut r = Refine::default();
        let t0 = Instant::now();
        // 0.7 deg of head movement: the click is refined (not a tap) but too small to learn from
        let outs = run_refine(&mut r, t0, 0, 700, |ms| ms < 400, |ms| (ms.min(300) as f32) / 100.0 * 0.7 / 3.0);
        assert!(outs.iter().any(|(_, o)| o.click));
        assert!(outs.iter().all(|(_, o)| o.lesson.is_none()));
    }

    #[test]
    fn lessons_measure_from_the_eye_not_the_head() {
        let mut g = GazeState::new();
        g.sync_config(true, "k", Affine3A::IDENTITY);
        // the eye is 6 cm right of the head origin; the click landed straight ahead of the eye
        let lesson = RefineLesson { look: Some((0.0, 0.0)), hmd_rot0: Quat::IDENTITY, hmd_pos0: Vec3::new(0.06, 0.0, 0.0) };
        g.record_refine(&lesson, Vec3::new(0.06, 0.0, -2.0));
        assert!(g.store.last_error_deg.unwrap().abs() < 1e-6);
    }

    #[test]
    fn refine_moving_head_does_not_press() {
        let mut r = Refine::default();
        let t0 = Instant::now();
        let outs = run_refine(&mut r, t0, 0, 1000, |ms| ms < 2000, |ms| ms as f32 / 200.0);
        assert!(outs.iter().all(|(_, o)| !o.click));
    }

    fn sample(t: Instant, raw: (f64, f64), hmd_rot: Quat, hmd_pos: Vec3, err: Vec3) -> CheckSample {
        let raw_dir = hmd_rot * angles_to_dir(raw.0, raw.1);
        CheckSample { t, raw, raw_dir, corr_dir: raw_dir + err, hmd_rot, hmd_pos }
    }

    #[test]
    fn gaze_check_steady_look_gives_a_lesson_and_a_wobbly_one_fails() {
        let t0 = Instant::now();
        let pos = Vec3::new(0.0, 1.6, 0.0);
        let mut c = GazeCheck::default();
        c.start(t0, Quat::IDENTITY, pos);
        assert!((c.target().unwrap() - Vec3::new(0.0, 1.6, -1.5)).length() < 1e-5);
        // looking 2 deg to the right of the target, steadily
        let raw = (-2.0, 0.0);
        let mut out = None;
        let mut ms = 0;
        while ms < 3000 && out.is_none() {
            out = c.step(t0 + Duration::from_millis(ms), Some(sample(t0 + Duration::from_millis(ms), raw, Quat::IDENTITY, pos, Vec3::ZERO)));
            ms += 11;
        }
        let Some(CheckOutcome::Done { look, truth }) = out else { panic!("{out:?}") };
        assert!(ms > 1000 && ms < 1200, "settle 0.45 + window 0.6, got {ms}");
        assert_eq!(look, raw);
        assert!(truth.0.abs() < 1e-3 && truth.1.abs() < 1e-3);

        // wobbling 3 deg: never steady
        let mut c = GazeCheck::default();
        c.start(t0, Quat::IDENTITY, pos);
        let mut out = None;
        let mut ms = 0u64;
        while ms < 9000 && out.is_none() {
            let w = if (ms / 50).is_multiple_of(2) { 2.0 } else { -2.0 };
            out = c.step(t0 + Duration::from_millis(ms), Some(sample(t0 + Duration::from_millis(ms), (w, 0.0), Quat::IDENTITY, pos, Vec3::ZERO)));
            ms += 11;
        }
        assert_eq!(out, Some(CheckOutcome::Failed("not steady")));

        // no samples at all
        let mut c = GazeCheck::default();
        c.start(t0, Quat::IDENTITY, pos);
        assert_eq!(c.step(t0 + Duration::from_secs(9), None), Some(CheckOutcome::Failed("no gaze")));
        // looking elsewhere
        let mut c = GazeCheck::default();
        c.start(t0, Quat::IDENTITY, pos);
        let s = sample(t0, (40.0, 0.0), Quat::IDENTITY, pos, Vec3::ZERO);
        assert_eq!(c.step(t0, Some(s)), None);
        assert_eq!(c.step(t0 + Duration::from_secs(9), Some(s)), Some(CheckOutcome::Failed("timed out")));
    }

    #[test]
    fn learned_correction_moves_the_gaze_and_forgetting_undoes_it() {
        let mut g = GazeState::new();
        g.sync_config(true, "k", Affine3A::IDENTITY);
        let t0 = Instant::now();
        let p = Affine3A::from_rotation_translation(Quat::IDENTITY, Vec3::new(0.0, 1.6, 0.0));
        // the eyes look straight ahead; the truth was 2 deg to the left
        for i in 0..5 {
            g.store.add(
                Lesson { time: 1.0 + f64::from(i), hy: 0.0, hp: 0.0, dy: 2.0, dp: 0.0, wy: 1.0, wp: 1.0, how: "check".into() },
                0.0,
            );
        }
        g.feed(Some(p), 0.0, 0.0, 0, t0);
        let d = g.ptr.pose.matrix3 * Vec3A::NEG_Z;
        let (yaw, _) = dir_to_angles(Vec3::from(d));
        assert!(yaw > 1.0 && yaw < 2.1, "{yaw}");
        // learning off: the raw gaze again
        g.sync_config(false, "k", Affine3A::IDENTITY);
        g.feed(Some(p), 0.0, 0.0, 0, t0 + Duration::from_millis(11));
        let d = g.ptr.pose.matrix3 * Vec3A::NEG_Z;
        assert!(dir_to_angles(Vec3::from(d)).0.abs() < 1e-3);
        g.sync_config(true, "k", Affine3A::IDENTITY);
        let st = g.learn_status();
        assert!(st.enabled && st.lessons == 5 && st.taken == 5 && st.offset_deg[0] > 0.5);
        g.shared.lock().unwrap().forget = true;
        g.poll_shared(true, t0);
        assert_eq!(g.learn_status().lessons, 0);
    }

    #[test]
    fn refine_lesson_is_refused_past_20_degrees_and_taken_otherwise() {
        let mut g = GazeState::new();
        g.sync_config(true, "k", Affine3A::IDENTITY);
        let lesson = RefineLesson { look: Some((0.0, 0.0)), hmd_rot0: Quat::IDENTITY, hmd_pos0: Vec3::ZERO };
        // a point 3 deg to the left at 2 m
        g.record_refine(&lesson, Vec3::new(-2.0 * 3.0f32.to_radians().tan(), 0.0, -2.0));
        assert_eq!((g.store.taken, g.store.refused), (1, 0));
        assert!((g.store.last_error_deg.unwrap() - 3.0).abs() < 0.05);
        g.record_refine(&lesson, Vec3::new(-2.0, 0.0, -1.0));
        assert_eq!((g.store.taken, g.store.refused), (1, 1));
        // learning off: nothing recorded
        g.sync_config(false, "k", Affine3A::IDENTITY);
        g.record_refine(&lesson, Vec3::new(-0.1, 0.0, -2.0));
        assert_eq!((g.store.taken, g.store.refused), (1, 1));
    }
}
