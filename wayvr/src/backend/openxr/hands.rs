//! Simple hand skeletons from XR_EXT_hand_tracking, drawn with the line pool (flat semi-transparent
//! strokes: one per finger, plus two across the palm).

use std::{cell::Cell, time::Instant};

use glam::Vec3A;
use openxr as xr;

use super::{
    XrState,
    helpers::posef_to_transform,
    lines::{HAND_COLOR, LinePool},
};
use crate::state::AppState;

/// (from, to) joint indices (XR_EXT_hand_tracking order) of each stroke.
const STROKES: [(usize, usize); 7] = [
    (3, 5),   // thumb proximal -> tip
    (7, 10),  // index
    (12, 15), // middle
    (17, 20), // ring
    (22, 25), // little
    (1, 12),  // wrist -> middle proximal (palm length)
    (7, 22),  // index proximal -> little proximal (palm width)
];
const STROKE_COUNT: usize = STROKES.len();

struct Hand {
    /// frames in the current diagnostics window: (located, active, all joints valid)
    seen: Cell<(u32, u32, u32)>,
    tracker: xr::HandTracker,
    lines: [usize; STROKE_COUNT],
}

pub(super) struct HandsVisual {
    hands: Vec<Hand>,
    log_since: Cell<Option<Instant>>,
}

/// Joint positions of a located hand, or `None` if any joint we need is not valid.
fn joint_positions(
    joints: &[xr::HandJointLocation; xr::HAND_JOINT_COUNT],
) -> Option<[Vec3A; STROKE_COUNT * 2]> {
    let mut out = [Vec3A::ZERO; STROKE_COUNT * 2];
    for (i, stroke) in STROKES.iter().enumerate() {
        for (k, j) in [stroke.0, stroke.1].into_iter().enumerate() {
            let loc = &joints[j];
            if !loc
                .location_flags
                .contains(xr::SpaceLocationFlags::POSITION_VALID)
            {
                return None;
            }
            out[i * 2 + k] = posef_to_transform(&loc.pose).translation;
        }
    }
    Some(out)
}

impl HandsVisual {
    /// `None` if the runtime does not offer hand tracking.
    pub(super) fn new(xr: &XrState, lines: &mut LinePool, app: &AppState) -> Option<Self> {
        if xr.instance.exts().ext_hand_tracking.is_none() {
            log::info!("Hand tracking not available, hands will not be shown.");
            return None;
        }
        let mut hands = Vec::with_capacity(2);
        for side in [xr::Hand::LEFT, xr::Hand::RIGHT] {
            let tracker = match xr.session.create_hand_tracker(side) {
                Ok(t) => t,
                Err(e) => {
                    log::warn!("Could not create hand tracker: {e}");
                    return None;
                }
            };
            let mut ids = [0usize; STROKE_COUNT];
            for id in &mut ids {
                *id = lines.allocate(xr, app).ok()?;
            }
            hands.push(Hand {
                seen: Cell::new((0, 0, 0)),
                tracker,
                lines: ids,
            });
        }
        log::info!("Hand tracking enabled, drawing hands.");
        Some(Self {
            hands,
            log_since: Cell::new(None),
        })
    }

    /// Locate joints at the predicted display time and queue the strokes for this frame.
    pub(super) fn update(&self, xr: &XrState, app: &AppState, lines: &mut LinePool) {
        if !app.session.config.show_hands {
            return;
        }
        for hand in &self.hands {
            let (mut n, mut act, mut ok) = hand.seen.get();
            n += 1;
            let Ok(Some(joints)) = xr
                .stage
                .locate_hand_joints(&hand.tracker, xr.predicted_display_time)
            else {
                hand.seen.set((n, act, ok));
                continue; // inactive: hide
            };
            act += 1;
            let Some(p) = joint_positions(&joints) else {
                hand.seen.set((n, act, ok));
                continue; // invalid: hide
            };
            ok += 1;
            hand.seen.set((n, act, ok));
            for (i, &id) in hand.lines.iter().enumerate() {
                lines.draw_between(id, p[i * 2], p[i * 2 + 1], HAND_COLOR, app.input_state.hmd);
            }
        }

        let now = Instant::now();
        let since = self.log_since.get().unwrap_or(now);
        self.log_since.set(Some(since));
        if now.duration_since(since).as_secs() >= 5 {
            for (i, hand) in self.hands.iter().enumerate() {
                let (n, act, ok) = hand.seen.replace((0, 0, 0));
                log::info!(
                    "hand {}: {n} frames, runtime-active {act}, joints valid (drawn) {ok}",
                    if i == 0 { "left" } else { "right" }
                );
            }
            self.log_since.set(Some(now));
        }
    }
}
