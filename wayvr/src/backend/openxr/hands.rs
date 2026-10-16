//! Simple hand skeletons from XR_EXT_hand_tracking, drawn with the line pool (flat opaque strokes: two per
//! finger so a curl shows as a bend, plus two across the palm). Wider and in a stronger colour than the pointer rays:
//! the first version (2 mm, 45 % light blue) was hard to see even where the hands were in view.
//!
//! Only CAMERA hands are drawn (trackers that ask for the unobstructed data source, XR_EXT_hand_tracking_data_source):
//! a runtime that makes up hand joints from a controller (Monado's Pimax driver: trigger/grip -> curls, data source
//! CONTROLLER) would otherwise draw an amber hand over every controller model. Without the data-source extension the
//! two cannot be told apart, so no hands are drawn.

use std::{cell::Cell, time::Instant};

use glam::Vec3A;
use openxr as xr;

use super::{
    XrState,
    helpers::posef_to_transform,
    lines::{HAND_COLOR, HAND_LINE_WIDTH, LinePool},
};
use crate::state::AppState;

/// (from, to) joint indices (XR_EXT_hand_tracking order) of each stroke.
const STROKES: [(usize, usize); 12] = [
    (3, 4),   // thumb proximal -> distal
    (4, 5),   // thumb distal -> tip
    (7, 8),   // index proximal -> intermediate
    (8, 10),  // index intermediate -> tip
    (12, 13), // middle
    (13, 15),
    (17, 18), // ring
    (18, 20),
    (22, 23), // little
    (23, 25),
    (1, 12), // wrist -> middle proximal (palm length)
    (7, 22), // index proximal -> little proximal (palm width)
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
        if xr.instance.exts().ext_hand_tracking_data_source.is_none() {
            log::info!(
                "No XR_EXT_hand_tracking_data_source: camera hands cannot be told from controller hands, hands will not be shown."
            );
            return None;
        }
        let mut hands = Vec::with_capacity(2);
        for side in [xr::Hand::LEFT, xr::Hand::RIGHT] {
            // camera hands only: never the hand a controller emulates (drawn over the controller model)
            let Some(tracker) = super::camera_hands::create_unobstructed(xr, side) else {
                log::info!("No camera hand tracker for {side:?}, hands will not be shown.");
                return None;
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
        log::info!("Camera hand tracking enabled, drawing camera hands (never controller hands).");
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
                lines.draw_between_width(
                    id,
                    p[i * 2],
                    p[i * 2 + 1],
                    HAND_COLOR,
                    app.input_state.hmd,
                    HAND_LINE_WIDTH,
                );
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
