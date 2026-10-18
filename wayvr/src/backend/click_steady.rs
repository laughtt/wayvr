//! Steady clicks: a held click does not jitter.
//!
//! On click-down the hit (overlay + position) is remembered. While the click is held on the same overlay and the
//! real hit stays within `click_steady_px` (overlay pixels) of the press point, the press point is reported, so a
//! shaky hand or a wobbling gaze cannot turn a click into a tiny drag. Once the real hit has moved further, it is
//! followed until release (no snapping back). 0 = off.

use glam::Vec2;

use crate::windowing::OverlayID;

#[derive(Debug, Clone, Copy, Default)]
enum Phase {
    #[default]
    Idle,
    /// Reporting the press point.
    Hold { overlay: OverlayID, uv: Vec2 },
    /// Left the steady zone (or never had one): the real hit is reported until release.
    Free,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Steady {
    phase: Phase,
}

impl Steady {
    pub const fn reset(&mut self) {
        self.phase = Phase::Idle;
    }

    /// The UV to report for this frame's hit.
    ///
    /// `down_edge`: the click went down this frame. `held`: the click is down (or was down last frame, so the
    /// release is reported at the press point too). `px_per_uv`: overlay pixels per UV unit on each axis.
    pub fn apply(
        &mut self,
        down_edge: bool,
        held: bool,
        overlay: OverlayID,
        uv: Vec2,
        px_per_uv: Vec2,
        limit_px: f32,
    ) -> Vec2 {
        if limit_px <= 0.0 || !held || !px_per_uv.is_finite() || px_per_uv.min_element() <= 0.0 {
            self.phase = Phase::Idle;
            return uv;
        }
        if down_edge {
            self.phase = Phase::Hold { overlay, uv };
            return uv;
        }
        match self.phase {
            Phase::Hold { overlay: o, uv: press } => {
                if o == overlay && ((uv - press) * px_per_uv).length() <= limit_px {
                    press
                } else {
                    self.phase = Phase::Free;
                    uv
                }
            }
            Phase::Idle => {
                // the press did not start on an overlay: nothing to hold on to
                self.phase = Phase::Free;
                uv
            }
            Phase::Free => uv,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PX: Vec2 = Vec2::new(1920.0, 1080.0);

    fn at(px: f32, py: f32) -> Vec2 {
        Vec2::new(px, py) / PX
    }

    #[test]
    fn jitter_stays_on_the_press_point() {
        let mut s = Steady::default();
        let id = OverlayID::default();
        let p = at(500.0, 300.0);
        assert_eq!(s.apply(true, true, id, p, PX, 32.0), p);
        for (dx, dy) in [(3.0, -2.0), (-10.0, 8.0), (20.0, 20.0), (-25.0, 0.0), (0.0, 31.0)] {
            let r = s.apply(false, true, id, at(500.0 + dx, 300.0 + dy), PX, 32.0);
            assert_eq!(r, p, "{dx},{dy}");
        }
        // the release frame (before.click) is still at the press point
        assert_eq!(s.apply(false, true, id, at(510.0, 305.0), PX, 32.0), p);
        // not held: reports the real hit and forgets
        let real = at(511.0, 306.0);
        assert_eq!(s.apply(false, false, id, real, PX, 32.0), real);
    }

    #[test]
    fn a_deliberate_drag_passes_after_the_limit_and_does_not_snap_back() {
        let mut s = Steady::default();
        let id = OverlayID::default();
        let p = at(500.0, 300.0);
        s.apply(true, true, id, p, PX, 32.0);
        let far = at(540.0, 300.0);
        assert_eq!(s.apply(false, true, id, far, PX, 32.0), far);
        // back near the press point: the real hit is followed, no snap
        let near = at(505.0, 300.0);
        assert_eq!(s.apply(false, true, id, near, PX, 32.0), near);
    }

    #[test]
    fn off_at_zero() {
        let mut s = Steady::default();
        let id = OverlayID::default();
        let p = at(500.0, 300.0);
        s.apply(true, true, id, p, PX, 0.0);
        let q = at(502.0, 301.0);
        assert_eq!(s.apply(false, true, id, q, PX, 0.0), q);
    }

    #[test]
    fn a_different_overlay_releases_it() {
        let mut steady = Steady::default();
        let mut map = slotmap::SlotMap::<OverlayID, ()>::with_key();
        let first = map.insert(());
        let second = map.insert(());
        let press = at(500.0, 300.0);
        steady.apply(true, true, first, press, PX, 32.0);
        let near = at(502.0, 301.0);
        assert_eq!(steady.apply(false, true, second, near, PX, 32.0), near);
        // and stays released even when back on the first overlay
        assert_eq!(steady.apply(false, true, first, near, PX, 32.0), near);
    }

    #[test]
    fn a_press_that_started_in_the_air_is_not_held() {
        let mut s = Steady::default();
        let id = OverlayID::default();
        let q = at(502.0, 301.0);
        assert_eq!(s.apply(false, true, id, q, PX, 32.0), q);
    }

    #[test]
    fn unknown_size_is_off() {
        let mut s = Steady::default();
        let id = OverlayID::default();
        let p = at(500.0, 300.0);
        s.apply(true, true, id, p, Vec2::ZERO, 32.0);
        let q = at(900.0, 700.0);
        assert_eq!(s.apply(false, true, id, q, Vec2::ZERO, 32.0), q);
    }
}
