//! Camera hands for the pointer: hand trackers that ask for the unobstructed data source only
//! (XR_EXT_hand_tracking_data_source), so a hand is "tracked" only when the runtime sees the bare hand with its
//! cameras, never for the hand a controller emulates. The pointer logic is in `backend/hand_pointer.rs`.

use std::{ptr, time::Instant};

use glam::{Affine3A, Vec3};
use openxr as xr;
use xr::sys::Handle as _;

use super::{XrState, helpers::posef_to_transform};
use crate::backend::hand_pointer::{HandGestures, HandJoints, HandSample};

const LOG_SECS: u64 = 5;

pub(super) struct CameraHands {
    trackers: [xr::HandTracker; 2],
    gestures: [HandGestures; 2],
    /// last logged `controllers_in_work`
    mode: Option<bool>,
    /// frames in the log window: (frames, left seen, right seen, frames the hands had the pointer)
    stats: (u32, u32, u32, u32),
    log_since: Option<Instant>,
}

fn create_unobstructed(xr: &XrState, hand: xr::Hand) -> Option<xr::HandTracker> {
    let fp = xr.instance.exts().ext_hand_tracking.as_ref()?;
    let mut sources = [xr::sys::HandTrackingDataSourceEXT::UNOBSTRUCTED];
    let source_info = xr::sys::HandTrackingDataSourceInfoEXT {
        ty: xr::sys::HandTrackingDataSourceInfoEXT::TYPE,
        next: ptr::null(),
        requested_data_source_count: 1,
        requested_data_sources: sources.as_mut_ptr(),
    };
    let info = xr::sys::HandTrackerCreateInfoEXT {
        ty: xr::sys::HandTrackerCreateInfoEXT::TYPE,
        next: (&raw const source_info).cast(),
        hand,
        hand_joint_set: xr::sys::HandJointSetEXT::DEFAULT,
    };
    let mut out = xr::sys::HandTrackerEXT::NULL;
    // SAFETY: valid session, create info and output handle; the chained structs live until the call returns.
    let r = unsafe { (fp.create_hand_tracker)(xr.session.as_raw(), &info, &mut out) };
    if r != xr::sys::Result::SUCCESS {
        log::info!(
            "camera hand pointer: no camera hand tracker for {hand:?} ({r:?}); controllers stay the pointer"
        );
        return None;
    }
    // SAFETY: just created on this session.
    Some(unsafe { xr::HandTracker::from_raw(&xr.session, out) })
}

impl CameraHands {
    /// `None` when the runtime cannot tell camera hands from controller hands (no data-source extension) or has no
    /// camera hand tracking (Monado started without `--hands`).
    pub(super) fn new(xr: &XrState) -> Option<Self> {
        let exts = xr.instance.exts();
        if exts.ext_hand_tracking.is_none() || exts.ext_hand_tracking_data_source.is_none() {
            log::info!(
                "camera hand pointer: the runtime has no XR_EXT_hand_tracking(_data_source); controllers stay the pointer"
            );
            return None;
        }
        let left = create_unobstructed(xr, xr::Hand::LEFT)?;
        let right = create_unobstructed(xr, xr::Hand::RIGHT)?;
        log::info!(
            "camera hand pointer: ready (camera hands are the pointer; controllers ignored in Work unless controllers_in_work)"
        );
        Some(Self {
            trackers: [left, right],
            gestures: [HandGestures::default(); 2],
            mode: None,
            stats: (0, 0, 0, 0),
            log_since: None,
        })
    }

    fn joints(&self, xr: &XrState, i: usize) -> Option<HandJoints> {
        let joints = xr
            .stage
            .locate_hand_joints(&self.trackers[i], xr.predicted_display_time)
            .ok()??;
        let at = |j: xr::HandJoint| -> Option<Vec3> {
            let loc = &joints[j.into_raw() as usize];
            loc.location_flags
                .contains(xr::SpaceLocationFlags::POSITION_VALID)
                .then(|| posef_to_transform(&loc.pose).translation.into())
        };
        Some(HandJoints {
            wrist: at(xr::HandJoint::WRIST)?,
            index_proximal: at(xr::HandJoint::INDEX_PROXIMAL)?,
            thumb_tip: at(xr::HandJoint::THUMB_TIP)?,
            index_tip: at(xr::HandJoint::INDEX_TIP)?,
            middle_tip: at(xr::HandJoint::MIDDLE_TIP)?,
        })
    }

    /// The camera hand pointers for this frame (tracked or not; the caller applies the rule).
    pub(super) fn update(
        &mut self,
        xr: &XrState,
        hmd: &Affine3A,
        controllers_in_work: bool,
    ) -> [Option<HandSample>; 2] {
        let now = Instant::now();
        let mut out = [None, None];
        for (i, slot) in out.iter_mut().enumerate() {
            match self.joints(xr, i) {
                Some(j) => *slot = self.gestures[i].sample(&j, i == 0, hmd),
                None => self.gestures[i].reset(),
            }
        }
        let active = !controllers_in_work;
        if self.mode != Some(controllers_in_work) {
            self.mode = Some(controllers_in_work);
            log::info!(
                "camera hand pointer: {}",
                if active {
                    "hands only, controllers ignored entirely"
                } else {
                    "controllers on (a tracked controller wins; a tracked camera hand fills in)"
                }
            );
        }

        let s = &mut self.stats;
        s.0 += 1;
        s.1 += u32::from(out[0].is_some());
        s.2 += u32::from(out[1].is_some());
        s.3 += u32::from(active);
        let since = *self.log_since.get_or_insert(now);
        if now.duration_since(since).as_secs() >= LOG_SECS {
            let (n, l, r, a) = std::mem::take(&mut self.stats);
            if l + r + a > 0 {
                log::info!(
                    "camera hand pointer: {n} frames, left tracked {l}, right tracked {r}, hands had the pointer {a}"
                );
            }
            self.log_since = Some(now);
        }
        out
    }
}
