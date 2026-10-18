// Contents of this file should be the same as on wayvr.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::ipc::Serial;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeSuccess {
	pub runtime: String, // Runtime name, for example "wayvr"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Disconnect {
	pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct WvrProcessHandle {
	pub user: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct WvrWindowHandle {
	pub user: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WvrWindow {
	pub size_x: u32,
	pub size_y: u32,
	pub visible: bool,
	pub handle: WvrWindowHandle,
	pub process_handle: WvrProcessHandle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WvrWindowList {
	pub list: Vec<WvrWindow>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WvrProcess {
	pub name: String,
	pub handle: WvrProcessHandle,
	pub userdata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WvrProcessList {
	pub list: Vec<WvrProcess>,
}

#[derive(Default, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Margins {
	pub left: u16,
	pub right: u16,
	pub top: u16,
	pub bottom: u16,
}

#[derive(Default, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct StackingOptions {
	pub margins_first: Margins,
	pub margins_rest: Margins,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum WvrStateChanged {
	ProcessCreated,
	ProcessRemoved,
	WindowCreated,
	WindowRemoved,
	DashboardShown,
	DashboardHidden,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct WlxInputStatePointer {
	pub pos: [f32; 3],
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct WlxInputState {
	pub hmd_pos: [f32; 3],
	pub left: WlxInputStatePointer,
	pub right: WlxInputStatePointer,
}

/// One gaze fixation (or the one in progress) on an overlay. Times are Unix epoch milliseconds.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct WlxGazeFixation {
	/// Overlay name, e.g. the screen `SUNSHINE`
	pub overlay: String,
	/// Mean hit point in overlay UV (0..1, origin top-left)
	pub uv: [f32; 2],
	/// The same point in desktop (compositor logical) pixels, for screen overlays; None for other overlays
	pub px: Option<[f32; 2]>,
	pub start_ms: u64,
	/// Time of the last sample in the fixation
	pub end_ms: u64,
	pub duration_ms: u32,
	/// 0..1: share of valid frames, spread and duration
	pub confidence: f32,
	/// RMS angular spread of the samples around the mean, degrees
	pub dispersion_deg: f32,
	/// Distance from the eyes to the hit point, metres
	pub dist_m: f32,
	pub samples: u32,
}

/// Answer to `WlxGaze`.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct WlxGaze {
	/// Unix epoch milliseconds when the answer was made
	pub ts_ms: u64,
	/// Silent gaze tracking runs (`gaze_target`, `gaze_laser` or `gaze_pointer` on)
	pub tracking: bool,
	/// The laser is drawn
	pub laser: bool,
	/// A valid gaze sample arrived this frame (false during blinks / lost tracking)
	pub valid: bool,
	/// Minimum fixation length for `last_stable`, milliseconds
	pub min_fixation_ms: u32,
	/// The fixation in progress (any length)
	pub current: Option<WlxGazeFixation>,
	/// The current fixation if it is long enough, else the newest long-enough one that ended in the last ~2 s
	pub last_stable: Option<WlxGazeFixation>,
	/// Long-enough fixations of the last ~10 s, oldest first (the current one is not included)
	pub history: Vec<WlxGazeFixation>,
}

/// Answer to `WlxGazeLearn`: the learned gaze correction (gaze click v2).
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct WlxGazeLearn {
	/// `gaze_learn` is on
	pub enabled: bool,
	/// Lessons kept (at most 150)
	pub lessons: u32,
	/// Lessons taken / refused since the file was created
	pub taken: u32,
	pub refused: u32,
	/// Error of the corrected gaze at the newest lesson, before it was learned, degrees
	pub last_error_deg: Option<f32>,
	/// The offset term of the fitted correction: yaw (+ left), pitch (+ up), degrees
	pub offset_deg: [f32; 2],
	/// Calibration fingerprint (`gaze_learn_key`)
	pub key: String,
}

/// Answer to `WlxGazeCheck`.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct WlxGazeCheck {
	pub ok: bool,
	/// Error of the corrected gaze at the target before / after the lesson, degrees (ok only)
	pub error_before_deg: Option<f32>,
	pub error_after_deg: Option<f32>,
	/// "no gaze", "not steady", "timed out" or "learning off" (not ok only)
	pub reason: Option<String>,
}

// "Wvr" prefixes are WayVR-specific

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PacketServer {
	Disconnect(Disconnect),
	HandshakeSuccess(HandshakeSuccess),
	WlxInputStateResponse(Serial, WlxInputState),
	WvrWindowListResponse(Serial, Option<WvrWindowList>),
	WvrProcessGetResponse(Serial, Option<WvrProcess>),
	WvrProcessLaunchResponse(Serial, Result<WvrProcessHandle, String>),
	WvrProcessListResponse(Serial, WvrProcessList),
	WvrStateChanged(WvrStateChanged),
	/// Appended last: no protocol bump needed
	WlxGazeResponse(Serial, WlxGaze),
	WlxGazeLearnResponse(Serial, WlxGazeLearn),
	WlxGazeCheckResponse(Serial, WlxGazeCheck),
}

impl PacketServer {
	pub fn serial(&self) -> Option<&Serial> {
		match self {
			PacketServer::Disconnect(_) => None,
			PacketServer::HandshakeSuccess(_) => None,
			PacketServer::WlxInputStateResponse(serial, _) => Some(serial),
			PacketServer::WvrWindowListResponse(serial, _) => Some(serial),
			PacketServer::WvrProcessGetResponse(serial, _) => Some(serial),
			PacketServer::WvrProcessLaunchResponse(serial, _) => Some(serial),
			PacketServer::WvrProcessListResponse(serial, _) => Some(serial),
			PacketServer::WvrStateChanged(_) => None,
			PacketServer::WlxGazeResponse(serial, _) => Some(serial),
			PacketServer::WlxGazeLearnResponse(serial, _) => Some(serial),
			PacketServer::WlxGazeCheckResponse(serial, _) => Some(serial),
		}
	}
}
