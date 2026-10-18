// Contents of this file should be the same as on wayvr.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{ipc::Serial, packet_server};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Handshake {
	pub protocol_version: u32, // always set to PROTOCOL_VERSION
	pub magic: String,         // always set to CONNECTION_MAGIC
	pub client_name: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub enum PositionMode {
	Float,
	Anchor,
	Static,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum HandsfreeMode {
	None,
	Hmd,
	HmdPinch,
	EyeTracking,
	EyeTrackingPinch,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum HandsfreeAction {
	Click,
	Grab,
	RightModifier,
	MiddleModifier,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum HandsfreeParams {
	SetMode(HandsfreeMode),
	Press(HandsfreeAction),
	Release(HandsfreeAction),
	Toggle(HandsfreeAction),
	Scroll(f32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WvrProcessLaunchParams {
	pub name: String,
	pub exec: String,
	pub env: Vec<String>,
	pub args: String,
	pub icon: Option<String>,
	pub resolution: [u32; 2],
	pub pos_mode: PositionMode,
	pub userdata: HashMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WlxHapticsParams {
	pub intensity: f32,
	pub duration: f32,
	pub frequency: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WlxModifyPanelCommand {
	SetText(String),
	SetColor(String),
	SetImage(String),
	SetVisible(bool),
	SetStickyState(bool),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WlxModifyPanelParams {
	pub overlay: String,
	pub element: String,
	pub command: WlxModifyPanelCommand,
}

/// Desk layout: the first `screens` screens / app panels (0 = all) in an arc in front of the head, the rest hidden.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WlxDeskLayoutParams {
	pub screens: u32,
	pub curvature: f32,
	/// Width of each screen in metres.
	pub width: f32,
	/// Distance from the head in metres.
	pub distance: f32,
}

/// Which place a screen takes around the main one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WlxDeskSlot {
	Main,
	Left,
	Right,
	Up,
	/// Above the left screen (edge to edge)
	UpLeft,
	/// Above the right screen (edge to edge)
	UpRight,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WlxDeskScreen {
	/// Overlay name (the Wayland output name for screens)
	pub name: String,
	pub slot: WlxDeskSlot,
	pub visible: bool,
}

/// Place/show/hide named screens: main in front of the head, left/right/up edge to edge around it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WlxDeskScreensParams {
	pub screens: Vec<WlxDeskScreen>,
	/// Re-place the listed visible screens even if they are already shown
	pub place: bool,
	/// Hide every other screen / app panel / mirror
	pub hide_others: bool,
	/// Width of the main screen in metres (used when it is placed)
	pub width: f32,
	/// Distance from the head in metres (used when the main is placed)
	pub distance: f32,
	pub curvature: f32,
	/// Gap between neighbouring screen edges in metres
	pub gap: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WlxNudgeDir {
	Left,
	Right,
	Up,
	Down,
	Closer,
	Farther,
	Bigger,
	Smaller,
}

/// Move/scale a screen in space. `step`: fraction of its displayed width (left/right), height (up/down), of its
/// distance to the head (closer/farther) or of its size (bigger/smaller).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WlxScreenNudgeParams {
	pub name: String,
	pub dir: WlxNudgeDir,
	pub step: f32,
}

/// Where `WlxScreenPlace` puts a screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WlxPlaceRel {
	/// Left of the reference screen (without one: of its nearest neighbour on the left)
	Left,
	Right,
	/// Above
	Up,
	/// Below
	Down,
	/// Straight in front of the head ("bring it here")
	Front,
	/// Change places with the reference screen
	Swap,
}

/// Place screen `name` once, from the current head position, so that from there it overlaps no other screen: beside
/// `of` (or its nearest neighbour on that side), in front, or swapped with `of`; the other screens shift minimally
/// along the arc around the head.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WlxScreenPlaceParams {
	pub name: String,
	pub rel: WlxPlaceRel,
	pub of: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WlxScreenNumbersAction {
	/// Mode: never shown
	Off,
	/// Mode: shown for a few seconds when a screen is added or moved, or on flash (default)
	OnDemand,
	/// Mode: always shown
	Always,
	/// Show them now for the flash time (unless the mode is off)
	Show,
	/// Hide them now (until the next flash or screen change)
	Hide,
	/// Show them now for the flash time and highlight screen `n` (1..=6) if given
	Flash,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WlxScreenNumbersParams {
	pub action: WlxScreenNumbersAction,
	pub n: Option<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum PacketClient {
	Handshake(Handshake),
	WvrWindowList(Serial),
	WvrWindowSetVisible(packet_server::WvrWindowHandle, bool),
	WvrProcessGet(Serial, packet_server::WvrProcessHandle),
	WvrProcessLaunch(Serial, WvrProcessLaunchParams),
	WvrProcessList(Serial),
	WvrProcessTerminate(packet_server::WvrProcessHandle),
	WvrInputCapture(bool),
	WlxInputState(Serial),
	WlxModifyPanel(WlxModifyPanelParams),
	WlxDeviceHaptics(usize, WlxHapticsParams),
	WlxShowHide,
	WlxSwitchSet(Option<usize>),
	WlxHandsfree(HandsfreeParams),
	/// Put all screens back in front of the head (appended last: no protocol bump needed)
	WlxResetScreens,
	/// Recenter the play space (appended last: no protocol bump needed)
	WlxRecenter,
	/// Show/arrange screens and app panels (appended last)
	WlxDeskLayout(WlxDeskLayoutParams),
	/// Backdrop drawn over the main app: None = off, Some("") = built-in sky, Some(file) = DDS relative to the
	/// config root (appended last)
	WlxSetSkybox(Option<String>),
	/// Place/show/hide named screens around the main one (appended last)
	WlxDeskScreens(WlxDeskScreensParams),
	/// Move a screen left/right/up/down/closer/farther, or scale it (appended last)
	WlxScreenNudge(WlxScreenNudgeParams),
	/// Where the user is looking (silent eye tracking): answered with `WlxGazeResponse` (appended last)
	WlxGaze(Serial),
	/// The subtle gaze laser: Some(on/off), None = toggle (appended last)
	WlxGazeLaser(Option<bool>),
	/// Hands only, controllers ignored entirely (on) / controllers work too (off) (`controllers_in_work` = !on): Some(on/off), None = toggle (appended last)
	WlxPreferHands(Option<bool>),
	/// Screen number badges: mode, show/hide now, flash (optionally highlighting screen N) (appended last)
	WlxScreenNumbers(WlxScreenNumbersParams),
	/// Place a screen relative to another as seen from the head now; the others make room (appended last)
	WlxScreenPlace(WlxScreenPlaceParams),
	/// Read only: desk screen poses and the head pose, answered with `WlxScreensResponse` (appended last)
	WlxScreens(Serial),
	/// Locked desk layouts: the layout anchor from the head now (yaw only), the whole layout placed again in front
	/// of the user, same shape (appended last)
	WlxDeskRecenter,
}
