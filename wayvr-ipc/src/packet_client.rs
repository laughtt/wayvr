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
	/// Where the user is looking (silent eye tracking): answered with `WlxGazeResponse` (appended last)
	WlxGaze(Serial),
	/// The subtle gaze laser: Some(on/off), None = toggle (appended last)
	WlxGazeLaser(Option<bool>),
}
