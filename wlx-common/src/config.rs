use std::{collections::HashMap, rc::Rc, sync::Arc};

use chrono::Offset;
use idmap::IdMap;
use serde::{Deserialize, Serialize};
use strum::{AsRefStr, EnumProperty, EnumString, VariantArray, VariantNames};
use wayvr_ipc::packet_client::WvrProcessLaunchParams;
use wgui::drawing::{self, HsvColor};

use crate::{
	astr_containers::{AStrMap, AStrSet},
	locale::{self},
	overlays::{BackendAttribValue, ToastDisplayMethod, ToastTopic},
	windowing::OverlayWindowState,
};

pub type PwTokenMap = AStrMap<String>;
pub type SerializedWindowStates = HashMap<Arc<str>, OverlayWindowState>;

#[derive(Default, Clone, Copy, Serialize, Deserialize, AsRefStr, EnumString, EnumProperty, VariantArray)]
pub enum CaptureMethod {
	#[default]
	#[serde(alias = "auto")]
	#[strum(props(Translation = "APP_SETTINGS.OPTION.AUTO", Tooltip = "APP_SETTINGS.OPTION.AUTO_HELP"))]
	Auto,

	#[serde(alias = "pipewire")]
	#[strum(props(
		Text = "PipeWire GPU",
		Tooltip = "APP_SETTINGS.OPTION.PIPEWIRE_HELP",
		Desktop = "Wayland"
	))]
	PipeWire,

	#[strum(props(
		Text = "ScreenCopy GPU",
		Tooltip = "APP_SETTINGS.OPTION.SCREENCOPY_GPU_HELP",
		Desktop = "Wayland"
	))]
	ScreenCopyGpu,

	#[serde(alias = "pw-fallback")]
	#[strum(props(
		Text = "PipeWire CPU",
		Tooltip = "APP_SETTINGS.OPTION.PW_FALLBACK_HELP",
		Desktop = "Wayland"
	))]
	PipeWireCpu,

	#[serde(alias = "screencopy")]
	#[strum(props(
		Text = "ScreenCopy CPU",
		Tooltip = "APP_SETTINGS.OPTION.SCREENCOPY_HELP",
		Desktop = "Wayland"
	))]
	ScreenCopyCpu,
}

#[derive(Default, Clone, Copy, Serialize, Deserialize, AsRefStr, EnumString, EnumProperty, VariantArray)]
pub enum InputEmulationMethod {
	#[default]
	#[strum(props(
		Translation = "APP_SETTINGS.OPTION.UINPUT",
		Tooltip = "APP_SETTINGS.OPTION.UINPUT_HELP"
	))]
	Uinput,

	#[strum(props(
		Translation = "APP_SETTINGS.OPTION.WL_VIRTUAL",
		Tooltip = "APP_SETTINGS.OPTION.WL_VIRTUAL_HELP",
		Desktop = "Wayland"
	))]
	WlVirtual,

	#[strum(props(
		Translation = "APP_SETTINGS.OPTION.NONE",
		Tooltip = "APP_SETTINGS.OPTION.NONE_INPUT_HELP"
	))]
	None,
}

#[derive(Default, Clone, Copy, Serialize, Deserialize, AsRefStr, EnumString, EnumProperty, VariantArray)]
pub enum InputCaptureMethod {
	#[strum(props(Tooltip = "APP_SETTINGS.OPTION.EVDEV_HELP"))]
	Evdev,

	#[default]
	#[strum(props(
		Translation = "APP_SETTINGS.OPTION.NONE",
		Tooltip = "APP_SETTINGS.OPTION.NONE_INPUT_CAPTURE_HELP"
	))]
	None,
}

#[derive(Default, Clone, Copy, Serialize, Deserialize, AsRefStr, EnumString, EnumProperty, VariantArray)]
pub enum DefaultPositioning {
	#[default]
	#[strum(props(Translation = "DEFINITIONS.ANCHORED", Tooltip = "APP_SETTINGS.OPTION.ANCHORED_HELP"))]
	Anchored,

	#[strum(props(Translation = "DEFINITIONS.FLOATING", Tooltip = "APP_SETTINGS.OPTION.FLOATING_HELP",))]
	Floating,

	#[strum(props(Translation = "DEFINITIONS.STATIC", Tooltip = "APP_SETTINGS.OPTION.STATIC_HELP"))]
	Static,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, AsRefStr, EnumString, EnumProperty, VariantArray)]
pub enum AltModifier {
	#[default]
	#[strum(props(Translation = "APP_SETTINGS.OPTION.NONE"))]
	None,
	Shift,
	Ctrl,
	Alt,
	Super,
	#[serde(alias = "Meta")]
	AltGr,
}

/// Look of the gaze laser (`gaze_laser_style`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GazeLaserStyle {
	/// A tiny dot at the hit point (about 0.6 degrees)
	#[default]
	Dot,
	/// A short thin ray that ends at the hit point
	Ray,
}

/// Screen number badges on the desk screens (`screen_numbers`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScreenNumbers {
	/// Never shown
	Off,
	/// Shown for a few seconds when a screen is added or moved, or on `wayvrctl screen-numbers flash`
	#[default]
	#[serde(alias = "ondemand", alias = "on-demand")]
	OnDemand,
	/// Always shown
	Always,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, AsRefStr, EnumString, EnumProperty, VariantArray)]
pub enum HandsfreePointer {
	#[strum(props(Translation = "APP_SETTINGS.OPTION.NONE"))]
	None,
	#[strum(props(Translation = "APP_SETTINGS.OPTION.HMD_PINCH"))]
	#[default]
	Hmd,
	#[strum(props(Translation = "APP_SETTINGS.OPTION.HMD_ONLY"))]
	HmdOnly,
	#[strum(props(Translation = "APP_SETTINGS.OPTION.EYE_PINCH", Backend = "OpenXR"))]
	EyeTracking,
	#[strum(props(Translation = "APP_SETTINGS.OPTION.EYE_ONLY", Backend = "OpenXR"))]
	EyeTrackingOnly,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, AsRefStr, EnumString, EnumProperty, VariantArray)]
pub enum HandsfreeAltTab {
	#[strum(props(Translation = "APP_SETTINGS.OPTION.HMD_ONLY"))]
	#[default]
	Hmd,
	#[strum(props(Translation = "APP_SETTINGS.OPTION.EYE_ONLY", Backend = "OpenXR"))]
	EyeTracking,
}

impl From<HandsfreeAltTab> for HandsfreePointer {
	fn from(value: HandsfreeAltTab) -> Self {
		match value {
			HandsfreeAltTab::Hmd => HandsfreePointer::HmdOnly,
			HandsfreeAltTab::EyeTracking => HandsfreePointer::EyeTrackingOnly,
		}
	}
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ChromaKeyParams {
	pub hsv_min: [f32; 3],
	pub hsv_max: [f32; 3],
	pub curve: f32,
	pub despill: f32,
}

fn hsv_green() -> [f32; 3] {
	[2.0 / 6.0, 1.0, 1.0]
}

impl Default for ChromaKeyParams {
	fn default() -> Self {
		Self {
			hsv_min: hsv_green(),
			hsv_max: hsv_green(),
			curve: 0.0, // Monado will ignore chroma keying completely if this value is zero
			despill: 0.0,
		}
	}
}

impl ChromaKeyParams {
	pub fn update_hsv_range_from_rgb(&mut self, rgb_color: drawing::Color, range_h: f32, range_s: f32, range_v: f32) {
		let hsv = HsvColor::from(rgb_color);
		self.hsv_min[0] = hsv.h - range_h / 2.0;
		self.hsv_min[1] = hsv.s - range_s / 2.0;
		self.hsv_min[2] = hsv.v - range_v / 2.0;
		self.hsv_max[0] = hsv.h + range_h / 2.0;
		self.hsv_max[1] = hsv.s + range_s / 2.0;
		self.hsv_max[2] = hsv.v + range_v / 2.0;
	}

	// Inverse of `update_hsv_range_from_rgb` function
	pub fn get_rgb_and_hsv_ranges(&self) -> (drawing::Color, f32, f32, f32) {
		(
			HsvColor::new(
				(self.hsv_min[0] + self.hsv_max[0]) / 2.0,
				(self.hsv_min[1] + self.hsv_max[1]) / 2.0,
				(self.hsv_min[2] + self.hsv_max[2]) / 2.0,
				1.0,
			)
			.to_rgb(),
			self.hsv_max[0] - self.hsv_min[0],
			self.hsv_max[1] - self.hsv_min[1],
			self.hsv_max[2] - self.hsv_min[2],
		)
	}
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SerializedWindowSet {
	pub name: Arc<str>,

	#[serde(default)]
	pub overlays: SerializedWindowStates,

	#[serde(default)]
	pub hidden_overlays: SerializedWindowStates,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, Eq, PartialEq, EnumString, VariantNames, AsRefStr)]
pub enum AppPosMode {
	Floating,
	Anchored,
	Static,
}
#[derive(Debug, Serialize, Deserialize, Clone, Copy, Eq, PartialEq, EnumString, VariantNames, AsRefStr)]
pub enum AppOrientationMode {
	Wide,
	SemiWide,
	Square,
	SemiTall,
	Tall,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, Eq, PartialEq, EnumString, VariantNames, AsRefStr)]
pub enum AppCompositorMode {
	Cage,
	Native,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, Eq, PartialEq, EnumString, VariantNames, AsRefStr)]
pub enum AppResMode {
	Res1440,
	Res1080,
	Res720,
	Res480,
}

#[derive(Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct PinnedApp {
	pub app_id: Rc<str>, // desktop entry app id, for example "libreoffice-draw"
	pub compositor_mode: AppCompositorMode,
	pub pos_mode: AppPosMode,
	pub orientation_mode: AppOrientationMode,
	pub res_mode: AppResMode,
}

pub const fn def_pw_tokens() -> PwTokenMap {
	AStrMap::new()
}

const fn def_mouse_move_interval_ms() -> i32 {
	10 // 100fps
}

const fn def_click_freeze_time_ms() -> i32 {
	300
}

const fn def_true() -> bool {
	true
}

const fn def_true_wl() -> bool {
	true
}
const fn def_grab_dist_rate() -> f32 {
	0.8
}
const fn def_size_min() -> f32 {
	0.3
}
const fn def_size_max() -> f32 {
	20.0
}
const fn def_dist_min() -> f32 {
	0.4
}
const fn def_dist_max() -> f32 {
	20.0
}

const fn def_one() -> f32 {
	1.0
}

const fn def_half() -> f32 {
	0.5
}

const fn def_point7() -> f32 {
	0.7
}

const fn def_point3() -> f32 {
	0.3
}

const fn def_gaze_laser_alpha() -> f32 {
	0.35
}

const fn def_gaze_fixation_ms() -> u32 {
	200
}

const fn def_gaze_deadzone() -> f32 {
	0.6
}

const fn def_point15() -> f32 {
	0.15
}

const fn def_osc_port() -> u16 {
	9000
}

fn def_timezones() -> Vec<String> {
	const EMEA: i32 = -60 * 60; // UTC-1
	const APAC: i32 = 5 * 60 * 60; // UTC+5

	let offset = chrono::Local::now().offset().fix();
	match offset.local_minus_utc() {
		i32::MIN..EMEA => vec!["Europe/Paris".into(), "Asia/Tokyo".into()],
		EMEA..APAC => vec!["America/New_York".into(), "Asia/Tokyo".into()],
		APAC..=i32::MAX => vec!["Europe/Paris".into(), "America/New_York".into()],
	}
}

fn def_default() -> Arc<str> {
	"Default".into()
}

fn def_theme_path() -> Arc<str> {
	"theme".into()
}

const fn def_max_height() -> u16 {
	1440
}

#[derive(Deserialize, Serialize)]
pub struct GeneralConfig {
	#[serde(default = "def_theme_path")]
	pub theme_path: Arc<str>,

	pub language: Option<locale::Language>, // auto-detected at runtime if unset

	#[serde(default = "def_one")]
	#[serde(alias = "ui_animation_speed", alias = "animation_speed" /* old name */)]
	pub ui_animation_speed: f32,

	#[serde(default = "def_one")]
	#[serde(alias = "ui_round_multiplier", alias = "round_multiplier" /* old name */)]
	pub ui_round_multiplier: f32,

	#[serde(default = "def_point3")]
	pub ui_gradient_intensity: f32,

	pub default_keymap: Option<String>,

	#[serde(default)]
	pub keyboard_layouts: Vec<Arc<str>>,

	#[serde(default)]
	pub attribs: AStrMap<Vec<BackendAttribValue>>,

	#[serde(default = "def_click_freeze_time_ms")]
	pub click_freeze_time_ms: i32,

	#[serde(default)]
	pub invert_scroll_direction_x: bool,

	#[serde(default)]
	pub invert_scroll_direction_y: bool,

	#[serde(default = "def_one")]
	pub scroll_speed: f32,

	#[serde(default = "def_one")]
	pub long_press_duration: f32,

	#[serde(default = "def_mouse_move_interval_ms")]
	pub mouse_move_interval_ms: i32,

	#[serde(default = "def_true")]
	pub notifications_enabled: bool,

	#[serde(default = "def_true")]
	pub notifications_sound_enabled: bool,

	#[serde(default)]
	pub notification_topics: IdMap<ToastTopic, ToastDisplayMethod>,

	#[serde(default = "def_true")]
	pub keyboard_sound_enabled: bool,

	#[serde(default = "def_one")]
	pub keyboard_scale: f32,

	#[serde(default = "def_one")]
	pub default_overlay_scale: f32,

	#[serde(default = "def_half")]
	pub watch_view_angle_min: f32,

	#[serde(default = "def_point7")]
	pub watch_view_angle_max: f32,

	#[serde(default = "def_osc_port")]
	pub osc_out_port: u16,

	#[serde(default)]
	pub upright_screen_fix: bool,

	#[serde(default)]
	pub double_cursor_fix: bool,

	#[serde(default = "def_true")]
	pub enable_watch: bool,

	/// false: the keyboard overlay is never shown (desk mode with voice captions)
	#[serde(default = "def_true")]
	pub enable_keyboard: bool,

	#[serde(default)]
	pub sets_on_watch: bool,

	#[serde(default)]
	pub hide_grab_help: bool,

	/// Desk mode: grabbing moves one overlay (never the anchor); on release it becomes Static (world-locked, saved).
	/// The `pin` action toggles `grabbable` on the hovered overlay (pinned overlays cannot be grabbed).
	#[serde(default)]
	pub world_lock_grab: bool,

	/// The `pin` action (A/X click) is only active with this (default off).
	#[serde(default)]
	pub world_lock_pin: bool,

	/// With world_lock_grab: also send a screen back to the front after it was out of view (>80 degrees) for
	/// 10 s at any time. Default off: a screen the user placed never moves because they looked away. Lost screens
	/// (no saved pose, invalid pose) and the first 30 s after start-up are always handled.
	#[serde(default)]
	pub world_lock_out_of_view_reset: bool,

	/// omarchyvr locked layouts (default on): desk screens (desktop captures) cannot be grabbed, moved or resized,
	/// are never auto-reset, refuse `screen-nudge` / `screen-place`, and `desk-screens` places them around the stored
	/// layout anchor (`desk-anchor.conf`; set by `desk-recenter`) instead of the head. Off = the old behaviour.
	#[serde(default = "def_true")]
	pub desk_locked: bool,

	/// With world_lock_grab: while gripping, stick = push/pull (rate per second, e-fold); resize = two-hand grip.
	#[serde(default = "def_grab_dist_rate")]
	pub grab_dist_rate: f32,
	/// Changing the distance scales the overlay with it (same apparent size).
	#[serde(default = "def_true_wl")]
	pub grab_keep_apparent_size: bool,
	/// Overlay width limits in metres (scale) and distance limits in metres.
	#[serde(default = "def_size_min")]
	pub grab_size_min: f32,
	#[serde(default = "def_size_max")]
	pub grab_size_max: f32,
	#[serde(default = "def_dist_min")]
	pub grab_dist_min: f32,
	#[serde(default = "def_dist_max")]
	pub grab_dist_max: f32,

	#[serde(default)]
	pub custom_panels: AStrSet,

	/// Name of the overlay the `menu` binding (Oculus Touch: the left menu button) opens and closes, normally a
	/// custom panel (list it in `custom_panels` too): `menu_panel: work_menu` in conf.d. That panel is global,
	/// 0.45 m wide, pointable and not grabbable, and each time it is shown it appears about 0.6 m in front of the
	/// head. Unset, or no overlay with this name: the `menu` binding toggles the dashboard instead. Not saved by the app.
	#[serde(default)]
	pub menu_panel: Option<Arc<str>>,

	/// Show the `menu_panel` overlay once when WayVR starts (in front of the head), instead of only on the `menu`
	/// binding: the VR home of omarchyvr (`menu_open_at_start: true` in conf.d). Default false. Not saved by the app.
	#[serde(default)]
	pub menu_open_at_start: bool,

	#[serde(default)]
	pub capture_method: CaptureMethod,

	#[serde(default)]
	pub input_emulation_method: InputEmulationMethod,

	#[serde(default = "def_true")]
	pub allow_sliding: bool,

	/// omarchyvr: ON by default. Pointing at a screen does not move the real desktop cursor; it moves on click,
	/// trigger or grab (and while `move_mouse` is held). Off = every hover moves the cursor (upstream default).
	#[serde(default = "def_true")]
	pub focus_follows_mouse_mode: bool,

	#[serde(default)]
	pub left_handed_mouse: bool,

	#[serde(default)]
	pub block_game_input: bool,

	#[serde(default = "def_true")]
	pub block_game_input_ignore_watch: bool,

	#[serde(default = "def_true")]
	pub block_poses_on_kbd_interaction: bool,

	#[serde(default = "def_one")]
	pub space_drag_multiplier: f32,

	#[serde(default)]
	pub skybox_texture: Arc<str>,

	#[serde(default = "def_true")]
	pub use_skybox: bool,

	#[serde(default = "def_true")]
	pub use_passthrough: bool,

	#[serde(default = "def_max_height")]
	pub screen_max_height: u16,

	#[serde(default)]
	pub screen_render_down: bool,

	#[serde(default = "def_point3")]
	pub pointer_lerp_factor: f32,

	#[serde(default = "def_true")]
	pub space_drag_unlocked: bool,

	#[serde(default)]
	pub space_drag_affects_world: bool,

	#[serde(default)]
	pub space_rotate_unlocked: bool,

	#[serde(default)]
	pub space_gravity_enabled: bool,

	#[serde(default = "def_one")]
	pub space_gravity_gravity: f32,

	#[serde(default = "def_one")]
	pub space_gravity_damping: f32,

	#[serde(default = "def_one")]
	pub space_gravity_fling_strength: f32,

	#[serde(default = "def_one")]
	pub space_gravity_ground_friction: f32,

	#[serde(default)]
	pub space_gravity_floor_height: f32,

	#[serde(default)]
	pub alt_click_down: Vec<String>,

	#[serde(default)]
	pub alt_click_up: Vec<String>,

	#[serde(default = "def_timezones")]
	pub timezones: Vec<String>,

	#[serde(default)]
	pub clock_12h: bool,

	#[serde(default)]
	pub sets: Vec<SerializedWindowSet>,

	#[serde(default)]
	/// Obsolete, only used for reading
	/// Use `sets` with `name: "global"`
	pub global_set: SerializedWindowStates,

	#[serde(default)]
	pub spawn_overlays: Vec<Arc<str>>,

	#[serde(default)]
	pub autostart_apps: Vec<WvrProcessLaunchParams>,

	#[serde(default)]
	pub pinned_apps: Vec<PinnedApp>,

	#[serde(default)]
	pub last_set: u32,

	#[serde(default)]
	pub hide_username: bool,

	#[serde(default)]
	pub opaque_background: bool,

	#[serde(default)]
	pub xwayland_by_default: bool,

	#[serde(default)]
	pub context_menu_hold_and_release: bool,

	#[serde(default)]
	pub keyboard_middle_click_mode: AltModifier,

	#[serde(default)]
	pub handsfree_pointer: HandsfreePointer,

	#[serde(default)]
	pub handsfree_alt_tab: HandsfreeAltTab,

	/// OpenXR only: use eye gaze (XR_EXT_eye_gaze_interaction) as an additional hover pointer.
	#[serde(default)]
	pub gaze_pointer: bool,

	/// Gaze has priority over controller rays: while gaze is valid, controller rays are not used for
	/// hovering (a controller trigger clicks where you look). Needed when the runtime reports
	/// controllers with synthesized poses that always point at the overlays.
	#[serde(default)]
	pub gaze_over_controllers: bool,

	/// OpenXR only: draw simple hands (needs XR_EXT_hand_tracking). Ignored if the runtime has no hand tracking.
	#[serde(default = "def_true")]
	pub show_hands: bool,

	/// OpenXR only: in Work the pointer is a bare hand the runtime tracks with its cameras (ray from the shoulder
	/// through the knuckle, index pinch = click, middle pinch = grab) and the controllers are ignored ENTIRELY
	/// (rays, buttons, triggers, sticks, grabs). true = controllers work too (a tracked controller wins).
	/// Needs XR_EXT_hand_tracking_data_source (without it the controllers always work).
	/// Toggle at run time with `wayvrctl controllers on|off|toggle` (`prefer-hands` is the inverse alias).
	#[serde(default)]
	pub controllers_in_work: bool,

	/// OpenXR only: draw a 3D controller model (Oculus Touch v3 / Quest 2, WebXR input profiles, MIT) at each
	/// controller's grip pose while that controller is the pointer, trigger/grip/stick/buttons animated, like SteamVR
	/// draws its render models. Hidden while camera hands are the pointer. Read at startup (Touch profile only).
	#[serde(default = "def_true")]
	pub show_controller_models: bool,

	/// OpenXR only, debug: draw axes at each controller's aim (ray origin) and grip pose: red +X, green +Y,
	/// blue = forward (-Z, the ray direction). Read at startup.
	#[serde(default)]
	pub controller_axes: bool,

	/// OpenXR only: pinch (XR_EXT_hand_interaction) clicks overlays. Read at startup.
	#[serde(default = "def_true")]
	pub hand_pinch_click: bool,

	/// Gaze dwell-click time in milliseconds. 0 = disabled.
	#[serde(default)]
	pub gaze_dwell_ms: u32,

	/// Gaze smoothing, 0.0 (none) to 1.0 (heavy).
	#[serde(default = "def_half")]
	pub gaze_smoothing: f32,

	/// Gaze fixation dead zone in degrees. The cursor holds still within this angle.
	#[serde(default = "def_gaze_deadzone")]
	pub gaze_deadzone_deg: f32,

	/// OpenXR only: silent gaze tracking. The gaze is ray-cast against the overlays every frame without hovering,
	/// clicking or moving the mouse; `wayvrctl gaze` reports what is looked at only while `gaze_pointer` (the one gaze
	/// input gate, `backend::gaze::gate`) is on.
	#[serde(default)]
	pub gaze_target: bool,

	/// OpenXR only: draw a subtle marker where the eyes land (implies silent gaze tracking).
	/// Toggle at run time with `wayvrctl gaze-laser on|off|toggle`.
	#[serde(default)]
	pub gaze_laser: bool,

	/// Gaze laser look: a tiny dot at the hit point, or a short thin ray ending there.
	#[serde(default)]
	pub gaze_laser_style: GazeLaserStyle,

	/// Gaze laser opacity, 0..1.
	#[serde(default = "def_gaze_laser_alpha")]
	pub gaze_laser_alpha: f32,

	/// Number badges (1-6) at the top centre of every desk screen: off, on_demand (default, ~5 s when a screen is
	/// added or moved or voice flashes them) or always. Change at run time with `wayvrctl screen-numbers`.
	#[serde(default)]
	pub screen_numbers: ScreenNumbers,

	/// Shortest fixation (ms) that counts as "looked at" for `wayvrctl gaze` (`last_stable`, `history`).
	#[serde(default = "def_gaze_fixation_ms")]
	pub gaze_fixation_ms: u32,

	#[serde(default = "def_one")]
	pub grid_opacity: f32,

	#[serde(default)]
	pub chroma_key_params: ChromaKeyParams,

	#[serde(default)]
	pub tutorial_graduated: bool,

	#[serde(default)]
	pub whisper_model: Arc<str>,

	#[serde(default = "def_default")]
	pub color_palette: Arc<str>,

	#[serde(default)]
	pub snap_angle_deg: f32,

	#[serde(default = "def_true")]
	pub wvr_mouse_acceleration: bool,

	#[serde(default)]
	pub wvr_mouse_speed: f32,

	#[serde(default)]
	pub wvr_input_capture: InputCaptureMethod,

	#[serde(default)]
	pub default_positioning: DefaultPositioning,

	#[serde(default = "def_one")]
	pub default_opacity: f32,

	#[serde(default = "def_point15")]
	pub default_curvature: f32,
}

#[cfg(test)]
mod omarchyvr_tests {
	use super::*;

	#[test]
	fn pointing_does_not_move_the_desktop_cursor_by_default() {
		let c: GeneralConfig = serde_json5::from_str("{}").expect("empty config parses");
		assert!(c.focus_follows_mouse_mode, "hover must not move the cursor unless the user asks");
		let c: GeneralConfig = serde_json5::from_str("{focus_follows_mouse_mode: false}").unwrap();
		assert!(!c.focus_follows_mouse_mode, "the app switch turns it back on");
	}

	#[test]
	fn desk_screens_are_locked_by_default() {
		let c: GeneralConfig = serde_json5::from_str("{}").expect("empty config parses");
		assert!(c.desk_locked);
		let c: GeneralConfig = serde_json5::from_str("{desk_locked: false}").unwrap();
		assert!(!c.desk_locked);
	}
}
