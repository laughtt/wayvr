use serde::{Deserialize, Serialize};

use crate::config_io;

const DEFAULT_XR_INPUT_PROFILES: &str = include_str!("../assets/openxr_actions.json5");

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany<T> {
	One(T),
	Many(Vec<T>),
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct OpenXrInputAction {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub left: Option<OneOrMany<String>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub right: Option<OneOrMany<String>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub handsfree: Option<OneOrMany<String>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub threshold_left: Option<[f32; 2]>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub threshold_right: Option<[f32; 2]>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub double_click: Option<bool>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub triple_click: Option<bool>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct OpenXrInputProfile {
	pub profile: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub pose: Option<OpenXrInputAction>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub click: Option<OpenXrInputAction>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub grab: Option<OpenXrInputAction>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub alt_click: Option<OpenXrInputAction>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub show_hide: Option<OpenXrInputAction>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub toggle_dashboard: Option<OpenXrInputAction>,
	/// Opens and closes the Work menu (the overlay named by the `menu_panel` setting; the dashboard without one).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub menu: Option<OpenXrInputAction>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub space_drag: Option<OpenXrInputAction>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub space_rotate: Option<OpenXrInputAction>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub space_reset: Option<OpenXrInputAction>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub pin: Option<OpenXrInputAction>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub reset_screens: Option<OpenXrInputAction>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub click_modifier_right: Option<OpenXrInputAction>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub click_modifier_middle: Option<OpenXrInputAction>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub move_mouse: Option<OpenXrInputAction>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub scroll: Option<OpenXrInputAction>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub haptic: Option<OpenXrInputAction>,
}

/// The profiles shipped in `assets/openxr_actions.json5`, without the user's override file.
pub fn default_xr_input_profiles() -> Vec<OpenXrInputProfile> {
	serde_json5::from_str(DEFAULT_XR_INPUT_PROFILES).unwrap() // want panic
}

pub fn load_xr_input_profiles() -> Vec<OpenXrInputProfile> {
	let mut profiles = default_xr_input_profiles();

	let Some(conf) = config_io::load("openxr_actions.json5") else {
		return profiles;
	};

	match serde_json5::from_str::<Vec<OpenXrInputProfile>>(&conf) {
		Ok(override_profiles) => {
			for new in override_profiles {
				if let Some(i) = profiles.iter().position(|old| old.profile == new.profile) {
					profiles[i] = new;
				} else {
					profiles.push(new);
				}
			}
		}
		Err(e) => {
			log::error!("Failed to load openxr_actions.json5: {e}");
		}
	}

	profiles
}

#[cfg(test)]
mod tests {
	use std::str::FromStr;

	use super::*;
	use crate::{
		openxr_bindings_schema::{XrInputComponent, XrInputSide, XrInputSubpathKind},
		openxr_controller_profiles::OPENXR_INPUT_PROFILES,
	};

	const TOUCH: &str = "/interaction_profiles/oculus/touch_controller";
	const INDEX: &str = "/interaction_profiles/valve/index_controller";
	const VIVE: &str = "/interaction_profiles/htc/vive_controller";
	const LEFT_MENU_CLICK: &str = "/user/hand/left/input/menu/click";
	const RIGHT_SYSTEM_CLICK: &str = "/user/hand/right/input/system/click";

	fn profile(id: &str) -> OpenXrInputProfile {
		default_xr_input_profiles()
			.into_iter()
			.find(|p| p.profile == id)
			.unwrap_or_else(|| panic!("no default profile {id}"))
	}

	/// Every binding path of an action (left, right and handsfree).
	fn paths(action: &OpenXrInputAction) -> Vec<&str> {
		[action.left.as_ref(), action.right.as_ref(), action.handsfree.as_ref()]
			.into_iter()
			.flatten()
			.flat_map(|spec| match spec {
				OneOrMany::One(p) => vec![p.as_str()],
				OneOrMany::Many(v) => v.iter().map(String::as_str).collect(),
			})
			.collect()
	}

	#[test]
	fn default_profiles_parse() {
		let profiles = default_xr_input_profiles();
		assert!(profiles.len() >= 5, "only {} default profiles", profiles.len());
		assert!(profiles.iter().any(|p| p.profile == TOUCH));
	}

	#[test]
	fn touch_left_menu_click_opens_the_menu() {
		let touch = profile(TOUCH);
		let menu = touch.menu.as_ref().expect("Touch binds `menu`");
		assert!(
			matches!(&menu.left, Some(OneOrMany::One(p)) if p == LEFT_MENU_CLICK),
			"{:?}",
			menu.left
		);
		// the right hand's menu button (Monado: /input/system/click on the right Touch) opens the same menu
		assert!(
			matches!(&menu.right, Some(OneOrMany::One(p)) if p == RIGHT_SYSTEM_CLICK),
			"{:?}",
			menu.right
		);
		assert!(menu.handsfree.is_none());
		// a single press: a double or triple click would not open it at the first press
		assert!(!menu.double_click.unwrap_or(false) && !menu.triple_click.unwrap_or(false));
	}

	#[test]
	fn touch_menu_button_no_longer_drags_or_resets_the_stage() {
		let touch = profile(TOUCH);
		assert!(touch.space_drag.is_none(), "space_drag still bound on Touch");
		assert!(touch.space_reset.is_none(), "space_reset still bound on Touch");
		// nothing else may share the left menu button with `menu`
		let users: Vec<String> = serde_json::to_value(&touch)
			.unwrap()
			.as_object()
			.unwrap()
			.iter()
			.filter(|(name, action)| name.as_str() != "profile" && action.to_string().contains(LEFT_MENU_CLICK))
			.map(|(name, _)| name.clone())
			.collect();
		assert_eq!(users, ["menu"]);
		// nor the right system button
		let users: Vec<String> = serde_json::to_value(&touch)
			.unwrap()
			.as_object()
			.unwrap()
			.iter()
			.filter(|(name, action)| name.as_str() != "profile" && action.to_string().contains(RIGHT_SYSTEM_CLICK))
			.map(|(name, _)| name.clone())
			.collect();
		assert_eq!(users, ["menu"]);
	}

	#[test]
	fn other_touch_bindings_are_kept() {
		let touch = profile(TOUCH);
		let show_hide = touch.show_hide.as_ref().unwrap();
		assert_eq!(show_hide.double_click, Some(true));
		assert_eq!(paths(show_hide), ["/user/hand/left/input/y/click"]);
		assert!(touch.click.is_some() && touch.grab.is_some() && touch.pin.is_some() && touch.reset_screens.is_some());
	}

	#[test]
	fn index_and_vive_bind_menu_to_a_free_left_button() {
		for (id, path) in [
			(INDEX, "/user/hand/left/input/system/click"),
			(VIVE, "/user/hand/left/input/menu/click"),
		] {
			let p = profile(id);
			assert_eq!(
				paths(p.menu.as_ref().unwrap_or_else(|| panic!("{id} binds menu"))),
				[path]
			);
			// the button is used by no other action of the profile
			let users = serde_json::to_value(&p)
				.unwrap()
				.as_object()
				.unwrap()
				.iter()
				.filter(|(name, action)| name.as_str() != "profile" && action.to_string().contains(path))
				.count();
			assert_eq!(users, 1, "{id}: {path} is shared");
		}
	}

	/// A wrong path makes the runtime reject ALL suggested bindings of the profile, so the `menu` paths are checked
	/// against the controller schema.
	#[test]
	fn menu_paths_exist_on_their_controller() {
		for p in default_xr_input_profiles() {
			let Some(menu) = p.menu.as_ref() else { continue };
			let schema = OPENXR_INPUT_PROFILES
				.iter()
				.find(|s| s.profile_id == p.profile)
				.unwrap_or_else(|| panic!("{}: menu bound but no controller schema", p.profile));
			for path in paths(menu) {
				// /user/hand/<side>/input/<subpath>/<component>
				let parts: Vec<&str> = path.split('/').collect();
				assert!(
					parts.len() == 7 && parts[1] == "user" && parts[2] == "hand" && parts[4] == "input",
					"{path}"
				);
				let side = XrInputSide::from_str(parts[3]).unwrap_or_else(|_| panic!("{path}: bad side"));
				let kind = XrInputSubpathKind::from_str(parts[5]).unwrap_or_else(|_| panic!("{path}: bad subpath"));
				let component = XrInputComponent::from_str(parts[6]).unwrap_or_else(|_| panic!("{path}: bad component"));
				let sub = schema
					.find_userpath(side)
					.and_then(|u| u.find_subpath(kind))
					.unwrap_or_else(|| panic!("{}: no {path}", p.profile));
				assert!(
					sub.components.contains(&component),
					"{}: {path} has no such component",
					p.profile
				);
			}
		}
	}
}
