//! Screen number badges: a rounded pill with a big digit (1-6) at the top centre of every desk screen, so the user
//! can say "move the terminal from screen 2 to screen 4".
//!
//! Numbering (the one `pimax desk screens show N` uses, docs/SCREENS.md): 1 main, 2 left, 3 right, 4 up, 5 up-left,
//! 6 up-right. Modes (`screen_numbers` config, `wayvrctl screen-numbers off|ondemand|always`): off, on demand (default:
//! shown for a few seconds when a screen is added or moved, or on `show` / `flash`) and always. `flash N` also
//! highlights screen N (bigger, amber) for a moment: the target of a voice command.
//!
//! The badges are six small non-interactive overlays kept at the top centre of their screen every tick, so they follow
//! when a screen is moved, resized or re-placed. They are created the first time they are needed.

use std::{
    sync::{Arc, LazyLock, Mutex},
    time::{Duration, Instant},
};

use anyhow::Context;
use glam::{Affine3A, Mat3A, Vec3A};
use wayvr_ipc::packet_client::{WlxDeskSlot, WlxScreenNumbersAction, WlxScreenNumbersParams};
use wgui::{i18n::Translation, widget::label::WidgetLabel};
use wlx_common::{
    config::ScreenNumbers,
    windowing::{OverlayWindowState, Positioning},
};

use crate::{
    backend::task::ModifyPanelCommand,
    gui::panel::{GuiPanel, NewGuiPanelParams, OnCustomIdFunc},
    state::AppState,
    windowing::{
        OverlayID, Z_ORDER_ANCHOR,
        backend::OverlayEventData,
        manager::OverlayWindowManager,
        window::{OverlayCategory, OverlayWindowConfig, OverlayWindowData},
    },
};

/// How long on-demand badges stay.
pub const FLASH: Duration = Duration::from_secs(5);
/// How long a highlighted badge stays highlighted.
pub const HIGHLIGHT: Duration = Duration::from_millis(2500);
/// Badge width as a fraction of its screen's displayed width (a highlighted one is bigger).
const WIDTH_FRAC: f32 = 0.15;
const HIGHLIGHT_SCALE: f32 = 1.4;

/// The number of a desk slot (1 = main ... 6 = up-right).
pub const fn number_of(slot: WlxDeskSlot) -> u8 {
    match slot {
        WlxDeskSlot::Main => 1,
        WlxDeskSlot::Left => 2,
        WlxDeskSlot::Right => 3,
        WlxDeskSlot::Up => 4,
        WlxDeskSlot::UpLeft => 5,
        WlxDeskSlot::UpRight => 6,
    }
}

pub const fn slot_label(n: u8) -> &'static str {
    match n {
        1 => "main",
        2 => "left",
        3 => "right",
        4 => "up",
        5 => "up-left",
        6 => "up-right",
        _ => "",
    }
}

pub fn badge_name(n: u8) -> Arc<str> {
    format!("screen-number-{n}").into()
}

/// What the badges should do now; pure, so it is unit tested.
#[derive(Debug, Default, Clone)]
pub struct Plan {
    /// End of the on-demand window.
    pub show_until: Option<Instant>,
    /// `hide` was said: hidden until the next flash.
    pub hidden: bool,
    /// Highlighted screen number and the end of the highlight.
    pub highlight: Option<(u8, Instant)>,
    /// The screens (overlay name, number) the last `desk-screens` call listed as shown.
    pub shown: Vec<(Arc<str>, u8)>,
}

impl Plan {
    pub fn visible(&self, mode: ScreenNumbers, now: Instant) -> bool {
        match mode {
            ScreenNumbers::Off => false,
            ScreenNumbers::Always => !self.hidden,
            ScreenNumbers::OnDemand => !self.hidden && self.show_until.is_some_and(|t| now < t),
        }
    }

    pub fn highlighted(&self, now: Instant) -> Option<u8> {
        self.highlight.filter(|(_, t)| now < *t).map(|(n, _)| n)
    }

    pub fn flash(&mut self, now: Instant, n: Option<u8>) {
        self.hidden = false;
        self.show_until = Some(now + FLASH);
        if let Some(n) = n {
            self.highlight = Some((n, now + HIGHLIGHT));
        }
    }

    /// The screens listed by a `desk-screens` call. A screen that appears, or a new set of screens, flashes the badges.
    pub fn note_screens(&mut self, now: Instant, shown: Vec<(Arc<str>, u8)>, replaced: bool) {
        let changed = self.shown != shown;
        self.shown = shown;
        if changed || replaced {
            self.flash(now, None);
        }
    }

    pub fn number_of_screen(&self, name: &str) -> Option<u8> {
        self.shown.iter().find(|(n, _)| n.as_ref() == name).map(|(_, n)| *n)
    }
}

struct Runtime {
    plan: Plan,
    /// Badge shown last tick, per number (index n-1), and whether it was highlighted.
    applied: [Option<bool>; 6],
}

static RT: LazyLock<Mutex<Runtime>> = LazyLock::new(|| Mutex::new(Runtime { plan: Plan::default(), applied: [None; 6] }));

/// `desk-screens` listed these screens (name, slot, shown). `place`: the call re-placed them.
pub fn note_desk_screens(screens: &[(Arc<str>, WlxDeskSlot, bool)], place: bool) {
    let mut shown: Vec<(Arc<str>, u8)> = screens.iter().filter(|s| s.2).map(|s| (s.0.clone(), number_of(s.1))).collect();
    shown.sort_by_key(|s| s.1);
    let mut rt = RT.lock().unwrap();
    rt.plan.note_screens(Instant::now(), shown, place);
}

/// The slot number (1 main .. 6 up-right) of a screen: from the last `desk-screens` call, else from the omarchyvr
/// output name (`PIMAX-LEFT`, `PIMAX-RIGHT`, `PIMAX-UP`, `PIMAX-UPLEFT`, `PIMAX-UPRIGHT`). None: not a desk slot.
pub fn slot_number(name: &str) -> Option<u8> {
    if let Some(n) = RT.lock().ok().and_then(|rt| rt.plan.number_of_screen(name)) {
        return Some(n);
    }
    slot_number_of_output(name)
}

/// The slot number an omarchyvr output name stands for (docs/SCREENS.md).
pub fn slot_number_of_output(name: &str) -> Option<u8> {
    match name {
        "PIMAX-LEFT" => Some(2),
        "PIMAX-RIGHT" => Some(3),
        "PIMAX-UP" => Some(4),
        "PIMAX-UPLEFT" => Some(5),
        "PIMAX-UPRIGHT" => Some(6),
        _ => None,
    }
}

/// The screens the last `desk-screens` call showed: (overlay name, slot number), sorted by number.
pub fn shown_screens() -> Vec<(Arc<str>, u8)> {
    RT.lock().unwrap().plan.shown.clone()
}

/// A screen was moved, resized or nudged: flash the badges and highlight it.
pub fn note_screen_moved(name: &str) {
    let mut rt = RT.lock().unwrap();
    let n = rt.plan.number_of_screen(name);
    rt.plan.flash(Instant::now(), n);
}

/// `wayvrctl screen-numbers ...`
pub fn handle(app: &mut AppState, p: WlxScreenNumbersParams) {
    use WlxScreenNumbersAction as A;
    let now = Instant::now();
    let mut rt = RT.lock().unwrap();
    let n = p.n.filter(|n| (1..=6).contains(n));
    match p.action {
        A::Off => app.session.config.screen_numbers = ScreenNumbers::Off,
        A::OnDemand => app.session.config.screen_numbers = ScreenNumbers::OnDemand,
        A::Always => {
            app.session.config.screen_numbers = ScreenNumbers::Always;
            rt.plan.hidden = false;
        }
        A::Show => rt.plan.flash(now, None),
        A::Hide => {
            rt.plan.hidden = true;
            rt.plan.show_until = None;
            rt.plan.highlight = None;
        }
        A::Flash => rt.plan.flash(now, n),
    }
    log::info!("screen numbers: {:?} {:?} (mode {:?})", p.action, n, app.session.config.screen_numbers);
}

fn new_badge(app: &mut AppState, n: u8) -> anyhow::Result<OverlayWindowConfig> {
    let num = Translation::from_raw_text(&n.to_string());
    let slot = Translation::from_raw_text(&slot_label(n).to_uppercase());
    let on_custom_id: OnCustomIdFunc<()> = Box::new(move |id, widget, _doc_params, layout, _parser_state, ()| {
        let text = match &*id {
            "num" => &num,
            "slot" => &slot,
            _ => return Ok(()),
        };
        let mut label = layout
            .state
            .widgets
            .get_as::<WidgetLabel>(widget)
            .context("screen-number.xml: label expected")?;
        let mut globals = layout.state.globals.get();
        label.set_text_simple(&mut globals, text.clone());
        Ok(())
    });
    let mut panel = GuiPanel::new_from_template(
        app,
        "gui/screen-number.xml",
        (),
        NewGuiPanelParams { on_custom_id: Some(on_custom_id), ..Default::default() },
    )?;
    panel.update_layout(app)?;
    panel.on_notify = Some(Box::new(|panel, app, event_data| {
        if let OverlayEventData::CustomCommand { element, command } = event_data {
            crate::gui::panel::apply_custom_command(panel, app, &element, &command)?;
        }
        Ok(())
    }));
    Ok(OverlayWindowConfig {
        name: badge_name(n),
        category: OverlayCategory::Internal,
        z_order: Z_ORDER_ANCHOR - 1,
        default_state: OverlayWindowState {
            interactable: false,
            grabbable: false,
            block_input: false,
            positioning: Positioning::Static,
            transform: Affine3A::from_scale(glam::Vec3::splat(0.2)),
            ..OverlayWindowState::default()
        },
        ..OverlayWindowConfig::from_backend(Box::new(panel))
    })
}

fn quad(t: &Affine3A, extent: [u32; 2]) -> Option<(f32, f32)> {
    if extent[0] == 0 || extent[1] == 0 {
        return None;
    }
    let aspect = extent[0] as f32 / extent[1] as f32;
    Some(if aspect < 1.0 {
        let major = t.matrix3.col(0).length();
        (major * aspect, major)
    } else {
        let major = t.matrix3.col(1).length();
        (major, major / aspect)
    })
}

/// The pose of a badge: top centre of the screen, just inside its upper edge, same orientation. `screen`: the
/// screen's transform and displayed (width, height); `aspect`/`k`: the badge panel's aspect and its meta scale.
pub fn badge_pose(screen: &Affine3A, size: (f32, f32), aspect: f32, k: f32, scale_up: f32) -> Affine3A {
    let right = screen.matrix3.x_axis.normalize_or_zero();
    let up = screen.matrix3.y_axis.normalize_or_zero();
    let fwd = screen.matrix3.z_axis.normalize_or_zero();
    let width = (size.0 * WIDTH_FRAC * scale_up).max(0.05);
    let height = width / aspect.max(0.1);
    let scale = width / k.max(1e-6);
    let margin = size.1 * 0.02;
    let centre = Vec3A::from(screen.translation) + up * (size.1 * 0.5 - margin - height * 0.5);
    Affine3A { matrix3: Mat3A::from_cols(right * scale, up * scale, fwd * scale), translation: centre }
}

/// Every tick: create the badges when first needed, show/hide them and keep them at their screens' top centre.
pub fn tick<O: Default>(app: &mut AppState, overlays: &mut OverlayWindowManager<O>) {
    let now = Instant::now();
    let mode = app.session.config.screen_numbers;
    let (visible, hl, shown) = {
        let rt = RT.lock().unwrap();
        (rt.plan.visible(mode, now), rt.plan.highlighted(now), rt.plan.shown.clone())
    };

    for n in 1..=6u8 {
        let name = badge_name(n);
        let screen_name = shown.iter().find(|(_, k)| *k == n).map(|(s, _)| s.clone());
        let want = visible && screen_name.is_some();
        let id: Option<OverlayID> = overlays.lookup(&name);

        let id = match id {
            Some(id) => id,
            None if want => match new_badge(app, n) {
                Ok(cfg) => overlays.add(OverlayWindowData::from_config(cfg), app),
                Err(e) => {
                    log::warn!("screen number {n}: could not create the badge: {e:?}");
                    continue;
                }
            },
            None => continue,
        };

        // the screen it sits on
        let screen = screen_name.as_ref().and_then(|sn| {
            let sid = overlays
                .iter()
                .find(|(_, o)| {
                    o.config.is_active()
                        && matches!(o.config.category, OverlayCategory::Screen | OverlayCategory::WayVR | OverlayCategory::Mirror)
                        && o.config.name.as_ref() == sn.as_ref()
                })
                .map(|(id, _)| id)?;
            let s = overlays.mut_by_id(sid)?;
            let st = s.config.active_state.as_ref()?.transform;
            let meta = s.config.backend.frame_meta()?;
            let size = quad(&(st * meta.transform), meta.extent)?;
            Some((st, size))
        });

        let Some(b) = overlays.mut_by_id(id) else { continue };
        let on = want && screen.is_some();
        if !on {
            if b.config.is_active() {
                b.config.deactivate();
            }
            RT.lock().unwrap().applied[(n - 1) as usize] = None;
            continue;
        }
        let (st, size) = screen.unwrap();
        if !b.config.is_active() {
            b.config.activate(app, false);
        }
        let highlighted = hl == Some(n);
        let Some(meta) = b.config.backend.frame_meta() else { continue };
        let aspect = meta.extent[0] as f32 / meta.extent[1].max(1) as f32;
        let k = meta.transform.matrix3.col(1).length();
        let pose = badge_pose(&st, size, aspect, k, if highlighted { HIGHLIGHT_SCALE } else { 1.0 });
        if let Some(state) = b.config.active_state.as_mut() {
            state.positioning = Positioning::Static;
            state.transform = pose;
            b.config.dirty = true;
        }

        let prev = RT.lock().unwrap().applied[(n - 1) as usize];
        if prev != Some(highlighted) {
            let (pill, digit, name_c) = if highlighted {
                ("#FFC21AF8", "#101820", "#3A2A00")
            } else {
                ("#0B1420F0", "#FFFFFF", "#9FB4C8")
            };
            for (element, color) in [("pill", pill), ("num", digit), ("slot", name_c)] {
                let _ = b.config.backend.notify(
                    app,
                    OverlayEventData::CustomCommand { element: element.into(), command: ModifyPanelCommand::SetColor(color.into()) },
                );
            }
            RT.lock().unwrap().applied[(n - 1) as usize] = Some(highlighted);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbering_matches_pimax_slot_order() {
        let order = [WlxDeskSlot::Main, WlxDeskSlot::Left, WlxDeskSlot::Right, WlxDeskSlot::Up, WlxDeskSlot::UpLeft, WlxDeskSlot::UpRight];
        let nums: Vec<u8> = order.iter().map(|s| number_of(*s)).collect();
        assert_eq!(nums, [1, 2, 3, 4, 5, 6]);
        assert_eq!(slot_label(5), "up-left");
    }

    #[test]
    fn modes() {
        let t0 = Instant::now();
        let mut p = Plan::default();
        assert!(!p.visible(ScreenNumbers::OnDemand, t0));
        assert!(p.visible(ScreenNumbers::Always, t0));
        p.flash(t0, Some(4));
        assert!(p.visible(ScreenNumbers::OnDemand, t0 + Duration::from_secs(4)));
        assert!(!p.visible(ScreenNumbers::OnDemand, t0 + Duration::from_secs(6)));
        assert!(!p.visible(ScreenNumbers::Off, t0));
        assert_eq!(p.highlighted(t0 + Duration::from_secs(1)), Some(4));
        assert_eq!(p.highlighted(t0 + Duration::from_secs(3)), None);
        p.hidden = true;
        assert!(!p.visible(ScreenNumbers::Always, t0));
        p.flash(t0, None);
        assert!(p.visible(ScreenNumbers::Always, t0));
    }

    #[test]
    fn a_new_screen_flashes_but_a_repeat_does_not() {
        let t0 = Instant::now();
        let mut p = Plan::default();
        let one: Vec<(Arc<str>, u8)> = vec![("A".into(), 1)];
        p.note_screens(t0, one.clone(), false);
        assert!(p.visible(ScreenNumbers::OnDemand, t0 + Duration::from_secs(1)));
        let later = t0 + Duration::from_secs(10);
        p.note_screens(later, one.clone(), false);
        assert!(!p.visible(ScreenNumbers::OnDemand, later));
        p.note_screens(later, vec![("A".into(), 1), ("B".into(), 2)], false);
        assert!(p.visible(ScreenNumbers::OnDemand, later));
        assert_eq!(p.number_of_screen("B"), Some(2));
    }

    #[test]
    fn badge_sits_at_the_top_centre_of_its_screen() {
        // a 2 m x 1.125 m screen 2.5 m in front, facing +Z
        let screen = Affine3A::from_translation(glam::Vec3::new(0.0, 1.0, -2.5));
        let pose = badge_pose(&screen, (2.0, 1.125), 2.0, 1.0, 1.0);
        let c = pose.translation;
        assert!(c.x.abs() < 1e-5);
        assert!(c.y < 1.0 + 0.5625 && c.y > 1.0 + 0.3, "{c:?}");
        assert!((c.z + 2.5).abs() < 1e-5);
        // width = 15 % of the screen
        assert!((pose.matrix3.col(1).length() - 0.3).abs() < 1e-4);
        // a bigger highlighted badge is wider
        let hl = badge_pose(&screen, (2.0, 1.125), 2.0, 1.0, HIGHLIGHT_SCALE);
        assert!(hl.matrix3.col(1).length() > pose.matrix3.col(1).length());
    }
}
