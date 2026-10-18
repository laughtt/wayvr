use std::f32::consts::PI;
use std::process::{Child, Command};
use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::{Affine3A, Vec2, Vec3A, Vec3Swizzles};

use idmap_derive::IntegerId;
use smallvec::{SmallVec, smallvec};
use strum::{AsRefStr, EnumIs};
use wayvr_ipc::packet_client::{HandsfreeAction, HandsfreeParams};
use wlx_common::common::LeftRight;
use wlx_common::windowing::{OverlayWindowState, Positioning};

use crate::backend::gaze::GazeState;
use crate::backend::task::{InputTask, OverlayTask};
use crate::overlays::anchor::{ANCHOR_NAME, GRAB_HELP_NAME};
use crate::overlays::watch::WATCH_NAME;
use crate::state::{AppSession, AppState};
use crate::subsystem::hid::WheelDelta;
use crate::subsystem::input::InputFocus;
use crate::windowing::backend::OverlayEventData;
use crate::windowing::grabmath;
use crate::windowing::manager::OverlayWindowManager;
use crate::windowing::window::{self, OverlayCategory, OverlayWindowData, realign, scalar_scale};
use crate::windowing::{OverlayID, OverlaySelector};

use super::task::TaskType;

#[derive(Clone, Default)]
pub struct HoverResult {
    pub haptics: Option<Haptics>,
    /// If true, the laser shows at this position and no further raycasting will be done.
    pub consume: bool,
}

impl HoverResult {
    pub fn consume() -> Self {
        HoverResult {
            consume: true,
            ..Default::default()
        }
    }
}

pub struct TrackedDevice {
    pub soc: Option<f32>,
    pub charging: bool,
    pub role: TrackedDeviceRole,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, IntegerId, AsRefStr)]
pub enum TrackedDeviceRole {
    None,
    Hmd,
    LeftHand,
    RightHand,
    Tracker,
}

#[derive(Debug, Eq, PartialEq, Clone, Copy, EnumIs)]
pub enum FocusPickState {
    None,
    /// aiming to pick the focus
    Aiming,
    /// picking the focus. this will be consumed on valid on_hover
    Picking,
}

pub struct InputState {
    pub hmd: Affine3A,
    pub ipd: f32,
    pub pointers: [Pointer; 2],
    pub devices: Vec<TrackedDevice>,
    pub handsfree_state: PointerState,
    pub gaze: GazeState,
    /// Steady clicks per pointer slot (`click_steady_px`).
    steady: [crate::backend::click_steady::Steady; 2],
    /// The gaze pointer's own steady state; swapped with the slot's while the gaze borrows a controller slot.
    gaze_steady: crate::backend::click_steady::Steady,
    pub picking_focus: FocusPickState,
    processes: Vec<Child>,
    disable_lerp_until: Instant,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            hmd: Affine3A::IDENTITY,
            ipd: 0.0,
            pointers: [Pointer::new(0), Pointer::new(1)],
            devices: Vec::new(),
            processes: Vec::new(),
            handsfree_state: PointerState::default(),
            gaze: GazeState::new(),
            steady: Default::default(),
            gaze_steady: Default::default(),
            picking_focus: FocusPickState::None,
            disable_lerp_until: Instant::now(),
        }
    }

    pub fn should_disable_lerp(&self) -> bool {
        self.disable_lerp_until > Instant::now()
    }

    pub fn stop_picking(&mut self) {
        if !self.picking_focus.is_none() {
            self.picking_focus = FocusPickState::None;
            self.disable_lerp_until = Instant::now() + Duration::from_millis(250);
        }
    }

    pub fn apply_handsfree_action(&mut self, params: HandsfreeParams) {
        const fn set_true(v: &mut bool) {
            *v = true;
        }
        const fn set_false(v: &mut bool) {
            *v = false;
        }
        const fn toggle(v: &mut bool) {
            *v = !*v;
        }

        let (action, apply) = match params {
            HandsfreeParams::SetMode(_) => {
                return;
            }
            HandsfreeParams::Press(action) => (action, set_true as fn(&mut bool)),
            HandsfreeParams::Release(action) => (action, set_false as fn(&mut bool)),
            HandsfreeParams::Toggle(action) => (action, toggle as fn(&mut bool)),
            HandsfreeParams::Scroll(val) => {
                self.handsfree_state.scroll_y = val;
                return;
            }
        };

        match action {
            HandsfreeAction::Click => apply(&mut self.handsfree_state.click),
            HandsfreeAction::RightModifier => apply(&mut self.handsfree_state.click_modifier_right),
            HandsfreeAction::MiddleModifier => {
                apply(&mut self.handsfree_state.click_modifier_middle);
            }
            HandsfreeAction::Grab => apply(&mut self.handsfree_state.grab),
        }
    }

    pub fn handle_task(&mut self, task: InputTask) {
        match task {
            InputTask::Haptics { device, haptics } => {
                if let Some(pointer) = self.pointers.get_mut(device) {
                    pointer.pending_haptics = Some(haptics);
                } else {
                    log::warn!("Can't trigger haptics on non-existing device: {device}");
                }
            }
        }
    }

    pub const fn pre_update(&mut self) {
        self.pointers[0].before = self.pointers[0].now;
        self.pointers[1].before = self.pointers[1].now;
    }

    pub fn post_update(&mut self, session: &AppSession) {
        for hand in &mut self.pointers {
            #[cfg(debug_assertions)]
            debug_print_hand(hand);

            if hand.now.click {
                hand.last_click = Instant::now();
            }

            // Prevent the mode from changing during a click
            if !hand.before.click {
                if hand.now.click_modifier_right {
                    hand.interaction.mode = PointerMode::Right;
                    continue;
                }

                if hand.now.click_modifier_middle {
                    hand.interaction.mode = PointerMode::Middle;
                    continue;
                }

                let hmd_up = self.hmd.transform_vector3a(Vec3A::Y);
                let dot = hmd_up.dot(hand.pose.transform_vector3a(Vec3A::X))
                    * 2.0f32.mul_add(-(hand.idx as f32), 1.0);

                hand.interaction.mode = if dot < -0.85 {
                    PointerMode::Right
                } else if dot > 0.7 {
                    PointerMode::Middle
                } else {
                    PointerMode::Left
                };

                let middle_click_orientation = false;
                let right_click_orientation = false;
                match hand.interaction.mode {
                    PointerMode::Middle => {
                        if !middle_click_orientation {
                            hand.interaction.mode = PointerMode::Left;
                        }
                    }
                    PointerMode::Right if !right_click_orientation => {
                        hand.interaction.mode = PointerMode::Left;
                    }
                    _ => {}
                }
            }

            if hand.now.alt_click != hand.before.alt_click {
                // Reap previous processes
                self.processes
                    .retain_mut(|child| !matches!(child.try_wait(), Ok(Some(_))));

                let mut args = if hand.now.alt_click {
                    session.config.alt_click_down.iter()
                } else {
                    session.config.alt_click_up.iter()
                };

                if let Some(program) = args.next()
                    && let Ok(child) = Command::new(program).args(args).spawn()
                {
                    self.processes.push(child);
                }
            }
        }
    }
}

#[cfg(debug_assertions)]
fn debug_print_hand(hand: &Pointer) {
    {
        if hand.now.click != hand.before.click {
            log::debug!("Hand {}: click {}", hand.idx, hand.now.click);
        }
        if hand.now.grab != hand.before.grab {
            log::debug!("Hand {}: grab {}", hand.idx, hand.now.grab);
        }
        if hand.now.grab_float != hand.before.grab_float {
            log::debug!("Hand {}: grab {}", hand.idx, hand.now.grab);
        }
        if hand.now.alt_click != hand.before.alt_click {
            log::debug!("Hand {}: alt_click {}", hand.idx, hand.now.alt_click);
        }
        if hand.now.show_hide != hand.before.show_hide {
            log::debug!("Hand {}: show_hide {}", hand.idx, hand.now.show_hide);
        }
        if hand.now.toggle_dashboard != hand.before.toggle_dashboard {
            log::debug!(
                "Hand {}: toggle_dashboard {}",
                hand.idx,
                hand.now.toggle_dashboard
            );
        }
        if hand.now.menu != hand.before.menu {
            log::debug!("Hand {}: menu {}", hand.idx, hand.now.menu);
        }
        if hand.now.space_drag != hand.before.space_drag {
            log::debug!("Hand {}: space_drag {}", hand.idx, hand.now.space_drag);
        }
        if hand.now.space_rotate != hand.before.space_rotate {
            log::debug!("Hand {}: space_rotate {}", hand.idx, hand.now.space_rotate);
        }
        if hand.now.space_reset != hand.before.space_reset {
            log::debug!("Hand {}: space_reset {}", hand.idx, hand.now.space_reset);
        }
        if hand.now.click_modifier_right != hand.before.click_modifier_right {
            log::debug!(
                "Hand {}: click_modifier_right {}",
                hand.idx,
                hand.now.click_modifier_right
            );
        }
        if hand.now.click_modifier_middle != hand.before.click_modifier_middle {
            log::debug!(
                "Hand {}: click_modifier_middle {}",
                hand.idx,
                hand.now.click_modifier_middle
            );
        }
    }
}

pub struct InteractionState {
    pub mode: PointerMode,
    pub grabbed: Option<GrabData>,
    pub clicked_id: Option<OverlayID>,
    pub hovered_id: Option<OverlayID>,
    pub should_block_input: bool,
    pub should_block_poses: bool,
    pub kbd_block_activated: bool,
}

impl Default for InteractionState {
    fn default() -> Self {
        Self {
            mode: PointerMode::Left,
            grabbed: None,
            clicked_id: None,
            hovered_id: None,
            should_block_input: false,
            should_block_poses: false,
            kbd_block_activated: false,
        }
    }
}

pub struct Pointer {
    pub idx: usize,
    pub pose: Affine3A,
    pub raw_pose: Affine3A,
    pub now: PointerState,
    pub before: PointerState,
    pub last_click: Instant,
    pub pending_haptics: Option<Haptics>,
    pub(super) interaction: InteractionState,
    pub tracked: bool,
    pub handsfree: bool,
}

impl Pointer {
    pub fn new(idx: usize) -> Self {
        debug_assert!(idx == 0 || idx == 1);
        Self {
            idx,
            pose: Affine3A::IDENTITY,
            raw_pose: Affine3A::IDENTITY,
            now: PointerState::default(),
            before: PointerState::default(),
            last_click: Instant::now(),
            pending_haptics: None,
            interaction: InteractionState::default(),
            tracked: false,
            handsfree: false,
        }
    }

    pub const fn hand(&self) -> Option<LeftRight> {
        match self.idx {
            0 => Some(LeftRight::Left),
            1 => Some(LeftRight::Right),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PointerState {
    pub scroll_x: f32,
    pub scroll_y: f32,
    pub click: bool,
    pub grab: bool,
    pub grab_float: bool,
    pub alt_click: bool,
    pub show_hide: bool,
    pub toggle_dashboard: bool,
    /// Work menu (`menu` binding): rising edge opens or closes the `menu_panel` overlay.
    pub menu: bool,
    pub space_drag: bool,
    pub space_rotate: bool,
    pub space_reset: bool,
    pub pin: bool,
    pub reset_screens: bool,
    pub click_modifier_right: bool,
    pub click_modifier_middle: bool,
    pub move_mouse: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PointerHit {
    pub pointer: usize,
    pub overlay: OverlayID,
    pub mode: PointerMode,
    pub primary: bool,
    pub uv: Vec2,
}

#[derive(Clone, Copy)]
pub struct Haptics {
    pub intensity: f32,
    pub duration: f32,
    pub frequency: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct RayHit {
    overlay: OverlayID,
    global_pos: Vec3A,
    local_pos: Vec2,
    dist: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GrabData {
    pub offset: Affine3A,
    pub grabbed_id: OverlayID,
    pub grab_anchor: bool,
    /// two-hand gesture state (world_lock_grab), kept on the lower-index hand
    pub two: Option<TwoHand>,
    /// this hand was part of a two-hand grab and must re-base its offset when alone again
    pub was_two: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct TwoHand {
    pub v0: Vec3A,
    pub m0: Vec3A,
    pub t0: Affine3A,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, Default)]
pub enum PointerMode {
    #[default]
    Left,
    Right,
    Middle,
    Special,
}

pub struct PointerLine {
    pub mode: PointerMode,
    pub a: Vec3A,
    pub b: Vec3A,
}

fn populate_lines(
    lines: &mut Vec<PointerLine>,
    pointer: &Pointer,
    maybe_hit: Option<(PointerHit, RayHit)>,
    hmd: &Affine3A,
) {
    let Some((hit, raw_hit)) = maybe_hit else {
        return;
    };

    if pointer.handsfree {
        const HALF_SIZE: f32 = 0.01;

        // horizontal arm
        lines.push(PointerLine {
            mode: hit.mode,
            a: raw_hit.global_pos - (hmd.x_axis * HALF_SIZE),
            b: raw_hit.global_pos + (hmd.x_axis * HALF_SIZE),
        });

        // vertical arm
        lines.push(PointerLine {
            mode: PointerMode::Special,
            a: raw_hit.global_pos - (hmd.y_axis * HALF_SIZE),
            b: raw_hit.global_pos + (hmd.y_axis * HALF_SIZE),
        });
    } else {
        lines.push(PointerLine {
            mode: hit.mode,
            a: pointer.pose.translation,
            b: raw_hit.global_pos,
        });
    }
}

fn update_focus(app: &mut AppState, overlay_input_focus: Option<InputFocus>) {
    if let Some(f) = &overlay_input_focus
        && app.input_state.picking_focus.is_none()
        && app
            .hid_provider
            .set_input_focus(app.wvr_server.as_mut(), *f)
    {
        log::debug!("Setting keyboard focus to {:?}", *f);
    }
}

pub fn interact<O>(
    overlays: &mut OverlayWindowManager<O>,
    app: &mut AppState,
    lines: &mut Vec<PointerLine>,
) -> [Option<Haptics>; 2]
where
    O: Default,
{
    // gaze has priority: controllers do not hover (unless they hold a grab); only with gaze input on (gaze::gate)
    let gaze_first = {
        use crate::backend::gaze::gate;
        let input = &app.input_state;
        let cfg = &app.session.config;
        gate::over_controllers(cfg.gaze_pointer, cfg.gaze_over_controllers)
            && gate::may_point(
                cfg.gaze_pointer,
                input.gaze.active,
                !input.picking_focus.is_none(),
                input.pointers.iter().any(|p| p.handsfree),
            )
    };
    let hand = |idx: usize, overlays: &mut OverlayWindowManager<O>, app: &mut AppState| {
        if gaze_first && app.input_state.pointers[idx].interaction.grabbed.is_none() {
            let pointer = &mut app.input_state.pointers[idx];
            let pending_haptics = pointer.pending_haptics.take();
            let hovered = pointer.interaction.hovered_id.take();
            handle_no_hit(idx, hovered, overlays, app);
            (None, pending_haptics)
        } else {
            interact_hand(idx, overlays, app)
        }
    };
    let hits = if app.input_state.pointers[1].last_click > app.input_state.pointers[0].last_click {
        let right = hand(1, overlays, app);
        let left = hand(0, overlays, app);
        [left, right]
    } else {
        let left = hand(0, overlays, app);
        let right = hand(1, overlays, app);
        [left, right]
    };

    for (idx, hit) in hits.iter().enumerate() {
        populate_lines(
            lines,
            &app.input_state.pointers[idx],
            hit.0,
            &app.input_state.hmd,
        );
    }

    learn_from_lasers(overlays, app, &hits);
    interact_gaze(overlays, app, lines, &hits);
    track_gaze_silent(overlays, app);
    {
        let cfg = &app.session.config;
        let tracking = cfg.gaze_target || cfg.gaze_laser || cfg.gaze_pointer;
        app.input_state.gaze.poll_shared(tracking, Instant::now());
    }

    [hits[0].1, hits[1].1]
}

/// Passive gaze lessons: a controller laser that clicks down on a screen tells where the eyes should have been.
/// Allowed even with `gaze_pointer` off because it only learns; it never sets a click or moves anything.
fn learn_from_lasers<O>(
    overlays: &mut OverlayWindowManager<O>,
    app: &mut AppState,
    hits: &[HandHit; 2],
) where
    O: Default,
{
    let input = &app.input_state;
    if !input.gaze.learn || !input.gaze.active || !input.gaze.fresh {
        return;
    }
    if input
        .pointers
        .iter()
        .any(|p| p.interaction.grabbed.is_some() || p.now.grab || p.handsfree)
    {
        return;
    }
    let now = Instant::now();
    for (idx, h) in hits.iter().enumerate() {
        let p = &app.input_state.pointers[idx];
        if !p.now.click || p.before.click {
            continue;
        }
        let Some((hit, raw_hit)) = h.0 else {
            continue;
        };
        let on_screen = overlays
            .mut_by_id(hit.overlay)
            .is_some_and(|o| o.config.category == OverlayCategory::Screen);
        if on_screen {
            app.input_state
                .gaze
                .record_laser(raw_hit.global_pos.into(), now);
        }
    }
}

/// Silent gaze tracking: ray-cast the filtered gaze ray against the overlays WITHOUT hover, click or mouse
/// moves, feed the fixation tracker (`wayvrctl gaze`) and place the laser marker. Runs with `gaze_target`,
/// `gaze_laser` or `gaze_pointer` on; the laser shows only on a fresh (non-blink) sample that hits an overlay.
fn track_gaze_silent<O>(overlays: &mut OverlayWindowManager<O>, app: &mut AppState)
where
    O: Default,
{
    use crate::backend::gaze_target::{GazeSample, laser_segment, unix_ms};

    let cfg = &app.session.config;
    let tracking = cfg.gaze_target || cfg.gaze_laser || cfg.gaze_pointer;
    let (laser_on, style, min_ms) = (cfg.gaze_laser, cfg.gaze_laser_style, cfg.gaze_fixation_ms);
    let gaze = &mut app.input_state.gaze;
    gaze.tracking = tracking;
    // targets for `wayvrctl gaze` (voice "this") only with gaze input on: by default the gaze is a visual point
    gaze.reporting = crate::backend::gaze::gate::reporting(cfg.gaze_pointer);
    gaze.laser_on = laser_on;
    gaze.laser = None;
    if !tracking {
        gaze.target.reset();
        return;
    }
    let now_ms = unix_ms();
    if !(gaze.active && gaze.fresh) {
        gaze.target.update(now_ms, None, false, min_ms);
        return;
    }
    let pose = gaze.ptr.pose;

    let mut hits: SmallVec<[RayHit; 8]> = smallvec!();
    for (id, overlay) in overlays.iter() {
        let Some(state) = overlay.config.active_state.as_ref() else {
            continue;
        };
        if !state.interactable {
            continue;
        }
        if let Some(hit) = ray_test(&pose, id, &state.transform, state.curvature.as_ref())
            && hit.dist.is_finite()
        {
            hits.push(hit);
        }
    }
    hits.sort_by(|a, b| a.dist.total_cmp(&b.dist));

    let mut sample = None;
    for hit in hits {
        let Some(overlay) = overlays.mut_by_id(hit.overlay) else {
            continue;
        };
        let backend = overlay.config.backend.as_mut();
        let Some(uv) = backend
            .get_interaction_transform()
            .map(|a| a.transform_point2(hit.local_pos))
        else {
            continue;
        };
        if !(0.0..=1.0).contains(&uv.x) || !(0.0..=1.0).contains(&uv.y) {
            continue;
        }
        let px = backend.uv_to_desktop(uv);
        sample = Some((
            GazeSample {
                overlay: overlay.config.name.clone(),
                uv,
                px,
                dir: (hit.global_pos - pose.translation).normalize_or_zero().into(),
                dist: hit.dist,
            },
            hit.global_pos,
        ));
        break;
    }

    let hmd = app.input_state.hmd;
    let gaze = &mut app.input_state.gaze;
    if laser_on && let Some((_, at)) = sample.as_ref() {
        gaze.laser = Some(laser_segment(style, pose.translation, *at, &hmd));
    }
    gaze.target.update(now_ms, sample.map(|(s, _)| s), true, min_ms);
}

/// Swap the gaze pointer state with the controller pointer in `slot`. Calling it twice restores both.
const fn swap_gaze(app: &mut AppState, slot: usize) {
    let input = &mut app.input_state;
    let (p, g) = (&mut input.pointers[slot], &mut input.gaze.ptr);
    std::mem::swap(&mut p.pose, &mut g.pose);
    std::mem::swap(&mut p.raw_pose, &mut g.raw_pose);
    std::mem::swap(&mut p.now, &mut g.now);
    std::mem::swap(&mut p.before, &mut g.before);
    std::mem::swap(&mut p.interaction, &mut g.interaction);
    std::mem::swap(&mut p.tracked, &mut g.tracked);
    std::mem::swap(&mut p.handsfree, &mut g.handsfree);
    std::mem::swap(&mut input.steady[slot], &mut input.gaze_steady);
}

/// Stop gaze hovering: send on_left and the click release (if any) to whatever the gaze was on.
fn release_gaze<O>(overlays: &mut OverlayWindowManager<O>, app: &mut AppState)
where
    O: Default,
{
    let Some(slot) = app.input_state.gaze.slot.take() else {
        return;
    };
    let gaze = &mut app.input_state.gaze.ptr;
    gaze.before = gaze.now;
    gaze.now = PointerState::default();
    let hovered = gaze.interaction.hovered_id.take();
    let clicked = gaze.interaction.clicked_id;
    gaze.interaction.should_block_input = false;
    gaze.interaction.should_block_poses = false;
    gaze.interaction.kbd_block_activated = false;

    if hovered.is_none() && clicked.is_none() {
        return;
    }
    // The controller in that slot took over the same overlay this frame: nothing to leave.
    let ctrl_hover = app.input_state.pointers[slot].interaction.hovered_id;
    if hovered.is_some() && hovered == ctrl_hover {
        app.input_state.gaze.ptr.interaction.clicked_id = None;
        return;
    }
    swap_gaze(app, slot);
    handle_no_hit(slot, hovered, overlays, app);
    swap_gaze(app, slot);
}

type HandHit = (Option<(PointerHit, RayHit)>, Option<Haptics>);

fn interact_gaze<O>(
    overlays: &mut OverlayWindowManager<O>,
    app: &mut AppState,
    lines: &mut Vec<PointerLine>,
    hits: &[HandHit; 2],
) where
    O: Default,
{
    let input = &app.input_state;
    let usable = crate::backend::gaze::gate::may_point(
        app.session.config.gaze_pointer,
        input.gaze.active,
        !input.picking_focus.is_none(),
        input.pointers.iter().any(|p| p.handsfree),
    );

    // a controller slot is free if its ray hits nothing and it is not grabbing
    let free = |i: usize| hits[i].0.is_none() && input.pointers[i].interaction.grabbed.is_none();
    let slot = input
        .gaze
        .slot
        .filter(|&s| free(s))
        .or_else(|| (0..2).find(|&i| free(i)));

    let buttons = [app.input_state.pointers[0].now.click, app.input_state.pointers[1].now.click];
    let Some(slot) = slot.filter(|_| usable) else {
        release_gaze(overlays, app);
        app.input_state.gaze.dwell.reset();
        // drop the refine state but remember the raw buttons: a trigger held across a gaze drop makes no tap
        app.input_state.gaze.refine.observe(buttons);
        app.input_state.gaze.last_hit = None;
        return;
    };

    if app.input_state.gaze.slot.is_some_and(|s| s != slot) {
        release_gaze(overlays, app);
    }

    // "look and click": a controller click that is not aimed at an overlay clicks where we look
    let input = &app.input_state;
    let trigger = (0..2).any(|i| input.pointers[i].now.click && hits[i].0.is_none());
    let (gaze_pointer, gaze_refine) = (app.session.config.gaze_pointer, app.session.config.gaze_refine);
    let dwell_ms = crate::backend::gaze::gate::dwell_ms(gaze_pointer, input.gaze.dwell_ms);
    let last_hit = input.gaze.last_hit;
    let now = Instant::now();

    // hold-to-refine click (gate::refine): the trigger freezes the dot, the head refines it
    let refine_out = if crate::backend::gaze::gate::refine(gaze_pointer, gaze_refine) {
        let (_, hmd_rot, _) = input.hmd.to_scale_rotation_translation();
        let free = [hits[0].0.is_none(), hits[1].0.is_none()];
        let g = &mut app.input_state.gaze;
        let (cur, look, eye) = (g.current_dir(), g.look(now), g.eye_pos);
        // the RAW buttons decide: a ray sweeping onto an overlay during a Hold is not a release
        g.refine.update(buttons, free, now, hmd_rot, eye, cur, look)
    } else {
        app.input_state.gaze.refine.observe(buttons);
        crate::backend::gaze::RefineOut::default()
    };

    let mut dwell = app.input_state.gaze.dwell.update(last_hit, now, dwell_ms);
    if app.input_state.gaze.refine.active() {
        dwell = false;
        app.input_state.gaze.dwell.reset();
    }

    let gaze = &mut app.input_state.gaze.ptr;
    gaze.before = gaze.now;
    gaze.now = PointerState::default();
    gaze.now.click =
        crate::backend::gaze::gate::pointer_click(gaze_pointer, gaze_refine, trigger, dwell, refine_out.click);
    gaze.tracked = true;
    gaze.handsfree = false;
    if let Some(d) = refine_out.dir {
        let origin = glam::Vec3::from(gaze.pose.translation);
        gaze.pose = Affine3A::from_rotation_translation(glam::Quat::from_rotation_arc(glam::Vec3::NEG_Z, d), origin);
    }
    app.input_state.gaze.slot = Some(slot);

    swap_gaze(app, slot);
    let (hit, _) = interact_hand(slot, overlays, app);
    swap_gaze(app, slot);

    app.input_state.gaze.last_hit = hit.map(|(h, raw)| (h.overlay, raw.global_pos));
    if let (Some(lesson), Some((_, raw_hit))) = (refine_out.lesson, hit) {
        app.input_state
            .gaze
            .record_refine(&lesson, raw_hit.global_pos.into());
    }

    if let Some((hit, raw_hit)) = hit {
        // small plus-shaped cursor facing the HMD
        const HALF_SIZE: f32 = 0.0105;
        let hmd = &app.input_state.hmd;
        for axis in [hmd.x_axis, hmd.y_axis] {
            lines.push(PointerLine {
                mode: if app.input_state.gaze.ptr.now.click {
                    PointerMode::Special
                } else {
                    hit.mode
                },
                a: raw_hit.global_pos - (axis * HALF_SIZE),
                b: raw_hit.global_pos + (axis * HALF_SIZE),
            });
        }
    }
}

fn interact_hand<O>(
    idx: usize,
    overlays: &mut OverlayWindowManager<O>,
    app: &mut AppState,
) -> (Option<(PointerHit, RayHit)>, Option<Haptics>)
where
    O: Default,
{
    let mut pointer = &mut app.input_state.pointers[idx];
    let pending_haptics = pointer.pending_haptics.take();

    if !pointer.tracked {
        pointer.interaction.should_block_input = false;
        pointer.interaction.should_block_poses = false;
        pointer.interaction.kbd_block_activated = false;
        return (None, pending_haptics); // no hit
    }

    // already grabbing, ignore everything else
    if let Some(grab_data) = pointer.interaction.grabbed {
        if let Some(grabbed) = overlays.mut_by_id(grab_data.grabbed_id) {
            handle_grabbed(idx, grabbed, app);
        } else {
            log::warn!("Grabbed overlay {:?} does not exist", grab_data.grabbed_id);
            pointer.interaction.grabbed = None;
        }
        return (None, pending_haptics);
    }

    let hovered_id = pointer.interaction.hovered_id.take();
    let (Some((mut hit, raw_hit)), haptics) = get_nearest_hit(idx, overlays, app) else {
        handle_no_hit(idx, hovered_id, overlays, app);
        return (None, pending_haptics); // no hit
    };

    // focus change
    if let Some(hovered_id) = hovered_id
        && hovered_id != hit.overlay
        && let Some(old_hovered) = overlays.mut_by_id(hovered_id)
    {
        if old_hovered.primary_pointer.is_some_and(|i| i == idx) {
            old_hovered.primary_pointer = None;
        }
        log::debug!("{} on_left (focus changed)", old_hovered.config.name);
        old_hovered.config.backend.on_left(app, idx);
        old_hovered.hover_pointers[idx] = false;
        if !old_hovered.hover_pointers.iter().any(|x| *x) {
            overlays.edit_overlay(hovered_id, false, app);
        }
    }

    overlays.edit_overlay(hit.overlay, true, app);
    let edit_mode = overlays.get_edit_mode();

    let Some(hovered) = overlays.mut_by_id(hit.overlay) else {
        log::warn!("Hit overlay {:?} does not exist", hit.overlay);
        return (None, pending_haptics); // no hit
    };
    pointer = &mut app.input_state.pointers[idx];
    pointer.interaction.hovered_id = Some(hit.overlay);
    hovered.hover_pointers[idx] = true;

    if let Some(primary_pointer) = hovered.primary_pointer {
        if hit.pointer <= primary_pointer {
            hovered.primary_pointer = Some(hit.pointer);
            hit.primary = true;
        }
    } else {
        hovered.primary_pointer = Some(hit.pointer);
        hit.primary = true;
    }

    if let Some(state) = hovered.config.active_state.as_ref() {
        pointer.interaction.should_block_input = state.block_input
            && (hovered.config.name.as_ref() != WATCH_NAME
                || !app.session.config.block_game_input_ignore_watch);

        pointer.interaction.should_block_poses = state.block_input
            && app.session.config.block_poses_on_kbd_interaction
            && hovered.config.category == OverlayCategory::Keyboard
            && pointer.interaction.kbd_block_activated;
    } else {
        pointer.interaction.should_block_input = false;
        pointer.interaction.should_block_poses = false;
    }

    #[cfg(debug_assertions)]
    log::trace!("Hit: {} {:?}", hovered.config.name, hit);

    let hovered_state = hovered.config.active_state.as_mut().unwrap();

    let grab_float = pointer.now.grab_float;
    let grab_start = pointer.now.grab && !pointer.before.grab;

    // pin / unpin (desk mode): toggles grabbable on the hovered overlay
    if app.session.config.world_lock_grab
        && app.session.config.world_lock_pin
        && pointer.now.pin
        && !pointer.before.pin
        && !matches!(hovered.config.category, OverlayCategory::Internal)
    {
        hovered_state.grabbable = !hovered_state.grabbable;
        hovered.config.dirty = true;
        log::info!(
            "Hand {}: {} {}",
            idx,
            if hovered_state.grabbable { "unpinned" } else { "pinned" },
            hovered.config.name
        );
        let intensity = if hovered_state.grabbable { 0.2 } else { 0.7 };
        return (
            Some((hit, raw_hit)),
            Some(Haptics {
                intensity,
                duration: 0.15,
                frequency: 0.2,
            }),
        );
    }

    // grab (locked layouts: never a desk screen; the pointer still clicks and scrolls it)
    if grab_start
        && hovered_state.grabbable
        && crate::windowing::desklock::locked(app.session.config.desk_locked, hovered.config.category)
    {
        log::debug!("Hand {idx}: {} is locked (desk_locked), not grabbed", hovered.config.name);
    }
    if grab_start
        && hovered_state.grabbable
        && !crate::windowing::desklock::locked(app.session.config.desk_locked, hovered.config.category)
    {
        update_focus(app, hovered.config.input_focus);
        start_grab(StartGrabParams {
            idx,
            id: hit.overlay,
            name: hovered.config.name.clone(),
            editing: hovered.config.editing,
            state: hovered_state,
            app,
            edit_mode,
            grab_float,
        });
        log::debug!("Hand {}: grabbed {}", hit.pointer, hovered.config.name);
        return (
            Some((hit, raw_hit)),
            Some(Haptics {
                intensity: 0.25,
                duration: 0.1,
                frequency: 0.1,
            }),
        );
    }

    handle_scroll(&hit, hovered, app);

    // click / release
    let pointer = &mut app.input_state.pointers[hit.pointer];
    if pointer.now.click && !pointer.before.click {
        pointer.interaction.clicked_id = Some(hit.overlay);
        if hovered.config.category == OverlayCategory::Keyboard {
            pointer.interaction.kbd_block_activated = true;
        }
        update_focus(app, hovered.config.input_focus);
        hovered.config.backend.on_pointer(app, &hit, true);
    } else if !pointer.now.click && pointer.before.click {
        // send release event to overlay that was originally clicked
        if let Some(clicked_id) = pointer.interaction.clicked_id.take() {
            if let Some(clicked) = overlays.mut_by_id(clicked_id) {
                clicked.config.backend.on_pointer(app, &hit, false);
            }
        } else {
            hovered.config.backend.on_pointer(app, &hit, false);
        }
    }

    (Some((hit, raw_hit)), haptics.or(pending_haptics))
}

fn handle_no_hit<O>(
    pointer_idx: usize,
    hovered_id: Option<OverlayID>,
    overlays: &mut OverlayWindowManager<O>,
    app: &mut AppState,
) {
    if let Some(hovered_id) = hovered_id
        && let Some(hovered) = overlays.mut_by_id(hovered_id)
    {
        log::debug!("{} on_left (no hit)", hovered.config.name);
        hovered.config.backend.on_left(app, pointer_idx);
        hovered.hover_pointers[pointer_idx] = false;
        if !hovered.hover_pointers.iter().any(|x| *x) {
            overlays.edit_overlay(hovered_id, false, app);
        }
    }

    let pointer = &mut app.input_state.pointers[pointer_idx];
    pointer.interaction.should_block_input = false;
    pointer.interaction.should_block_poses = false;
    pointer.interaction.kbd_block_activated = false;

    // in case click released while not aiming at anything
    // send release event to overlay that was originally clicked
    if !pointer.now.click
        && pointer.before.click
        && let Some(clicked_id) = pointer.interaction.clicked_id.take()
        && let Some(clicked) = overlays.mut_by_id(clicked_id)
    {
        let hit = PointerHit {
            pointer: pointer.idx,
            overlay: clicked_id,
            mode: pointer.interaction.mode,
            ..Default::default()
        };
        clicked.config.backend.on_pointer(app, &hit, false);
    }
}

fn handle_scroll<O>(hit: &PointerHit, hovered: &mut OverlayWindowData<O>, app: &mut AppState) {
    let pointer = &mut app.input_state.pointers[hit.pointer];
    if pointer.now.scroll_x.abs() <= 0.1 && pointer.now.scroll_y.abs() <= 0.1 {
        return;
    }

    let config = &app.session.config;
    let scroll_factor = 90.0 * app.delta_time;

    let scroll_x = pointer.now.scroll_x
        * scroll_factor
        * config.scroll_speed
        * if config.invert_scroll_direction_x {
            -1.0
        } else {
            1.0
        };
    let scroll_y = pointer.now.scroll_y
        * scroll_factor
        * config.scroll_speed
        * if config.invert_scroll_direction_x {
            -1.0
        } else {
            1.0
        };

    if app.input_state.pointers[1 - hit.pointer]
        .interaction
        .grabbed
        .is_some_and(|x| x.grabbed_id == hit.overlay)
    {
        let can_curve = hovered
            .frame_meta()
            .is_some_and(|e| e.extent[0] >= e.extent[1]);

        // re-borrow
        let hovered_state = hovered.config.active_state.as_mut().unwrap();
        if can_curve {
            let cur = hovered_state.curvature.unwrap_or(0.0);
            let new = scroll_y.mul_add(-0.01, cur).min(0.5);
            if new <= f32::EPSILON {
                hovered_state.curvature = None;
            } else {
                hovered_state.curvature = Some(new);
            }
        } else {
            hovered_state.curvature = None;
        }
    } else {
        hovered.config.backend.on_scroll(
            app,
            hit,
            WheelDelta {
                x: scroll_x,
                y: scroll_y,
            },
        );
    }
}

fn get_nearest_hit<O>(
    pointer_idx: usize,
    overlays: &mut OverlayWindowManager<O>,
    app: &mut AppState,
) -> (Option<(PointerHit, RayHit)>, Option<Haptics>)
where
    O: Default,
{
    let pointer = &mut app.input_state.pointers[pointer_idx];
    if !pointer.tracked {
        return (None, None);
    }

    let ray_origin = pointer.pose;
    let mode = pointer.interaction.mode;
    let held = pointer.now.click || pointer.before.click;
    let down_edge = pointer.now.click && !pointer.before.click;
    let steady_px = app.session.config.click_steady_px;
    if !held {
        app.input_state.steady[pointer_idx].reset();
    }
    let edit_mode = overlays.get_edit_mode();

    let mut hits: SmallVec<[RayHit; 8]> = smallvec!();

    for (id, overlay) in overlays.iter() {
        let Some(overlay_state) = overlay.config.active_state.as_ref() else {
            continue;
        };
        if !overlay_state.interactable && !edit_mode {
            continue;
        }

        if let Some(hit) = ray_test(
            &ray_origin,
            id,
            &overlay_state.transform,
            overlay_state.curvature.as_ref(),
        ) && hit.dist.is_finite()
        {
            hits.push(hit);
        }
    }

    hits.sort_by(|a, b| a.dist.total_cmp(&b.dist));

    for hit in hits {
        let overlay = overlays.mut_by_id(hit.overlay).unwrap(); // safe because we just got the id from the overlay

        let Some(uv) = overlay
            .config
            .backend
            .as_mut()
            .get_interaction_transform()
            .map(|a| a.transform_point2(hit.local_pos))
        else {
            continue;
        };

        // steady click: a held click keeps reporting its press point while the real hit stays near it
        let mut steady = app.input_state.steady[pointer_idx];
        let uv = if held && steady_px > 0.0 {
            let backend = overlay.config.backend.as_mut();
            let px_per_uv = match (backend.uv_to_desktop(Vec2::ZERO), backend.uv_to_desktop(Vec2::ONE)) {
                (Some(a), Some(b)) => (b - a).abs(),
                _ => backend
                    .frame_meta()
                    .map_or(Vec2::ZERO, |m| Vec2::new(m.extent[0] as f32, m.extent[1] as f32)),
            };
            steady.apply(down_edge, held, hit.overlay, uv, px_per_uv, steady_px, Instant::now())
        } else {
            uv
        };

        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) && !overlay.config.resizing {
            continue;
        }

        let pointer_hit = PointerHit {
            pointer: pointer_idx,
            overlay: hit.overlay,
            mode,
            primary: false,
            uv,
        };

        let result = overlay.config.backend.on_hover(app, &pointer_hit);
        if result.consume || overlay.config.editing {
            app.input_state.steady[pointer_idx] = steady;
            return (Some((pointer_hit, hit)), result.haptics);
        }
    }

    (None, None)
}

struct StartGrabParams<'a> {
    idx: usize,
    id: OverlayID,
    name: Arc<str>,
    editing: bool,
    state: &'a mut OverlayWindowState,
    app: &'a mut AppState,
    edit_mode: bool,
    grab_float: bool,
}

fn start_grab(par: StartGrabParams) {
    let (app, id, state) = (par.app, par.id, par.state);

    let pointer = &mut app.input_state.pointers[par.idx];

    // Grab anchor if:
    // - grabbed overlay is Anchored
    // - not in editmode
    // - not using grab_float
    // - grabbing with one hand. (grabbing with the 2nd hand will grab the individual overlay instead)
    let grab_anchor = !par.edit_mode
        && !app.session.config.world_lock_grab
        && !par.grab_float
        && !app.anchor_grabbed
        && matches!(state.positioning, Positioning::Anchored);

    let relative_grab_transform = if grab_anchor {
        app.anchor
    } else {
        state.transform
    };

    let offset = pointer.pose.inverse() * relative_grab_transform;

    app.anchor_grabbed = grab_anchor;

    pointer.interaction.grabbed = Some(GrabData {
        offset,
        grabbed_id: id,
        grab_anchor,
        two: None,
        was_two: false,
    });

    app.tasks.enqueue(TaskType::Overlay(OverlayTask::Modify(
        OverlaySelector::Id(id),
        Box::new({
            let pos = state.positioning;
            let name = par.name.clone();
            move |app, o| {
                let _ = o
                    .backend
                    .notify(
                        app,
                        OverlayEventData::OverlayGrabbed {
                            name,
                            pos,
                            editing: par.editing,
                        },
                    )
                    .inspect_err(|e| log::warn!("Error during Notify OverlayGrabbed: {e:?}"));
            }
        }),
    )));

    // Show anchor
    app.tasks.enqueue(TaskType::Overlay(OverlayTask::Modify(
        OverlaySelector::Name(ANCHOR_NAME.clone()),
        Box::new(|app, o| {
            o.activate(app, true);
        }),
    )));

    if let Some(hand) = pointer.hand()
        && !app.session.config.hide_grab_help
        && !pointer.handsfree
    {
        let pos = state.positioning;
        app.tasks.enqueue(TaskType::Overlay(OverlayTask::Modify(
            OverlaySelector::Name(GRAB_HELP_NAME.clone()),
            Box::new(move |app, o| {
                let _ = o
                    .backend
                    .notify(
                        app,
                        OverlayEventData::OverlayGrabbed {
                            name: par.name,
                            pos,
                            editing: par.editing,
                        },
                    )
                    .inspect_err(|e| log::warn!("Error during Notify OverlayGrabbed: {e:?}"));

                o.default_state.positioning = Positioning::FollowHand { hand, lerp: 0.1 };
                o.default_state.align_to_hmd = true;
                o.activate(app, true);
            }),
        )));
    }
}

fn handle_scale(transform: &mut Affine3A, scroll_y: f32) {
    let cur_scale = transform.x_axis.length();
    if cur_scale < 0.1 && scroll_y > 0.0 {
        return;
    }
    if cur_scale > 20. && scroll_y < 0.0 {
        return;
    }

    transform.matrix3 = transform
        .matrix3
        .mul_scalar(0.025f32.mul_add(-scroll_y, 1.0));
}

#[allow(clippy::too_many_lines)]
/// Both controllers grip the same overlay: scale follows the hand distance ratio, yaw follows the hand-to-hand
/// direction, position follows the midpoint. Returns true when handled (the caller skips the single-hand grab).
fn grab_limits(cfg: &wlx_common::config::GeneralConfig) -> grabmath::Limits {
    grabmath::Limits {
        size_min: cfg.grab_size_min,
        size_max: cfg.grab_size_max,
        dist_min: cfg.grab_dist_min,
        dist_max: cfg.grab_dist_max,
    }
}

fn handle_two_hand<O>(idx: usize, overlay: &mut OverlayWindowData<O>, app: &mut AppState) -> bool
where
    O: Default,
{
    let ps = &mut app.input_state.pointers;
    let other = 1 - idx;
    let (Some(mine), Some(theirs)) = (ps[idx].interaction.grabbed, ps[other].interaction.grabbed) else {
        return false;
    };
    if !(ps[idx].now.grab && ps[other].now.grab) || mine.grabbed_id != theirs.grabbed_id || mine.grab_anchor {
        return false;
    }
    if let Some(g) = ps[idx].interaction.grabbed.as_mut() {
        g.was_two = true;
    }
    if idx != 0 {
        return true; // hand 0 does the work
    }
    let Some(state) = overlay.config.active_state.as_mut() else {
        return true;
    };
    let (pa, pb) = (ps[0].pose.translation, ps[1].pose.translation);
    let (v, m) = (pb - pa, (pa + pb) * 0.5);
    let g = ps[0].interaction.grabbed.as_mut().unwrap();
    let two = *g.two.get_or_insert(TwoHand {
        v0: v,
        m0: m,
        t0: state.transform,
    });
    state.transform = grabmath::two_hand(&two.t0, two.v0, two.m0, v, m, &grab_limits(&app.session.config));
    overlay.config.pause_movement = true;
    overlay.config.dirty = true;
    true
}

fn handle_grabbed<O>(idx: usize, overlay: &mut OverlayWindowData<O>, app: &mut AppState)
where
    O: Default,
{
    if app.session.config.world_lock_grab && handle_two_hand(idx, overlay, app) {
        return;
    }

    let pointer = &mut app.input_state.pointers[idx];
    let Some(grab_data) = pointer.interaction.grabbed.as_mut() else {
        log::error!("Grabbed overlay does not exist");
        return;
    };
    let grab_anchor = grab_data.grab_anchor;

    if pointer.now.grab {
        let Some(overlay_state) = overlay.config.active_state.as_mut() else {
            // overlay got toggled off while being grabbed. those dastardly users!
            // just wait for them to release the grab
            return;
        };

        let scroll_factor = 90.0 * app.delta_time;

        if grab_anchor {
            if pointer.now.click {
                pointer.interaction.mode = PointerMode::Special;
                let grab_dist = grab_data.offset.translation.length().clamp(0.5, 5.0) * 0.2 + 0.4;
                handle_scale(
                    &mut app.anchor,
                    pointer.now.scroll_y * grab_dist * scroll_factor,
                );
                handle_scale(
                    &mut grab_data.offset,
                    pointer.now.scroll_y * grab_dist * scroll_factor,
                );
            } else if app.session.config.allow_sliding && pointer.now.scroll_y.is_finite() {
                // single grab push/pull
                let grab_dist = grab_data.offset.translation.length().clamp(0.5, 5.0);
                grab_data.offset.translation.z -=
                    pointer.now.scroll_y * scroll_factor * 0.02 * grab_dist;
                grab_data.offset.translation.z = grab_data.offset.translation.z.min(-0.05);
            }
            if pointer.now.click_modifier_right {
                app.anchor = pointer.pose * grab_data.offset;
            } else {
                app.anchor.translation =
                    pointer.pose.transform_point3a(grab_data.offset.translation);
                let scale = scalar_scale(&app.anchor);
                realign(
                    &mut app.anchor,
                    &app.input_state.hmd,
                    scale,
                    app.session.config.snap_angle_deg,
                );
            }
        } else {
            // single grab resize
            if app.session.config.world_lock_grab {
                // desk mode: stick = push/pull; the size follows (keep apparent size). Resize is the two-hand gesture.
                if grab_data.was_two {
                    // second hand let go: continue as a single grab from the current pose
                    grab_data.was_two = false;
                    grab_data.two = None;
                    grab_data.offset = pointer.pose.inverse() * overlay_state.transform;
                }
                let cfg = &app.session.config;
                let lim = grab_limits(cfg);
                let (dt, y) = (app.delta_time, pointer.now.scroll_y);
                if y.is_finite() && y.abs() > 0.05 {
                    pointer.interaction.mode = PointerMode::Special;
                    let cur_scale = grabmath::uniform_scale(&overlay_state.transform);
                    let f = grabmath::push_pull(
                        &mut grab_data.offset,
                        cur_scale,
                        y,
                        dt,
                        cfg.grab_dist_rate,
                        cfg.grab_keep_apparent_size,
                        &lim,
                    );
                    overlay_state.transform.matrix3 = overlay_state.transform.matrix3.mul_scalar(f);
                }
            } else if pointer.now.click {
                pointer.interaction.mode = PointerMode::Special;
                let grab_dist = grab_data.offset.translation.length().clamp(0.5, 5.0) * 0.2 + 0.4;
                handle_scale(
                    &mut overlay_state.transform,
                    pointer.now.scroll_y * grab_dist * scroll_factor,
                );
                handle_scale(
                    &mut grab_data.offset,
                    pointer.now.scroll_y * grab_dist * scroll_factor,
                );
            } else if app.session.config.allow_sliding && pointer.now.scroll_y.is_finite() {
                // single grab push/pull
                let grab_dist = grab_data.offset.translation.length().clamp(0.5, 5.0);
                grab_data.offset.translation.z -=
                    pointer.now.scroll_y * scroll_factor * 0.02 * grab_dist;
                grab_data.offset.translation.z = grab_data.offset.translation.z.min(-0.05);
            }
            if pointer.now.click_modifier_right || app.session.config.world_lock_grab {
                overlay_state.transform = pointer.pose * grab_data.offset;
            } else {
                let scale = scalar_scale(&overlay_state.transform);
                overlay_state.transform.translation =
                    pointer.pose.transform_point3a(grab_data.offset.translation);
                realign(
                    &mut overlay_state.transform,
                    &app.input_state.hmd,
                    scale,
                    app.session.config.snap_angle_deg,
                );
            }
            overlay.config.pause_movement = true;
            overlay.config.dirty = true;
        }
    } else {
        // not now.grab
        pointer.interaction.grabbed = None;
        if grab_anchor {
            app.anchor_grabbed = false;
        } else {
            // single grab released
            if &*overlay.config.name == WATCH_NAME {
                // watch special: when dropped, follow the hand that wasn't grabbing
                if let Some(overlay_state) = overlay.config.active_state.as_mut() {
                    let align_to_hmd = overlay_state.align_to_hmd;
                    overlay_state.positioning = match overlay_state.positioning {
                        Positioning::FollowHand { hand, lerp } => match pointer.hand() {
                            Some(LeftRight::Left) => Positioning::FollowHand {
                                hand: LeftRight::Right,
                                lerp,
                            },
                            Some(LeftRight::Right) => Positioning::FollowHand {
                                hand: LeftRight::Left,
                                lerp,
                            },
                            _ => Positioning::FollowHand { hand, lerp },
                        },
                        x => x,
                    };
                    overlay_state.align_to_hmd = align_to_hmd;
                }
            } else if overlay.config.global
                && let Some(active_state) = overlay.config.active_state.as_ref()
            {
                let cur_scale = overlay.config.default_state.transform.x_axis.length();
                let tgt_scale = active_state.transform.x_axis.length();
                let mat = &mut overlay.config.default_state.transform.matrix3;
                *mat = mat.mul_scalar(tgt_scale / cur_scale);
            }
            overlay.config.pause_movement = false;
            let world_lock = app.session.config.world_lock_grab
                && !overlay.config.editing
                && &*overlay.config.name != WATCH_NAME
                && matches!(
                    overlay.config.category,
                    OverlayCategory::Screen | OverlayCategory::Keyboard | OverlayCategory::WayVR
                );
            if let Some(overlay_state) = overlay.config.active_state.as_mut() {
                if world_lock {
                    // released in the room: stays there (stage space) and is saved as such
                    overlay_state.positioning = Positioning::Static;
                }
                window::save_transform(overlay_state, app);
            }
        }

        // Hide anchor
        app.tasks.enqueue(TaskType::Overlay(OverlayTask::Modify(
            OverlaySelector::Name(ANCHOR_NAME.clone()),
            Box::new(|_app, o| {
                o.deactivate();
            }),
        )));
        app.tasks.enqueue(TaskType::Overlay(OverlayTask::Modify(
            OverlaySelector::Name(GRAB_HELP_NAME.clone()),
            Box::new(|_app, o| {
                o.deactivate();
            }),
        )));
        log::debug!("Hand {}: dropped {}", idx, overlay.config.name);
    }
}

fn ray_test(
    ray_origin: &Affine3A,
    overlay: OverlayID,
    overlay_pose: &Affine3A,
    curvature: Option<&f32>,
) -> Option<RayHit> {
    let (dist, local_pos) = curvature.map_or_else(
        || {
            Some(raycast_plane(
                ray_origin,
                Vec3A::NEG_Z,
                overlay_pose,
                Vec3A::NEG_Z,
            ))
        },
        |curvature| raycast_cylinder(ray_origin, Vec3A::NEG_Z, overlay_pose, *curvature),
    )?;

    if dist < 0.0 {
        // hit is behind us
        return None;
    }

    Some(RayHit {
        overlay,
        global_pos: ray_origin.transform_point3a(Vec3A::NEG_Z * dist),
        local_pos,
        dist,
    })
}

fn raycast_plane(
    source: &Affine3A,
    source_fwd: Vec3A,
    plane: &Affine3A,
    plane_norm: Vec3A,
) -> (f32, Vec2) {
    let plane_normal = plane.transform_vector3a(plane_norm);
    let ray_dir = source.transform_vector3a(source_fwd);

    let d = plane.translation.dot(-plane_normal);
    let mut dist = -(d + source.translation.dot(plane_normal)) / ray_dir.dot(plane_normal);

    let hit_local = plane
        .inverse()
        .transform_point3a(source.translation + ray_dir * dist)
        .xy();

    // hitting the backside of the plane, make the hit invalid
    if ray_dir.dot(plane_normal) < 0.0 && dist.is_sign_positive() {
        dist = -dist;
    }

    (dist, hit_local)
}

fn raycast_cylinder(
    source: &Affine3A,
    source_fwd: Vec3A,
    plane: &Affine3A,
    curvature: f32,
) -> Option<(f32, Vec2)> {
    // this is solved locally; (0,0) is the center of the cylinder, and the cylinder is aligned with the Y axis
    let size = plane.x_axis.length();
    let to_local = Affine3A {
        matrix3: plane.matrix3.mul_scalar(1.0 / size),
        translation: plane.translation,
    }
    .inverse();

    let radius = size / (2.0 * PI * curvature);

    let ray_dir = to_local.transform_vector3a(source.transform_vector3a(source_fwd));
    let ray_origin = to_local.transform_point3a(source.translation) + Vec3A::NEG_Z * radius;

    let v_dir = ray_dir.xz();
    let v_pos = ray_origin.xz();

    let l_dir = v_dir.dot(v_dir);
    let l_pos = v_dir.dot(v_pos);
    let c = radius.mul_add(-radius, v_pos.dot(v_pos));

    let d = l_pos.mul_add(l_pos, -(l_dir * c));
    if d < f32::EPSILON {
        return None;
    }

    let sqrt_d = d.sqrt();

    let t1 = (-l_pos - sqrt_d) / l_dir;
    let t2 = (-l_pos + sqrt_d) / l_dir;

    let mut t = t1.max(t2);

    if t < f32::EPSILON {
        return None;
    }

    let mut hit_local = ray_origin + ray_dir * t;
    if hit_local.z > 0.0 {
        // hitting the opposite half of the cylinder
        return None;
    }

    let normal = Vec3A::new(hit_local.x, 0.0, hit_local.z).normalize();
    // If hitting from the outside, flip t
    if ray_dir.dot(normal) < 0.0 && t.is_sign_positive() {
        t = -t;
    }

    let max_angle = 2.0 * (size / (2.0 * radius));
    let x_angle = (hit_local.x / radius).asin();

    hit_local.x = x_angle / max_angle;
    hit_local.y /= size;

    Some((t, hit_local.xy()))
}
