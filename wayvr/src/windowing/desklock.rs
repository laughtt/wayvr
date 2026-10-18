//! Locked desk layouts (omarchyvr, user decisions 2026-10-18): with `desk_locked` (default on) the desk screens
//! (desktop captures, `OverlayCategory::Screen`) never move by themselves and cannot be moved by hand:
//!
//! * no grab-move, no two-hand resize, no edit-mode resize of a desk screen;
//! * no auto-reset of any kind (not the start-up "out of view for more than 2 s" one, not an invalid pose);
//! * `screen-nudge` / `screen-place` / `ResetOverlay` on a desk screen are refused (a log line says so);
//! * `desk-screens` places every shown screen at its slot of the default layout (viewlayout::default_layout)
//!   built around the **layout anchor**, not around the head: a yaw and a position, kept in
//!   `<config root>/desk-anchor.conf` so the next session puts the layout where it was. The geometry (width,
//!   distance, curvature) of the last placing call is kept with it, so a later show/hide-only call (the screens
//!   watcher) uses the same size;
//! * `desk-recenter` (and Reset screens / double-click B while locked) sets the anchor from the head now (yaw only,
//!   level) and places the whole layout again: same shape, same height relation, in front of the user.
//!
//! First ever start (no anchor file), or a stored anchor far from the head at the first placing call of a session
//! (another room, the tracking origin moved): the anchor is taken from the head then, once, and saved.
use glam::{Affine3A, Quat, Vec3A};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::windowing::window::OverlayCategory;

/// The anchor file in WayVR's config root (the omarchyvr wayvr home).
pub const FILE_NAME: &str = "desk-anchor.conf";
/// A stored anchor is used at the first placement of a session only when the head is at most this far from it
/// (horizontally, metres): further away it belongs to another seat or another tracking origin.
pub const START_MAX_OFFSET: f32 = 1.5;
/// The same for height (metres).
pub const START_MAX_HEIGHT: f32 = 1.0;

/// Is this overlay locked in place? Desk screens (desktop captures) with `desk_locked`.
pub fn locked(desk_locked: bool, category: OverlayCategory) -> bool {
    desk_locked && category == OverlayCategory::Screen
}

/// The layout's size: main's width (metres), distance from the anchor (metres), curvature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub width: f32,
    pub distance: f32,
    pub curvature: f32,
}

impl Default for Geometry {
    /// omarchyvr's Work 4: 1.6 m wide at 2 m, flat.
    fn default() -> Self {
        Geometry { width: 1.6, distance: 2.0, curvature: 0.0 }
    }
}

/// Where the layout stands: the head's yaw (radians, about +Y; 0 = looking down -Z) and position when it was set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anchor {
    pub yaw: f32,
    pub pos: Vec3A,
}

impl Anchor {
    /// From a head pose: its forward direction flattened to the floor plane (pitch and roll ignored), its position.
    pub fn from_head(hmd: &Affine3A) -> Anchor {
        let f = -hmd.matrix3.z_axis;
        let f = Vec3A::new(f.x, 0.0, f.z);
        let yaw = if f.length() < 1e-3 { 0.0 } else { (-f.x).atan2(-f.z) };
        Anchor { yaw, pos: hmd.translation }
    }

    /// The "head" the default layout is built around: level, turned by `yaw`, at `pos`.
    pub fn transform(&self) -> Affine3A {
        Affine3A::from_rotation_translation(Quat::from_rotation_y(self.yaw), self.pos.into())
    }

    /// Can this stored anchor be used for a head at `hmd` (same seat)?
    pub fn usable_from(&self, hmd: &Affine3A) -> bool {
        let d = hmd.translation - self.pos;
        self.yaw.is_finite()
            && self.pos.is_finite()
            && Vec3A::new(d.x, 0.0, d.z).length() <= START_MAX_OFFSET
            && d.y.abs() <= START_MAX_HEIGHT
    }
}

/// The anchor file: `yaw = R`, `pos = X Y Z`, `width = W`, `distance = D`, `curvature = C` (`#` comments).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Stored {
    pub anchor: Option<Anchor>,
    pub geometry: Geometry,
}

impl Stored {
    pub fn render(&self) -> String {
        let mut s = String::from("# WayVR desk layout anchor (desk_locked; set by desk-recenter, read at every start)\n");
        if let Some(a) = self.anchor {
            s += &format!("yaw = {}\npos = {} {} {}\n", a.yaw, a.pos.x, a.pos.y, a.pos.z);
        }
        let g = self.geometry;
        s += &format!("width = {}\ndistance = {}\ncurvature = {}\n", g.width, g.distance, g.curvature);
        s
    }

    pub fn parse(text: &str) -> Stored {
        let mut st = Stored::default();
        let (mut yaw, mut pos) = (None, None);
        for l in text.lines() {
            let l = l.trim();
            if l.starts_with('#') {
                continue;
            }
            let Some((k, v)) = l.split_once('=') else { continue };
            let num = |v: &str| v.trim().parse::<f32>().ok().filter(|x| x.is_finite());
            match k.trim() {
                "yaw" => yaw = num(v),
                "pos" => {
                    let p: Vec<f32> = v.split_whitespace().filter_map(num).collect();
                    if p.len() == 3 {
                        pos = Some(Vec3A::new(p[0], p[1], p[2]));
                    }
                }
                "width" => st.geometry.width = num(v).filter(|x| *x > 0.0).unwrap_or(st.geometry.width),
                "distance" => st.geometry.distance = num(v).filter(|x| *x > 0.0).unwrap_or(st.geometry.distance),
                "curvature" => st.geometry.curvature = num(v).map(|c| c.clamp(0.0, 1.0)).unwrap_or(st.geometry.curvature),
                _ => {}
            }
        }
        if let (Some(yaw), Some(pos)) = (yaw, pos) {
            st.anchor = Some(Anchor { yaw, pos });
        }
        st
    }
}

/// Is there a head pose yet? Before the runtime reports the first one the HMD pose is exactly identity
/// (screens_query treats it the same way: no head).
pub fn head_known(hmd: &Affine3A) -> bool {
    *hmd != Affine3A::IDENTITY && hmd.is_finite()
}

/// Where a placement's anchor came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    /// The session's anchor (set by its first real placement or a recenter).
    Session,
    /// The stored anchor, the head is near it: it becomes the session's anchor.
    Stored,
    /// Made from the head now: it becomes the session's anchor and is saved.
    Fresh,
    /// No head pose yet (identity): a stand-in for this placement only (the stored anchor, else straight ahead of
    /// the origin). Never saved, never the session's anchor; the layout is placed again at the first real pose.
    Provisional,
}

/// The anchor for a placement now: the session's anchor once it is set; at the first placement of a session the
/// stored one when the head is near it, else one from the head ([`Pick::Fresh`] = it was made now and must be
/// saved). Without a head pose yet nothing is decided ([`Pick::Provisional`]).
pub fn anchor_for(session: Option<Anchor>, stored: Option<Anchor>, hmd: &Affine3A) -> (Anchor, Pick) {
    if let Some(a) = session {
        return (a, Pick::Session);
    }
    if !head_known(hmd) {
        return (stored.unwrap_or(Anchor { yaw: 0.0, pos: Vec3A::ZERO }), Pick::Provisional);
    }
    match stored {
        Some(a) if a.usable_from(hmd) => (a, Pick::Stored),
        _ => (Anchor::from_head(hmd), Pick::Fresh),
    }
}

// ------------------------------------------------------------------------------------------- the session state

struct State {
    loaded: bool,
    stored: Stored,
    /// Set by the first placement of this session (or a recenter); never changes by itself after that.
    session: Option<Anchor>,
    /// A placement used a provisional anchor (no head pose yet): place the layout again at the first real pose.
    replace_pending: bool,
    /// The last `desk-screens` listing (for a recenter: the same screens, shown and hidden, placed again).
    last: Option<wayvr_ipc::packet_client::WlxDeskScreensParams>,
}

static STATE: Mutex<State> = Mutex::new(State { loaded: false, stored: Stored { anchor: None, geometry: Geometry { width: 1.6, distance: 2.0, curvature: 0.0 } }, session: None, replace_pending: false, last: None });

fn path() -> PathBuf {
    wlx_common::config_io::get_config_root().join(FILE_NAME)
}

fn load(st: &mut State) {
    if !st.loaded {
        st.loaded = true;
        if let Ok(t) = std::fs::read_to_string(path()) {
            st.stored = Stored::parse(&t);
            log::info!("desk lock: anchor file {} ({:?})", path().display(), st.stored);
        }
    }
}

fn save(stored: &Stored) {
    let p = path();
    if let Err(e) = write_atomic(&p, &stored.render()) {
        log::warn!("desk lock: cannot save {}: {e}", p.display());
    }
}

fn write_atomic(p: &Path, text: &str) -> std::io::Result<()> {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = p.with_extension("tmp");
    std::fs::write(&tmp, text)?;
    std::fs::rename(tmp, p)
}

/// The anchor to place the layout around now (see [`anchor_for`]); a new one is saved. Without a head pose yet
/// the anchor is provisional: not kept, not saved, and [`replace_due`] asks for a new placement later.
pub fn session_anchor(hmd: &Affine3A) -> Anchor {
    let mut st = STATE.lock().unwrap();
    load(&mut st);
    let (a, save_now) = st.place(hmd);
    if save_now {
        save(&st.stored);
    }
    a
}

impl State {
    /// The session logic of [`session_anchor`] without the file: the anchor, and whether `stored` must be saved.
    fn place(&mut self, hmd: &Affine3A) -> (Anchor, bool) {
        let (a, pick) = anchor_for(self.session, self.stored.anchor, hmd);
        match pick {
            Pick::Session => {}
            Pick::Provisional => {
                if !self.replace_pending {
                    log::info!("desk lock: no head pose yet; screens placed provisionally, the layout anchor is set at the first real pose");
                }
                self.replace_pending = true;
                return (a, false);
            }
            Pick::Fresh => log::info!("desk lock: new layout anchor in front of the head (yaw {:.1} deg){}", a.yaw.to_degrees(), if self.stored.anchor.is_some() { " (the stored one is far from the head)" } else { "" }),
            Pick::Stored => log::info!("desk lock: layout anchor from the last session (yaw {:.1} deg)", a.yaw.to_degrees()),
        }
        self.session = Some(a);
        if pick == Pick::Fresh {
            self.stored.anchor = Some(a);
            return (a, true);
        }
        (a, false)
    }

    /// See [`replace_due`].
    fn replace_due(&mut self, hmd: &Affine3A) -> bool {
        if self.replace_pending && head_known(hmd) {
            self.replace_pending = false;
            return true;
        }
        false
    }
}

/// Once per provisional placement: true at the first tick with a real head pose (the caller places the layout
/// again, which sets and saves the session's anchor from that pose).
pub fn replace_due(hmd: &Affine3A) -> bool {
    let due = STATE.lock().unwrap().replace_due(hmd);
    if due {
        log::info!("desk lock: first head pose, placing the layout again");
    }
    due
}

/// `desk-recenter`: the anchor from the head now, saved.
pub fn recenter(hmd: &Affine3A) -> Anchor {
    let mut st = STATE.lock().unwrap();
    load(&mut st);
    if !head_known(hmd) {
        // no head pose yet: nothing to recenter on; the first real pose places the layout
        st.replace_pending = true;
        let (a, _) = anchor_for(st.session, st.stored.anchor, hmd);
        log::info!("desk lock: recenter before the first head pose, deferred");
        return a;
    }
    let a = Anchor::from_head(hmd);
    st.session = Some(a);
    st.stored.anchor = Some(a);
    save(&st.stored);
    log::info!("desk lock: recentered, anchor yaw {:.1} deg", a.yaw.to_degrees());
    a
}

/// The layout geometry of the last placing call (stored across sessions).
pub fn geometry() -> Geometry {
    let mut st = STATE.lock().unwrap();
    load(&mut st);
    st.stored.geometry
}

/// A placing call sets the geometry (saved when it changed).
pub fn set_geometry(g: Geometry) {
    let mut st = STATE.lock().unwrap();
    load(&mut st);
    if st.stored.geometry != g {
        st.stored.geometry = g;
        save(&st.stored);
    }
}

/// Remember the last `desk-screens` listing (place flag cleared: a recenter only re-places from the new anchor).
pub fn remember(p: &wayvr_ipc::packet_client::WlxDeskScreensParams) {
    let mut p = p.clone();
    p.place = false;
    STATE.lock().unwrap().last = Some(p);
}

pub fn last_listing() -> Option<wayvr_ipc::packet_client::WlxDeskScreensParams> {
    STATE.lock().unwrap().last.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::windowing::viewlayout::{self, Screen, Slot};
    use glam::Vec3;

    fn head(yaw_deg: f32, pitch_deg: f32, roll_deg: f32, pos: Vec3) -> Affine3A {
        let r = Quat::from_rotation_y(yaw_deg.to_radians()) * Quat::from_rotation_x(pitch_deg.to_radians()) * Quat::from_rotation_z(roll_deg.to_radians());
        Affine3A::from_rotation_translation(r, pos)
    }

    fn six(width: f32) -> Vec<(Slot, Screen)> {
        Slot::ALL.iter().map(|s| (*s, Screen { t: Affine3A::from_scale(Vec3::ONE * width), size: (width, width * 9.0 / 16.0) })).collect()
    }

    #[test]
    fn only_desk_screens_are_locked_and_only_with_the_switch() {
        assert!(locked(true, OverlayCategory::Screen));
        assert!(!locked(false, OverlayCategory::Screen));
        for c in [OverlayCategory::WayVR, OverlayCategory::Mirror, OverlayCategory::Panel, OverlayCategory::Keyboard, OverlayCategory::Dashboard, OverlayCategory::Internal] {
            assert!(!locked(true, c), "{c:?}");
        }
    }

    #[test]
    fn the_anchor_takes_the_yaw_only() {
        let a = Anchor::from_head(&head(30.0, 25.0, 10.0, Vec3::new(0.2, 1.6, -0.3)));
        assert!((a.yaw.to_degrees() - 30.0).abs() < 0.01, "{}", a.yaw.to_degrees());
        assert_eq!(a.pos, Vec3A::new(0.2, 1.6, -0.3));
        let t = a.transform();
        // level: its up axis is the world's
        assert!((t.matrix3.y_axis - Vec3A::Y).length() < 1e-5);
        // straight up / down still gives a usable yaw
        assert!(Anchor::from_head(&head(0.0, 90.0, 0.0, Vec3::ZERO)).yaw.is_finite());
    }

    #[test]
    fn the_file_round_trips_and_bad_lines_are_ignored() {
        let s = Stored { anchor: Some(Anchor { yaw: 1.25, pos: Vec3A::new(0.5, 1.7, -2.0) }), geometry: Geometry { width: 3.2, distance: 3.5, curvature: 0.35 } };
        assert_eq!(Stored::parse(&s.render()), s);
        let p = Stored::parse("# x\nyaw = nan\npos = 1 2\nwidth = -3\ndistance = 2.5\ncurvature = 7\nbogus\n");
        assert_eq!(p.anchor, None);
        assert_eq!(p.geometry, Geometry { width: 1.6, distance: 2.5, curvature: 1.0 });
        assert_eq!(Stored::parse(""), Stored::default());
    }

    #[test]
    fn a_stored_anchor_is_used_only_from_the_same_seat() {
        let stored = Anchor { yaw: 0.7, pos: Vec3A::new(0.0, 1.6, 0.0) };
        // the user sits down again, looking elsewhere: the layout comes back where it was (not where they look)
        let (a, pick) = anchor_for(None, Some(stored), &head(-80.0, 0.0, 0.0, Vec3::new(0.3, 1.5, 0.2)));
        assert_eq!((a, pick), (stored, Pick::Stored));
        // another room / the tracking origin moved: from the head, saved
        let h = head(-80.0, 0.0, 0.0, Vec3::new(4.0, 1.6, 0.0));
        let (a, pick) = anchor_for(None, Some(stored), &h);
        assert!(pick == Pick::Fresh && (a.yaw.to_degrees() + 80.0).abs() < 0.01);
        assert_eq!(anchor_for(None, Some(stored), &head(0.0, 0.0, 0.0, Vec3::new(0.0, 3.0, 0.0))).1, Pick::Fresh, "1.4 m higher");
        // first ever start: from the head
        assert_eq!(anchor_for(None, None, &h).1, Pick::Fresh);
        // once the session has one it never changes by itself, wherever the head goes
        let (a, pick) = anchor_for(Some(stored), None, &head(170.0, 40.0, 0.0, Vec3::new(9.0, 0.0, 9.0)));
        assert_eq!((a, pick), (stored, Pick::Session));
    }

    #[test]
    fn no_anchor_is_set_or_saved_before_the_first_head_pose() {
        let id = Affine3A::IDENTITY;
        assert!(!head_known(&id));
        let mut st = State { loaded: true, stored: Stored::default(), session: None, replace_pending: false, last: None };
        // first ever start, WayVR places the screens before Monado reports a head: provisional, nothing kept
        let (a, save) = st.place(&id);
        assert_eq!((a.yaw, a.pos, save), (0.0, Vec3A::ZERO, false));
        assert_eq!((st.session, st.stored.anchor), (None, None));
        assert!(st.replace_pending);
        // a second placement still without a head: still nothing kept
        assert!(!st.place(&id).1 && st.session.is_none());
        assert!(!st.replace_due(&id), "no re-place while the head is unknown");
        // the first real pose: re-place once, and that placement sets and saves the anchor from the head
        let h = head(30.0, 0.0, 0.0, Vec3::new(0.2, 1.6, 0.1));
        assert!(st.replace_due(&h));
        assert!(!st.replace_due(&h), "only once");
        let (a, save) = st.place(&h);
        assert!(save && (a.yaw.to_degrees() - 30.0).abs() < 0.01 && st.session == Some(a) && st.stored.anchor == Some(a));
        // with a stored anchor and no head yet: the stored one is used provisionally, not adopted
        let stored = Anchor { yaw: 0.7, pos: Vec3A::new(0.0, 1.6, 0.0) };
        let mut st = State { loaded: true, stored: Stored { anchor: Some(stored), geometry: Geometry::default() }, session: None, replace_pending: false, last: None };
        assert_eq!(st.place(&id), (stored, false));
        assert!(st.session.is_none() && st.replace_pending);
        assert!(st.replace_due(&h));
        assert_eq!(st.place(&h), (stored, false), "the head is near it: the stored anchor becomes the session's");
        assert_eq!(st.session, Some(stored));
    }

    #[test]
    fn the_layout_follows_the_anchor_not_the_head() {
        let anchor = Anchor { yaw: 0.0, pos: Vec3A::new(0.0, 1.6, 0.0) };
        let a = viewlayout::default_layout(&anchor.transform(), &six(1.6), 2.0, viewlayout::MARGIN);
        // the same anchor gives the same layout: nothing depends on where the head looks
        let b = viewlayout::default_layout(&anchor.transform(), &six(1.6), 2.0, viewlayout::MARGIN);
        assert_eq!(a, b);
        let main = a.iter().find(|(s, _)| *s == Slot::Main).unwrap().1;
        assert!((main.t.translation - Vec3A::new(0.0, 1.6, -2.0)).length() < 1e-4, "{:?}", main.t.translation);
        assert!(viewlayout::overlaps(&a.iter().map(|x| x.1).collect::<Vec<_>>(), anchor.pos, viewlayout::MARGIN * 0.99).is_empty());
    }

    #[test]
    fn recenter_turns_the_whole_layout_and_keeps_its_shape() {
        let old = Anchor { yaw: 0.0, pos: Vec3A::new(0.0, 1.6, 0.0) };
        let h = head(-65.0, -20.0, 5.0, Vec3::new(0.4, 1.2, 0.3));
        let new = Anchor::from_head(&h);
        let la = viewlayout::default_layout(&old.transform(), &six(1.6), 2.0, viewlayout::MARGIN);
        let lb = viewlayout::default_layout(&new.transform(), &six(1.6), 2.0, viewlayout::MARGIN);
        // every screen is the old one moved by the same rigid yaw + translation
        let k = new.transform() * old.transform().inverse();
        for ((sa, a), (sb, b)) in la.iter().zip(&lb) {
            assert_eq!(sa, sb);
            let moved = k * a.t;
            assert!((moved.translation - b.t.translation).length() < 1e-3, "{sa:?}");
            assert!((moved.matrix3.x_axis - b.t.matrix3.x_axis).length() < 1e-3 && (moved.matrix3.y_axis - b.t.matrix3.y_axis).length() < 1e-3, "{sa:?}");
        }
        // in front of the head: main straight ahead at its distance, at head height (pitch of the head ignored)
        let main = lb.iter().find(|(s, _)| *s == Slot::Main).unwrap().1;
        let d = main.t.translation - h.translation;
        assert!((d.y).abs() < 1e-4 && (Vec3A::new(d.x, 0.0, d.z).length() - 2.0).abs() < 1e-3);
        let (yaw, _) = viewlayout::yaw_pitch(d);
        assert!((yaw.to_degrees() + 65.0).abs() < 0.1, "{}", yaw.to_degrees());
    }
}
