//! Where the desk screens go so that none hides another **from the user's current head position** (omarchyvr, user
//! decisions 2026-10-17). Pure maths, no WayVR state; used once per command (never every frame):
//!
//! * [`overlap`]: do two flat screens overlap as seen from an eye point? Both screens are projected through the eye
//!   onto one plane (central / gnomonic projection keeps straight edges straight, so each screen is a convex
//!   quadrilateral) and tested with the separating axis theorem, with an angular margin.
//! * [`default_layout`]: up to six screens in two rows of three (`up-left up up-right` over `left main right`), main
//!   straight ahead, every neighbour turned about the head by the smallest angle that leaves the margin free. The
//!   angles come from each screen's own size and distance (no table of angles).
//! * [`place`]: "put THIS screen to the left of / right of / above / below THAT one", "bring it in front of me",
//!   "swap": the commanded screen is placed in one step from the current head pose, then [`resolve`] shifts the others
//!   along the arc around the head (yaw, or pitch) by the smallest angle that frees them, the commanded screen fixed.
//!
//! Rotations are always about the eye point, so a screen keeps its distance and keeps facing the user.
//! A screen transform here is WayVR's: uniform scale x rotation + translation, local +X right, +Y up, +Z towards the
//! viewer, centred on its translation. Its displayed size (width, height) in metres is given next to it.

use glam::{Affine3A, Quat, Vec2, Vec3A};

/// A screen: its transform and displayed (width, height) in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Screen {
    pub t: Affine3A,
    pub size: (f32, f32),
}

/// Default angular margin between two screens, as seen from the eye (1.5 deg; absorbs curved screens and the
/// eye moving a few cm inside the head).
pub const MARGIN: f32 = 1.5 * std::f32::consts::PI / 180.0;
/// The highest/lowest a screen centre may be turned (elevation, radians).
const MAX_ELEVATION: f32 = 75.0 * std::f32::consts::PI / 180.0;

impl Screen {
    pub fn centre(&self) -> Vec3A {
        self.t.translation
    }
    /// Unit right and up axes.
    pub fn axes(&self) -> (Vec3A, Vec3A) {
        (
            self.t.matrix3.x_axis.normalize_or_zero(),
            self.t.matrix3.y_axis.normalize_or_zero(),
        )
    }
    /// The four corners (top left, top right, bottom right, bottom left).
    pub fn corners(&self) -> [Vec3A; 4] {
        let (r, u) = self.axes();
        let (hw, hh) = (self.size.0 * 0.5, self.size.1 * 0.5);
        let c = self.centre();
        [
            c - r * hw + u * hh,
            c + r * hw + u * hh,
            c + r * hw - u * hh,
            c - r * hw - u * hh,
        ]
    }
    /// Rotate the whole screen about an axis through `eye`.
    pub fn rotated(&self, eye: Vec3A, axis: Vec3A, angle: f32) -> Screen {
        if axis.length_squared() < 1e-12 || angle == 0.0 {
            return *self;
        }
        let rot = Affine3A::from_quat(Quat::from_axis_angle(axis.normalize().into(), angle));
        let t = Affine3A::from_translation(eye.into())
            * rot
            * Affine3A::from_translation((-eye).into())
            * self.t;
        Screen { t, size: self.size }
    }
}

/// Unit direction from the eye to a point.
fn dir(eye: Vec3A, p: Vec3A) -> Vec3A {
    (p - eye).normalize_or(Vec3A::NEG_Z)
}

/// Azimuth (radians, positive = to the LEFT, i.e. a turn about +Y) and elevation (positive = up) of a direction.
pub fn yaw_pitch(d: Vec3A) -> (f32, f32) {
    let d = d.normalize_or(Vec3A::NEG_Z);
    ((-d.x).atan2(-d.z), d.y.clamp(-1.0, 1.0).asin())
}

/// The angle (radians) between two directions.
fn angle(a: Vec3A, b: Vec3A) -> f32 {
    a.normalize_or_zero()
        .dot(b.normalize_or_zero())
        .clamp(-1.0, 1.0)
        .acos()
}

/// Half the angle a screen covers around its centre direction (a cone that contains it).
fn cone(s: &Screen, eye: Vec3A) -> f32 {
    let c = dir(eye, s.centre());
    s.corners()
        .iter()
        .map(|&k| angle(c, k - eye))
        .fold(0.0, f32::max)
}

/// Do `a` and `b` overlap (one hides part of the other, or they come closer than `margin` radians) as seen from
/// `eye`? Exact for flat screens: both are projected through the eye onto the plane facing the mean of their
/// directions, and two convex quadrilaterals overlap unless an edge normal separates them.
pub fn overlap(a: &Screen, b: &Screen, eye: Vec3A, margin: f32) -> bool {
    let (da, db) = (dir(eye, a.centre()), dir(eye, b.centre()));
    // quick reject: the cones that contain them are apart
    if angle(da, db) > cone(a, eye) + cone(b, eye) + margin + 1e-4 {
        return false;
    }
    let m = (da + db).normalize_or(da);
    let u = Vec3A::Y.cross(m);
    let u = if u.length_squared() < 1e-8 {
        Vec3A::X
    } else {
        u.normalize()
    };
    let v = m.cross(u);
    let proj = |s: &Screen| -> Option<[Vec2; 4]> {
        let mut out = [Vec2::ZERO; 4];
        for (o, k) in out.iter_mut().zip(s.corners()) {
            let d = k - eye;
            let z = d.dot(m);
            if z < 1e-3 {
                return None;
            }
            *o = Vec2::new(d.dot(u) / z, d.dot(v) / z);
        }
        Some(out)
    };
    let (Some(pa), Some(pb)) = (proj(a), proj(b)) else {
        // a corner behind the projection plane (a screen wider than 180 deg around, or the eye inside it): the cones
        // already touch, so call it an overlap
        return true;
    };
    let g = margin.max(0.0).tan();
    let separated_on = |n: Vec2| {
        let n = n.normalize_or_zero();
        if n == Vec2::ZERO {
            return false;
        }
        let (amin, amax) = pa.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| {
            (lo.min(p.dot(n)), hi.max(p.dot(n)))
        });
        let (bmin, bmax) = pb.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| {
            (lo.min(p.dot(n)), hi.max(p.dot(n)))
        });
        amax + g <= bmin || bmax + g <= amin
    };
    for poly in [&pa, &pb] {
        for i in 0..4 {
            let e = poly[(i + 1) % 4] - poly[i];
            if separated_on(Vec2::new(-e.y, e.x)) {
                return false;
            }
        }
    }
    true
}

/// Every pair `(i, j)` (i < j) that overlaps from `eye`.
pub fn overlaps(screens: &[Screen], eye: Vec3A, margin: f32) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for i in 0..screens.len() {
        for j in i + 1..screens.len() {
            if overlap(&screens[i], &screens[j], eye, margin) {
                v.push((i, j));
            }
        }
    }
    v
}

/// A direction to turn a screen about the head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Turn {
    Left,
    Right,
    Up,
    Down,
}

impl Turn {
    pub const ALL: [Turn; 4] = [Turn::Left, Turn::Right, Turn::Up, Turn::Down];
    pub const fn horizontal(self) -> bool {
        matches!(self, Turn::Left | Turn::Right)
    }
}

/// `s` turned about the eye by `angle` (radians, >= 0) in direction `t`: left/right about the world up axis,
/// up/down about the horizontal axis at right angles to the screen's direction (it keeps facing the user).
pub fn turned(s: &Screen, eye: Vec3A, t: Turn, angle: f32) -> Screen {
    match t {
        Turn::Left => s.rotated(eye, Vec3A::Y, angle),
        Turn::Right => s.rotated(eye, Vec3A::Y, -angle),
        Turn::Up | Turn::Down => {
            let d = dir(eye, s.centre());
            let mut right = d.cross(Vec3A::Y);
            if right.length_squared() < 1e-8 {
                right = s.axes().0;
            }
            s.rotated(eye, right, if t == Turn::Up { angle } else { -angle })
        }
    }
}

/// The smallest turn (radians) of `s` in direction `t` after which it overlaps none of `others` (searched up to
/// `max`, 0.5 deg steps then bisection to ~0.01 deg). `Some(0.0)` if it is free already; `None` if no free place
/// in that direction (or the turn would push it above/below the elevation limit).
/// A screen that overlaps (or nearly touches) another must end the full `margin` away from it; one that was already
/// clear of another only has to stay at least half the margin away (so turning a screen along its row does not
/// count as hitting the row above, which it only passes at the margin).
pub fn free_turn(
    s: &Screen,
    others: &[Screen],
    eye: Vec3A,
    t: Turn,
    margin: f32,
    max: f32,
) -> Option<f32> {
    let need: Vec<f32> = others
        .iter()
        .map(|o| {
            if overlap(s, o, eye, margin * 0.5) {
                margin
            } else {
                margin * 0.5
            }
        })
        .collect();
    let clear = |x: &Screen| {
        !others
            .iter()
            .zip(&need)
            .any(|(o, m)| overlap(x, o, eye, *m))
    };
    let ok_elev = |x: &Screen| yaw_pitch(dir(eye, x.centre())).1.abs() <= MAX_ELEVATION;
    if clear(s) {
        return Some(0.0);
    }
    let step = 0.5f32.to_radians();
    let mut lo = 0.0;
    let mut a = step;
    while a <= max + 1e-6 {
        let x = turned(s, eye, t, a);
        if !t.horizontal() && !ok_elev(&x) {
            return None;
        }
        if clear(&x) {
            let mut hi = a;
            for _ in 0..12 {
                let mid = 0.5 * (lo + hi);
                if clear(&turned(s, eye, t, mid)) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            return Some(hi);
        }
        lo = a;
        a += step;
    }
    None
}

/// How much more an up/down push "costs" than a sideways one of the same angle: the screens stay in their rows
/// unless going up or down is much shorter.
const VERTICAL_COST: f32 = 2.5;

fn cost(t: Turn, a: f32) -> f32 {
    if t.horizontal() { a } else { a * VERTICAL_COST }
}

/// Push the screens apart in one go: every screen that overlaps another (from `eye`) is turned along the arc by the
/// smallest angle that frees it from the screens already settled. `fixed` screens never move (the commanded one, its
/// reference); the others settle one by one, nearest to a fixed screen first, each pushed AWAY from the screen it
/// overlapped (horizontally or vertically, whichever needs the smaller turn; up/down only within the elevation limit).
/// Returns the screens that moved.
pub fn resolve(screens: &mut [Screen], fixed: &[bool], eye: Vec3A, margin: f32) -> Vec<usize> {
    let n = screens.len();
    let mut settled: Vec<bool> = (0..n)
        .map(|i| fixed.get(i).copied().unwrap_or(false))
        .collect();
    let mut moved = Vec::new();
    for _ in 0..(2 * n + 2) {
        let pairs = overlaps(screens, eye, margin);
        let pairs: Vec<(usize, usize)> = pairs
            .into_iter()
            .filter(|&(i, j)| !(settled[i] && settled[j]))
            .collect();
        if pairs.is_empty() {
            break;
        }
        // a pair with a settled screen first; else settle the screen nearest to any settled one (or the first)
        let (anchor, mov) = if let Some(&(i, j)) =
            pairs.iter().find(|&&(i, j)| settled[i] != settled[j])
        {
            if settled[i] { (i, j) } else { (j, i) }
        } else {
            let near = |k: usize| {
                (0..n)
                    .filter(|&f| settled[f])
                    .map(|f| angle(dir(eye, screens[f].centre()), dir(eye, screens[k].centre())))
                    .fold(f32::MAX, f32::min)
            };
            let (i, j) = pairs[0];
            let (a, b) = if near(i) <= near(j) { (i, j) } else { (j, i) };
            settled[a] = true;
            (a, b)
        };
        let others: Vec<Screen> = (0..n)
            .filter(|&k| settled[k] && k != mov)
            .map(|k| screens[k])
            .collect();
        let (ya, pa) = yaw_pitch(dir(eye, screens[anchor].centre()));
        let (ym, pm) = yaw_pitch(dir(eye, screens[mov].centre()));
        let mut dy = ym - ya;
        if dy > std::f32::consts::PI {
            dy -= std::f32::consts::TAU;
        } else if dy < -std::f32::consts::PI {
            dy += std::f32::consts::TAU;
        }
        let dp = pm - pa;
        // away from the anchor; on a tie both ways are tried
        let mut cands: Vec<Turn> = Vec::new();
        if dy > 1e-4 {
            cands.push(Turn::Left)
        } else if dy < -1e-4 {
            cands.push(Turn::Right)
        } else {
            cands.extend([Turn::Left, Turn::Right])
        }
        if dp > 1e-4 {
            cands.push(Turn::Up)
        } else if dp < -1e-4 {
            cands.push(Turn::Down)
        } else {
            cands.extend([Turn::Up, Turn::Down])
        }
        let found = |cands: &[Turn]| {
            cands
                .iter()
                .filter_map(|&t| {
                    free_turn(&screens[mov], &others, eye, t, margin, std::f32::consts::PI)
                        .map(|a| (t, a))
                })
                // rows stay rows: a turn up or down counts VERTICAL_COST times its angle
                .min_by(|a, b| cost(a.0, a.1).total_cmp(&cost(b.0, b.1)))
        };
        // nothing free away from the anchor: any direction
        let best = found(&cands).or_else(|| found(&Turn::ALL));
        if let Some((t, a)) = best
            && a > 0.0
        {
            screens[mov] = turned(&screens[mov], eye, t, a);
            if !moved.contains(&mov) {
                moved.push(mov);
            }
        }
        settled[mov] = true;
    }
    moved
}

/// The six desk slots, numbered like the badges (docs/SCREENS.md): 1 main, 2 left, 3 right, 4 up, 5 up-left,
/// 6 up-right.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    Main,
    Left,
    Right,
    Up,
    UpLeft,
    UpRight,
}

impl Slot {
    pub const ALL: [Slot; 6] = [
        Slot::Main,
        Slot::Left,
        Slot::Right,
        Slot::Up,
        Slot::UpLeft,
        Slot::UpRight,
    ];
    pub const fn number(self) -> u8 {
        self as u8 + 1
    }
    pub fn from_number(n: u8) -> Option<Slot> {
        Slot::ALL.get(usize::from(n).checked_sub(1)?).copied()
    }
}

/// A screen of width `width` (its transform scale; landscape quads are as wide as their scale) and `size`, `distance`
/// metres straight ahead of the head (yaw only, level), facing it.
pub fn front(head: &Affine3A, scale: f32, size: (f32, f32), distance: f32) -> Screen {
    let f = -head.matrix3.z_axis;
    let f = Vec3A::new(f.x, 0.0, f.z);
    let f = if f.length() < 1e-3 {
        Vec3A::NEG_Z
    } else {
        f.normalize()
    };
    let rot = Quat::from_rotation_y((-f.x).atan2(-f.z));
    let pos = head.translation + f * distance.max(0.1);
    Screen {
        t: Affine3A::from_scale_rotation_translation(glam::Vec3::ONE * scale, rot, pos.into()),
        size,
    }
}

/// Lowest and highest elevation (radians) of a screen's outline as seen from the eye (its edges sampled).
pub fn elevation_range(s: &Screen, eye: Vec3A) -> (f32, f32) {
    let c = s.corners();
    let mut r = (f32::MAX, f32::MIN);
    for e in 0..4 {
        for k in 0..=16 {
            let p = c[e].lerp(c[(e + 1) % 4], k as f32 / 16.0);
            let el = yaw_pitch(p - eye).1;
            r = (r.0.min(el), r.1.max(el));
        }
    }
    r
}

/// `s` moved (same scale and size) to the pose of `at` (same direction, distance and orientation).
fn at_pose_of(s: &Screen, at: &Screen) -> Screen {
    let k = s.t.matrix3.x_axis.length() / at.t.matrix3.x_axis.length().max(1e-6);
    Screen {
        t: Affine3A {
            matrix3: at.t.matrix3 * k,
            translation: at.t.translation,
        },
        size: s.size,
    }
}

/// `s` re-aimed so its centre sits in direction `d` from the eye at its current distance, facing the eye (level).
pub fn aimed(s: &Screen, eye: Vec3A, d: Vec3A) -> Screen {
    let dist = (s.centre() - eye).length().max(0.1);
    let (yaw, pitch) = yaw_pitch(d);
    let rot = Quat::from_rotation_y(yaw)
        * Quat::from_rotation_x(pitch.clamp(-MAX_ELEVATION, MAX_ELEVATION));
    let scale = s.t.matrix3.x_axis.length();
    let pos = eye + Vec3A::from(rot * glam::Vec3::NEG_Z) * dist;
    Screen {
        t: Affine3A::from_scale_rotation_translation(glam::Vec3::ONE * scale, rot, pos.into()),
        size: s.size,
    }
}

/// `mov` placed right beside `reference` on side `t` (as seen from the eye): it takes the reference's pose and is
/// turned about the head by the smallest angle that leaves the margin between them.
pub fn beside(mov: &Screen, reference: &Screen, eye: Vec3A, t: Turn, margin: f32) -> Screen {
    beside_opt(mov, reference, eye, t, margin).unwrap_or_else(|| at_pose_of(mov, reference))
}

/// [`beside`], or None when there is no room on that side (above/below: the elevation limit).
pub fn beside_opt(
    mov: &Screen,
    reference: &Screen,
    eye: Vec3A,
    t: Turn,
    margin: f32,
) -> Option<Screen> {
    let start = at_pose_of(mov, reference);
    let a = free_turn(
        &start,
        std::slice::from_ref(reference),
        eye,
        t,
        margin,
        std::f32::consts::PI,
    )?;
    Some(turned(&start, eye, t, a))
}

/// The default layout (Work start, Reset screens). `screens[i]` gives each slot's screen with its current scale and
/// displayed size (the pose is ignored); hidden screens may be listed too, so a screen shown later has its place.
/// Two rows, each a band of elevation (as seen from the head):
/// * lower row: main straight ahead (level) at `distance`; left and right turned about the head (world up axis) by
///   the smallest angle that leaves `margin` between them and main;
/// * upper row: each screen turned up by the smallest angle that puts its LOWEST point `margin` above the HIGHEST
///   point of the lower row (so no turn along a row can ever make the rows overlap), up straight above main,
///   up-left / up-right at the azimuth of left / right, pushed further out along the row only if the upper row's
///   tighter azimuth makes them touch `up`.
///
/// No two screens overlap from the head (`head.translation`); every screen faces it at `distance`.
pub fn default_layout(
    head: &Affine3A,
    screens: &[(Slot, Screen)],
    distance: f32,
    margin: f32,
) -> Vec<(Slot, Screen)> {
    let eye = head.translation;
    let get = |slot: Slot| screens.iter().find(|(s, _)| *s == slot).map(|(_, x)| *x);
    // main's size drives the neighbours when main is not listed
    let main_in = get(Slot::Main).or_else(|| screens.first().map(|s| s.1));
    let Some(main_in) = main_in else {
        return Vec::new();
    };
    let main = front(
        head,
        main_in.t.matrix3.x_axis.length(),
        main_in.size,
        distance,
    );
    let own = |slot: Slot| at_pose_of(&get(slot).unwrap_or(main_in), &main);
    let left = beside(&own(Slot::Left), &main, eye, Turn::Left, margin);
    let right = beside(&own(Slot::Right), &main, eye, Turn::Right, margin);
    // the lower row's highest point, then each upper screen's pitch from its own size
    let top = [main, left, right]
        .iter()
        .map(|s| elevation_range(s, eye).1)
        .fold(f32::MIN, f32::max);
    let raise = |s: Screen| -> Screen {
        let ok = |a: f32| elevation_range(&turned(&s, eye, Turn::Up, a), eye).0 >= top + margin;
        let (mut lo, mut hi) = (0.0f32, MAX_ELEVATION);
        if !ok(hi) {
            return turned(&s, eye, Turn::Up, hi);
        }
        for _ in 0..24 {
            let mid = 0.5 * (lo + hi);
            if ok(mid) { hi = mid } else { lo = mid }
        }
        turned(&s, eye, Turn::Up, hi)
    };
    let yaw_of =
        |s: &Screen| yaw_pitch(dir(eye, s.centre())).0 - yaw_pitch(dir(eye, main.centre())).0;
    let up = raise(own(Slot::Up));
    let mut up_left = turned(&raise(own(Slot::UpLeft)), eye, Turn::Left, yaw_of(&left));
    let mut up_right = turned(
        &raise(own(Slot::UpRight)),
        eye,
        Turn::Right,
        -yaw_of(&right),
    );
    for (s, t) in [(&mut up_left, Turn::Left), (&mut up_right, Turn::Right)] {
        if let Some(a) = free_turn(
            s,
            std::slice::from_ref(&up),
            eye,
            t,
            margin,
            std::f32::consts::PI,
        ) {
            *s = turned(s, eye, t, a);
        }
    }
    let all = [main, left, right, up, up_left, up_right];
    Slot::ALL
        .iter()
        .zip(all)
        .filter(|(slot, _)| get(**slot).is_some() || **slot == Slot::Main && screens.is_empty())
        .map(|(slot, s)| (*slot, s))
        .collect()
}

/// What a screen command asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    /// To the left of the reference (or, without one, of its nearest neighbour on the left: one place left).
    Left,
    Right,
    Up,
    Down,
    /// Straight in front of the head (at its own distance).
    Front,
    /// Change places with the reference.
    Swap,
}

/// The screen nearest to `idx` on side `t` (its centre on that side, the two overlapping in the other axis's angular
/// range, i.e. in the same row / column), as seen from `eye`.
pub fn neighbour(screens: &[Screen], idx: usize, eye: Vec3A, t: Turn) -> Option<usize> {
    let ang = |s: &Screen| -> (f32, f32, f32, f32) {
        // angular box: yaw min/max, pitch min/max (relative to idx's yaw so the wrap is behind)
        let (y0, _) = yaw_pitch(dir(eye, screens[idx].centre()));
        let mut b = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for k in s.corners().into_iter().chain([s.centre()]) {
            let (y, p) = yaw_pitch(dir(eye, k));
            let mut y = y - y0;
            if y > std::f32::consts::PI {
                y -= std::f32::consts::TAU;
            } else if y < -std::f32::consts::PI {
                y += std::f32::consts::TAU;
            }
            b = (b.0.min(y), b.1.max(y), b.2.min(p), b.3.max(p));
        }
        b
    };
    let me = ang(&screens[idx]);
    let (myc, mpc) = (0.0f32, yaw_pitch(dir(eye, screens[idx].centre())).1);
    (0..screens.len())
        .filter(|&k| k != idx)
        .filter_map(|k| {
            let o = ang(&screens[k]);
            let (yc, pc) = ((o.0 + o.1) * 0.5, (o.2 + o.3) * 0.5);
            let (same_row, same_col) = (o.2 < me.3 && me.2 < o.3, o.0 < me.1 && me.0 < o.1);
            let d = match t {
                Turn::Left if same_row && yc > myc => yc - myc,
                Turn::Right if same_row && yc < myc => myc - yc,
                Turn::Up if same_col && pc > mpc => pc - mpc,
                Turn::Down if same_col && pc < mpc => mpc - pc,
                _ => return None,
            };
            Some((k, d))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(k, _)| k)
}

/// One screen command, computed once from the current head position: screen `idx` goes where `what` says (beside
/// `reference`, or beside its nearest neighbour on that side when `reference` is None; with no neighbour on that side
/// it turns one own width/height that way), then the others shift minimally along the arc so that, from `eye`, no two
/// overlap. `head` is the head pose (for `Front`). Returns the indexes that moved (idx first when it moved).
pub fn place(
    screens: &mut [Screen],
    idx: usize,
    what: Place,
    reference: Option<usize>,
    head: &Affine3A,
    margin: f32,
) -> Vec<usize> {
    let eye = head.translation;
    let n = screens.len();
    if idx >= n {
        return Vec::new();
    }
    let before = screens[idx];
    let reference = reference.filter(|&r| r < n && r != idx);
    let mut fixed = vec![false; n];
    fixed[idx] = true;
    let turn = match what {
        Place::Left => Some(Turn::Left),
        Place::Right => Some(Turn::Right),
        Place::Up => Some(Turn::Up),
        Place::Down => Some(Turn::Down),
        Place::Front | Place::Swap => None,
    };
    match (what, turn) {
        (_, Some(t)) => {
            let r = reference.or_else(|| neighbour(screens, idx, eye, t));
            match r {
                Some(r) => match beside_opt(&screens[idx], &screens[r], eye, t, margin) {
                    Some(x) => {
                        screens[idx] = x;
                        fixed[r] = true;
                    }
                    None => {
                        // no room on that side of it (the reference is near the top or bottom limit): this screen
                        // goes as far that way as allowed, in the reference's column, and the reference makes room
                        let s = at_pose_of(&screens[idx], &screens[r]);
                        let (yaw, _) = yaw_pitch(dir(eye, s.centre()));
                        let d = (s.centre() - eye).length().max(0.1);
                        let half = (s.size.1 * 0.5 / d).atan();
                        let el = (MAX_ELEVATION - half).max(0.0)
                            * if t == Turn::Up { 1.0 } else { -1.0 };
                        let to = Quat::from_rotation_y(yaw) * Quat::from_rotation_x(el);
                        screens[idx] = aimed(&s, eye, Vec3A::from(to * glam::Vec3::NEG_Z));
                    }
                },
                None => {
                    // already the last on that side: one own width (or height) further that way
                    let s = &screens[idx];
                    let d = (s.centre() - eye).length().max(0.1);
                    let ext = if t.horizontal() { s.size.0 } else { s.size.1 };
                    let a = 2.0 * (ext * 0.5 / d).atan() + margin;
                    let x = turned(s, eye, t, a);
                    if t.horizontal() || yaw_pitch(dir(eye, x.centre())).1.abs() <= MAX_ELEVATION {
                        screens[idx] = x;
                    }
                }
            }
        }
        (Place::Front, _) => {
            let f = -head.matrix3.z_axis;
            screens[idx] = aimed(&screens[idx], eye, f);
        }
        (Place::Swap, _) => {
            if let Some(r) = reference {
                let (a, b) = (screens[idx], screens[r]);
                screens[idx] = aimed(&a, eye, dir(eye, b.centre()));
                screens[r] = aimed(&b, eye, dir(eye, a.centre()));
                // not fixed: with different sizes or distances the two may touch after the swap; the other one gives way
            }
        }
        _ => {}
    }
    let mut moved: Vec<usize> = Vec::new();
    if screens[idx] != before {
        moved.push(idx);
    }
    if what == Place::Swap
        && let Some(r) = reference
    {
        moved.push(r);
    }
    for m in resolve(screens, &fixed, eye, margin) {
        if !moved.contains(&m) {
            moved.push(m);
        }
    }
    moved
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    const D: f32 = 2.5;
    fn head() -> Affine3A {
        Affine3A::from_translation(Vec3::new(0.0, 1.6, 0.0))
    }
    /// A 16:9 screen of width w (WayVR: scale = width).
    fn scr(w: f32) -> Screen {
        Screen {
            t: Affine3A::from_scale(Vec3::ONE * w),
            size: (w, w * 9.0 / 16.0),
        }
    }
    fn six(w: f32) -> Vec<(Slot, Screen)> {
        Slot::ALL.iter().map(|s| (*s, scr(w))).collect()
    }
    fn deg(r: f32) -> f32 {
        r.to_degrees()
    }
    /// No overlap, and at least half the margin between every two screens (what a command guarantees).
    fn none_overlap(s: &[Screen], eye: Vec3A) -> bool {
        overlaps(s, eye, MARGIN * 0.49).is_empty()
    }

    #[test]
    fn overlap_from_a_viewpoint() {
        let h = head();
        let eye = h.translation;
        let a = front(&h, 2.0, (2.0, 1.125), D);
        // the same screen 1 m further away, a bit to the side: hides part of it from here
        let mut b = front(&h, 2.0, (2.0, 1.125), D + 1.0);
        b.t.translation += Vec3A::new(0.5, 0.0, 0.0);
        assert!(overlap(&a, &b, eye, MARGIN));
        // seen from well above, the near one no longer hides the far one
        assert!(!overlap(&a, &b, eye + Vec3A::new(0.0, 6.0, -3.0), 0.0));
        // edge to edge by construction: apart, and the gap is the margin (just under it overlaps)
        let l = beside(&a, &a, eye, Turn::Left, MARGIN);
        assert!(!overlap(&a, &l, eye, MARGIN * 0.99));
        assert!(overlap(&a, &l, eye, MARGIN * 1.2));
        // from another head position the same pair can overlap
        let side_eye = eye + Vec3A::new(2.0, 0.0, 0.5);
        assert!(
            overlap(&a, &l, side_eye, MARGIN),
            "from the right, the left screen hides behind the main one"
        );
        // behind the user: never overlaps a screen in front
        let back = turned(&a, eye, Turn::Left, std::f32::consts::PI);
        assert!(!overlap(&a, &back, eye, MARGIN));
    }

    #[test]
    fn default_six_never_overlap_and_angles_come_from_the_sizes() {
        let h = head();
        let eye = h.translation;
        // from a small screen close by to a cinema screen far away (each 16:9, `w` wide, `d` away)
        for (w, d) in [
            (1.0f32, 1.0f32),
            (1.0, 2.5),
            (2.0, 2.0),
            (2.0, 2.5),
            (2.0, 4.0),
            (3.0, 2.5),
            (3.0, 4.0),
            (6.0, 8.0),
        ] {
            {
                let l = default_layout(&h, &six(w), d, MARGIN);
                assert_eq!(l.len(), 6);
                let s: Vec<Screen> = l.iter().map(|x| x.1).collect();
                assert!(
                    overlaps(&s, eye, MARGIN * 0.99).is_empty(),
                    "w {w} d {d}: {:?}",
                    overlaps(&s, eye, MARGIN * 0.99)
                );
                // every screen faces the head and keeps the distance
                for x in &s {
                    assert!(((x.centre() - eye).length() - d).abs() < 1e-3);
                    let facing =
                        x.t.matrix3
                            .z_axis
                            .normalize()
                            .dot((eye - x.centre()).normalize());
                    assert!(facing > 0.999, "facing {facing}");
                }
                // left/right yaw = the angular half widths of both + the margin (edge to edge seen from the head)
                let (yl, _) = yaw_pitch(s[1].centre() - eye);
                let expect = 2.0 * (w * 0.5 / d).atan() + MARGIN;
                assert!(
                    (yl - expect).abs() < 0.3f32.to_radians(),
                    "w {w} d {d}: yaw {} vs {}",
                    deg(yl),
                    deg(expect)
                );
                let (yr, _) = yaw_pitch(s[2].centre() - eye);
                assert!((yr + yl).abs() < 1e-3, "symmetric");
                // up row above, corner screens outside up
                let (_, pu) = yaw_pitch(s[3].centre() - eye);
                assert!(pu > 0.0);
                let (yul, pul) = yaw_pitch(s[4].centre() - eye);
                let (yur, _) = yaw_pitch(s[5].centre() - eye);
                assert!(yul > 0.0 && yur < 0.0 && pul > 0.0);
            }
        }
    }

    #[test]
    fn default_layout_follows_the_head_and_lists_only_given_slots() {
        let h = Affine3A::from_rotation_translation(
            Quat::from_rotation_y(1.0),
            Vec3::new(1.0, 1.7, -2.0),
        );
        let l = default_layout(
            &h,
            &[(Slot::Main, scr(2.0)), (Slot::Right, scr(1.5))],
            D,
            MARGIN,
        );
        assert_eq!(
            l.iter().map(|x| x.0).collect::<Vec<_>>(),
            [Slot::Main, Slot::Right]
        );
        let (ym, _) = yaw_pitch(l[0].1.centre() - h.translation);
        assert!(
            (ym - 1.0).abs() < 1e-3,
            "main straight ahead of the turned head"
        );
        assert!(none_overlap(&[l[0].1, l[1].1], h.translation));
    }

    #[test]
    fn move_left_puts_it_left_of_its_neighbour_and_nothing_overlaps() {
        let h = head();
        let eye = h.translation;
        let mut s: Vec<Screen> = default_layout(&h, &six(2.0), D, MARGIN)
            .into_iter()
            .map(|x| x.1)
            .collect();
        // "move this left" on the right screen (3): it goes left of main, main and the left one make room
        let moved = place(&mut s, 2, Place::Left, None, &h, MARGIN);
        assert_eq!(moved[0], 2);
        let y = |i: usize| yaw_pitch(s[i].centre() - eye).0;
        assert!(y(2) > y(0), "now left of main");
        assert!(
            none_overlap(&s, eye),
            "{:?}",
            overlaps(&s, eye, MARGIN * 0.99)
        );
        // main stayed put (the reference is fixed); the left screen was pushed further left
        assert!(y(0).abs() < 1e-3);
        assert!(y(1) > y(2), "the old left screen shifted out of the way");
        assert!(moved.contains(&1));
    }

    #[test]
    fn move_up_down_next_to_and_front_and_swap() {
        let h = head();
        let eye = h.translation;
        let base: Vec<Screen> = default_layout(&h, &six(2.0), D, MARGIN)
            .into_iter()
            .map(|x| x.1)
            .collect();
        // "put screen 2 above screen 1": left goes over main; up must make room
        let mut s = base.clone();
        place(&mut s, 1, Place::Up, Some(0), &h, MARGIN);
        assert!(none_overlap(&s, eye));
        let (y1, p1) = yaw_pitch(s[1].centre() - eye);
        assert!(y1.abs() < 1e-3 && p1 > 0.0, "straight above main");
        // down: below main
        let mut s = base.clone();
        place(&mut s, 3, Place::Down, Some(0), &h, MARGIN);
        assert!(yaw_pitch(s[3].centre() - eye).1 < 0.0 && none_overlap(&s, eye));
        // bring here: the head turned 30 deg right, the up-left screen comes in front of the eyes
        let turned_head = Affine3A::from_rotation_translation(
            Quat::from_rotation_y(-0.5),
            Vec3::new(0.0, 1.6, 0.0),
        );
        let mut s = base.clone();
        place(&mut s, 4, Place::Front, None, &turned_head, MARGIN);
        let (y4, p4) = yaw_pitch(s[4].centre() - eye);
        assert!((y4 + 0.5).abs() < 1e-3 && p4.abs() < 1e-3);
        assert!(none_overlap(&s, eye));
        // swap left and right: they change places, nothing overlaps
        let mut s = base.clone();
        let moved = place(&mut s, 1, Place::Swap, Some(2), &h, MARGIN);
        assert!(moved.contains(&1) && moved.contains(&2));
        assert!(yaw_pitch(s[1].centre() - eye).0 < 0.0 && yaw_pitch(s[2].centre() - eye).0 > 0.0);
        assert!(none_overlap(&s, eye));
    }

    #[test]
    fn worst_cases_six_screens_stacked_and_after_head_moves() {
        let h = head();
        let eye = h.translation;
        // all six on top of each other (a bad reset): bring one to the front, the rest spread out
        let mut s: Vec<Screen> = (0..6).map(|_| front(&h, 2.0, (2.0, 1.125), D)).collect();
        place(&mut s, 0, Place::Front, None, &h, MARGIN);
        assert!(
            none_overlap(&s, eye),
            "{:?}",
            overlaps(&s, eye, MARGIN * 0.99)
        );
        // mixed sizes and distances
        let mut s: Vec<Screen> = (0..6)
            .map(|i| {
                let mut x = front(
                    &h,
                    0.8 + 0.5 * i as f32,
                    (0.8 + 0.5 * i as f32, (0.8 + 0.5 * i as f32) * 0.5625),
                    1.5 + 0.6 * i as f32,
                );
                x = turned(&x, eye, Turn::Left, 0.05 * i as f32);
                x
            })
            .collect();
        place(&mut s, 3, Place::Front, None, &h, MARGIN);
        assert!(
            none_overlap(&s, eye),
            "{:?}",
            overlaps(&s, eye, MARGIN * 0.99)
        );
        // the user walks 1.5 m to the right and 1 m forward: from there screens overlap again; one command fixes it
        let mut s: Vec<Screen> = default_layout(&h, &six(2.0), D, MARGIN)
            .into_iter()
            .map(|x| x.1)
            .collect();
        let walked = Affine3A::from_translation(Vec3::new(1.5, 1.6, -1.0));
        assert!(
            !overlaps(&s, walked.translation, MARGIN).is_empty(),
            "the walk makes them overlap"
        );
        place(&mut s, 0, Place::Front, None, &walked, MARGIN);
        assert!(
            none_overlap(&s, walked.translation),
            "{:?}",
            overlaps(&s, walked.translation, MARGIN * 0.99)
        );
    }

    /// The default 6-screen layout as a picture: top-down (left) and first person (right, yaw/pitch in degrees as
    /// seen from the head). Written to `$OMARCHYVR_LAYOUT_SVG` (a file path) or the temp dir; numbers printed too.
    #[test]
    fn default_layout_diagram() {
        let h = head();
        let eye = h.translation;
        let l = default_layout(&h, &six(2.0), D, MARGIN);
        let names = [
            "1 main",
            "2 left",
            "3 right",
            "4 up",
            "5 up-left",
            "6 up-right",
        ];
        let colors = [
            "#2f7de1", "#1aa37a", "#d9822b", "#8e5bd8", "#c94f7c", "#2aa1b3",
        ];
        let mut svg = String::new();
        svg += "<svg xmlns='http://www.w3.org/2000/svg' width='1200' height='560' viewBox='0 0 1200 560' font-family='sans-serif'>\n";
        svg += "<rect width='1200' height='560' fill='#ffffff'/>\n";
        svg += "<text x='20' y='28' font-size='18' font-weight='bold'>Top-down (2 m wide 16:9 screens, 2.5 m away; dashed = upper row)</text>\n";
        svg += "<text x='520' y='28' font-size='18' font-weight='bold'>First person (degrees right / up from straight ahead)</text>\n";
        // top-down: x right, z forward is up; 1 m = 80 px, head at (250, 420)
        let (ox, oy, k) = (250.0f32, 420.0f32, 80.0f32);
        svg += &format!(
            "<circle cx='{ox}' cy='{oy}' r='{}' fill='none' stroke='#ccc' stroke-dasharray='4 4'/>\n",
            D * k
        );
        svg += &format!(
            "<circle cx='{ox}' cy='{oy}' r='8' fill='#333'/><text x='{}' y='{}' font-size='13'>head</text>\n",
            ox + 10.0,
            oy + 20.0
        );
        let mut report = String::new();
        for (i, (slot, s)) in l.iter().enumerate() {
            let (r, _) = s.axes();
            let a = s.centre() - r * s.size.0 * 0.5 - eye;
            let b = s.centre() + r * s.size.0 * 0.5 - eye;
            let px = |p: Vec3A| (ox + p.x * k, oy + p.z * k);
            let (ax, ay) = px(a);
            let (bx, by) = px(b);
            // upper row drawn thinner and dashed (it sits above the lower one)
            let dash = if matches!(slot, Slot::Up | Slot::UpLeft | Slot::UpRight) {
                " stroke-dasharray='8 4'"
            } else {
                ""
            };
            svg += &format!(
                "<line x1='{ax:.1}' y1='{ay:.1}' x2='{bx:.1}' y2='{by:.1}' stroke='{}' stroke-width='6'{dash}/>\n",
                colors[i]
            );
            let (cx, cy) = px(s.centre() - eye);
            let lift = if dash.is_empty() { 22.0 } else { -12.0 };
            svg += &format!(
                "<text x='{:.1}' y='{:.1}' font-size='13' text-anchor='middle' fill='{}'>{}</text>\n",
                cx,
                cy - lift,
                colors[i],
                names[i]
            );
            let (y, p) = yaw_pitch(s.centre() - eye);
            report += &format!(
                "{}: yaw {:+.1} deg, pitch {:+.1} deg, distance {:.2} m, size {:.2} x {:.3} m\n",
                names[i],
                y.to_degrees(),
                p.to_degrees(),
                (s.centre() - eye).length(),
                s.size.0,
                s.size.1
            );
        }
        // first person: x = -yaw (left on the left), 1 deg = 4 px, centre (850, 330)
        let (fx, fy, kd) = (850.0f32, 330.0f32, 4.0f32);
        for g in (-80..=80).step_by(20) {
            let x = fx - g as f32 * kd;
            svg += &format!(
                "<line x1='{x}' y1='60' x2='{x}' y2='540' stroke='#eee'/><text x='{x}' y='552' font-size='11' text-anchor='middle'>{}</text>\n",
                -g
            );
        }
        for g in (-20..=60).step_by(20) {
            let y = fy - g as f32 * kd;
            svg += &format!(
                "<line x1='520' y1='{y}' x2='1180' y2='{y}' stroke='#eee'/><text x='1185' y='{y}' font-size='11'>{g}</text>\n"
            );
        }
        for (i, (_, s)) in l.iter().enumerate() {
            let c = s.corners();
            let mut pts = String::new();
            for e in 0..4 {
                for step in 0..=12 {
                    let p = c[e].lerp(c[(e + 1) % 4], step as f32 / 12.0);
                    let (y, pi) = yaw_pitch(p - eye);
                    pts += &format!(
                        "{:.1},{:.1} ",
                        fx - y.to_degrees() * kd,
                        fy - pi.to_degrees() * kd
                    );
                }
            }
            svg += &format!(
                "<polygon points='{pts}' fill='{}' fill-opacity='0.25' stroke='{}' stroke-width='2'/>\n",
                colors[i], colors[i]
            );
            let (y, p) = yaw_pitch(s.centre() - eye);
            svg += &format!(
                "<text x='{:.1}' y='{:.1}' font-size='14' text-anchor='middle'>{}</text>\n",
                fx - y.to_degrees() * kd,
                fy - p.to_degrees() * kd + 5.0,
                names[i]
            );
        }
        svg += &format!("<circle cx='{fx}' cy='{fy}' r='4' fill='#333'/>\n");
        svg += "</svg>\n";
        let path = std::env::var("OMARCHYVR_LAYOUT_SVG")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("omarchyvr-default-layout.svg"));
        std::fs::write(&path, svg).unwrap();
        std::fs::write(path.with_extension("txt"), &report).unwrap();
        println!("{report}written {}", path.display());
    }

    #[test]
    fn stable_over_many_commands() {
        let h = head();
        let eye = h.translation;
        let mut s: Vec<Screen> = default_layout(&h, &six(2.0), D, MARGIN)
            .into_iter()
            .map(|x| x.1)
            .collect();
        // a free layout is left alone: resolve moves nothing
        assert!(resolve(&mut s.clone(), &[false; 6], eye, MARGIN).is_empty());
        let cmds = [
            Place::Left,
            Place::Right,
            Place::Up,
            Place::Down,
            Place::Front,
            Place::Swap,
        ];
        let mut rng = 7u32;
        for k in 0..600 {
            rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let idx = (rng >> 8) as usize % 6;
            let what = cmds[(rng >> 16) as usize % cmds.len()];
            let r = ((rng >> 4) as usize % 6, what == Place::Swap);
            // the user turns the head (+-30 deg) and moves it (+-20 cm) between commands
            let turn = ((rng >> 3) % 100) as f32 / 100.0 - 0.5;
            let shift = Vec3::new(
                ((rng >> 11) % 40) as f32 / 100.0 - 0.2,
                1.6,
                ((rng >> 19) % 40) as f32 / 100.0 - 0.2,
            );
            let hk = if k % 3 == 0 {
                h
            } else {
                Affine3A::from_rotation_translation(Quat::from_rotation_y(turn), shift)
            };
            let ek = hk.translation;
            let before: Vec<f32> = s.iter().map(|x| yaw_pitch(x.centre() - ek).1).collect();
            let moved = place(&mut s, idx, what, r.1.then_some(r.0), &hk, MARGIN);
            if !none_overlap(&s, ek) {
                for (i, x) in s.iter().enumerate() {
                    let (y, p) = yaw_pitch(x.centre() - ek);
                    eprintln!(
                        "{i}: yaw {:.1} pitch {:.1} elev {:?}",
                        y.to_degrees(),
                        p.to_degrees(),
                        elevation_range(x, ek)
                    );
                }
            }
            assert!(
                none_overlap(&s, ek),
                "step {k} {what:?} {idx}: {:?}",
                overlaps(&s, ek, MARGIN * 0.49)
            );
            for (i, x) in s.iter().enumerate() {
                assert!(x.t.to_cols_array().iter().all(|v| v.is_finite()));
                // what this command moved stays within the elevation limit seen from here (a screen placed from an
                // earlier head position may be beyond it now: the test's head jumps up to 57 cm between commands)
                if moved.contains(&i) && !matches!(what, Place::Up | Place::Down | Place::Swap) {
                    let el = yaw_pitch(x.centre() - ek).1;
                    assert!(
                        el.abs() <= (MAX_ELEVATION + 0.02).max(before[i].abs() + 0.02),
                        "step {k} {what:?} {idx}: screen {i} at elevation {}",
                        el.to_degrees()
                    );
                }
            }
        }
        // the same command twice changes nothing the second time (no creeping)
        let mut a = s.clone();
        place(&mut a, 0, Place::Front, None, &h, MARGIN);
        let b = a.clone();
        place(&mut a, 0, Place::Front, None, &h, MARGIN);
        for (x, y) in a.iter().zip(&b) {
            assert!((x.centre() - y.centre()).length() < 1e-4);
        }
    }
}
