//! Pure transform maths for desk-mode grabbing (push/pull, two-hand resize, sanitising, default placement).
//! Everything keeps a transform as `uniform scale * rotation + translation`.

use glam::{Affine3A, Quat, Vec3, Vec3A};

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub size_min: f32,
    pub size_max: f32,
    pub dist_min: f32,
    pub dist_max: f32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            size_min: 0.3,
            size_max: 6.0,
            dist_min: 0.4,
            dist_max: 8.0,
        }
    }
}

/// Mean axis length.
pub fn uniform_scale(t: &Affine3A) -> f32 {
    (t.matrix3.x_axis.length() + t.matrix3.y_axis.length() + t.matrix3.z_axis.length()) / 3.0
}

/// True when the basis is (within `tol`) a rotation times one uniform scale, finite, positive scale, right-handed.
pub fn is_orthonormal_uniform(t: &Affine3A, tol: f32) -> bool {
    let a: [f32; 12] = t.to_cols_array();
    if a.iter().any(|v| !v.is_finite()) {
        return false;
    }
    let (x, y, z) = (t.matrix3.x_axis, t.matrix3.y_axis, t.matrix3.z_axis);
    let s = uniform_scale(t);
    if s <= 1e-4 {
        return false;
    }
    let (lx, ly, lz) = (x.length() / s, y.length() / s, z.length() / s);
    let dots = [x.dot(y), x.dot(z), y.dot(z)];
    (lx - 1.0).abs() <= tol
        && (ly - 1.0).abs() <= tol
        && (lz - 1.0).abs() <= tol
        && dots.iter().all(|d| d.abs() / (s * s) <= tol)
        && x.cross(y).dot(z) > 0.0
}

/// Gram-Schmidt: rebuild a clean `scale * rotation` from x and y axes, scale clamped to the size limits.
/// None when the input is not finite or degenerate.
pub fn orthonormalize(t: &Affine3A, lim: &Limits) -> Option<Affine3A> {
    if t.to_cols_array().iter().any(|v| !v.is_finite()) {
        return None;
    }
    let s = uniform_scale(t);
    if s <= 1e-4 {
        return None;
    }
    let x = t.matrix3.x_axis;
    if x.length() < 1e-6 {
        return None;
    }
    let x = x.normalize();
    let y = t.matrix3.y_axis - x * x.dot(t.matrix3.y_axis);
    if y.length() < 1e-6 {
        return None;
    }
    let y = y.normalize();
    let z = x.cross(y);
    let s = s.clamp(lim.size_min, lim.size_max);
    Some(Affine3A::from_cols(x * s, y * s, z * s, t.translation))
}

/// Saved world-locked (Static) pose: orthonormalise, clamp scale and the distance from the origin.
pub fn sanitize_static(t: &Affine3A, lim: &Limits) -> Option<Affine3A> {
    let mut o = orthonormalize(t, lim)?;
    let len = o.translation.length();
    if len < 1e-3 {
        return None;
    }
    o.translation *= len.clamp(lim.dist_min, lim.dist_max) / len;
    Some(o)
}

/// Stick push/pull: changes `offset` (hand to overlay) distance by `exp(y * rate * dt)`, clamped. With `keep` the
/// overlay scale follows the distance ratio (clamped to the size limits). Returns the factor to apply to the overlay
/// basis (and to the offset basis, done here): the caller multiplies its transform basis by it.
pub fn push_pull(offset: &mut Affine3A, cur_scale: f32, y: f32, dt: f32, rate: f32, keep: bool, lim: &Limits) -> f32 {
    let len = offset.translation.length().max(0.01);
    let want = (len * (y * rate * dt).exp()).clamp(lim.dist_min, lim.dist_max);
    offset.translation *= want / len;
    if !keep {
        return 1.0;
    }
    let f = (cur_scale * want / len).clamp(lim.size_min, lim.size_max) / cur_scale;
    offset.matrix3 = offset.matrix3.mul_scalar(f);
    f
}

/// Two-hand gesture result from the start state (`t0`, hand vector `v0`, midpoint `m0`) and the current hands.
pub fn two_hand(t0: &Affine3A, v0: Vec3A, m0: Vec3A, v: Vec3A, m: Vec3A, lim: &Limits) -> Affine3A {
    let scale0 = uniform_scale(t0);
    let d0 = v0.length().max(0.02);
    let scale = (scale0 * v.length() / d0).clamp(lim.size_min, lim.size_max);
    let yaw = |a: Vec3A| a.x.atan2(a.z);
    let dq = if Vec3A::new(v.x, 0.0, v.z).length() > 0.05 && Vec3A::new(v0.x, 0.0, v0.z).length() > 0.05 {
        Quat::from_rotation_y(yaw(v) - yaw(v0))
    } else {
        Quat::IDENTITY
    };
    let rot0 = Quat::from_mat3a(&t0.matrix3.div_scalar(scale0));
    let pos = m + dq * (t0.translation - m0);
    Affine3A::from_scale_rotation_translation(Vec3::ONE * scale, dq * rot0, pos.into())
}

/// Default placement: in front of the head (yaw only), `slot` shifts screens sideways.
pub fn default_front(hmd: &Affine3A, slot: usize) -> Affine3A {
    let fwd = {
        let f = -Vec3A::from(hmd.matrix3.z_axis);
        let f = Vec3A::new(f.x, 0.0, f.z);
        if f.length() < 1e-3 { Vec3A::NEG_Z } else { f.normalize() }
    };
    let theta = (-fwd.x).atan2(-fwd.z);
    let rot = Quat::from_rotation_y(theta);
    let scale = 2.0;
    let right = rot * Vec3A::X;
    let pos = Vec3A::from(hmd.translation) + fwd * 2.5 + right * (slot as f32 * scale * 1.05);
    Affine3A::from_scale_rotation_translation(Vec3::ONE * scale, rot, pos.into())
}

/// Desk layout slot `i` of `n`: slot 0 in the middle, then alternately left and right on an arc around the head
/// (yaw only), each screen facing the head, `width` metres wide at `distance` metres.
pub fn desk_arc(hmd: &Affine3A, i: usize, n: usize, width: f32, distance: f32) -> Affine3A {
    let fwd = {
        let f = -Vec3A::from(hmd.matrix3.z_axis);
        let f = Vec3A::new(f.x, 0.0, f.z);
        if f.length() < 1e-3 { Vec3A::NEG_Z } else { f.normalize() }
    };
    let theta = (-fwd.x).atan2(-fwd.z);
    // angle between neighbours: the screen width plus a 5 % gap, seen from the head
    let step = 2.0 * ((width * 0.525) / distance.max(0.1)).atan();
    let k = i.div_ceil(2) as f32;
    let side = if i % 2 == 1 { 1.0 } else { -1.0 }; // 1 = left (positive yaw), 2 = right
    let yaw = theta + if i == 0 || n <= 1 { 0.0 } else { side * k * step };
    let rot = Quat::from_rotation_y(yaw);
    let dir = rot * Vec3A::NEG_Z;
    let pos = Vec3A::from(hmd.translation) + dir * distance;
    Affine3A::from_scale_rotation_translation(Vec3::ONE * width, rot, pos.into())
}

/// Where a neighbour goes relative to the main screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    Left,
    Right,
    Up,
}

/// Pose of the screen next to `main` (`main_size` = displayed width, height in metres): the whole affine is rotated
/// about an axis through `pivot` (the head) - left/right about the main's up axis, up about its right axis - by the
/// angle that leaves `gap` metres between the facing edges (flat screens, both facing the pivot). Scale is kept.
pub fn desk_neighbour(main: &Affine3A, main_size: (f32, f32), pivot: Vec3A, slot: Slot, gap: f32) -> Affine3A {
    let right = Vec3A::from(main.matrix3.x_axis).normalize_or_zero();
    let up = Vec3A::from(main.matrix3.y_axis).normalize_or_zero();
    let (axis, extent, sign) = match slot {
        Slot::Left => (up, main_size.0, 1.0),
        Slot::Right => (up, main_size.0, -1.0),
        Slot::Up => (right, main_size.1, 1.0),
    };
    if axis == Vec3A::ZERO {
        return *main;
    }
    let d = (Vec3A::from(main.translation) - pivot).length().max(0.1);
    let half = extent.max(0.01) * 0.5;
    // the facing edge points sit at radius r, angle phi off their screen's centre line; they are 2 r sin(dθ/2) apart
    let r = (d * d + half * half).sqrt();
    let phi = (half / d).atan();
    let g = gap.max(0.0);
    let theta = 2.0 * phi + 2.0 * (g / (2.0 * r)).clamp(0.0, 1.0).asin();
    let rot = Affine3A::from_quat(Quat::from_axis_angle(axis.into(), sign * theta));
    Affine3A::from_translation(pivot.into()) * rot * Affine3A::from_translation((-pivot).into()) * *main
}

/// Why a live (world-locked) overlay should be put back, or None if it is fine.
pub fn invalid_reason(t: &Affine3A, hmd: &Affine3A, lim: &Limits) -> Option<&'static str> {
    if t.to_cols_array().iter().any(|v| !v.is_finite()) {
        return Some("transform is not finite");
    }
    if !is_orthonormal_uniform(t, 0.05) {
        return Some("transform is skewed or mirrored");
    }
    let s = uniform_scale(t);
    if s < lim.size_min * 0.9 || s > lim.size_max * 1.1 {
        return Some("scale out of limits");
    }
    let d = (Vec3A::from(t.translation) - Vec3A::from(hmd.translation)).length();
    if d < lim.dist_min * 0.5 {
        return Some("too close to the head");
    }
    if d > lim.dist_max * 1.5 {
        return Some("too far from the head");
    }
    None
}

/// Angle (deg) between the head forward direction and the direction to `t`.
pub fn view_angle_deg(t: &Affine3A, hmd: &Affine3A) -> f32 {
    let dir = Vec3A::from(t.translation) - Vec3A::from(hmd.translation);
    let fwd = -Vec3A::from(hmd.matrix3.z_axis);
    if dir.length() < 1e-4 || fwd.length() < 1e-4 {
        return 0.0;
    }
    dir.normalize().dot(fwd.normalize()).clamp(-1.0, 1.0).acos().to_degrees()
}

#[cfg(test)]
mod tests {
    #[test]
    fn desk_arc_middle_left_right() {
        let hmd = Affine3A::IDENTITY;
        let m = desk_arc(&hmd, 0, 3, 2.0, 2.5);
        assert!((Vec3A::from(m.translation) - Vec3A::new(0.0, 0.0, -2.5)).length() < 1e-4);
        let l = Vec3A::from(desk_arc(&hmd, 1, 3, 2.0, 2.5).translation);
        let r = Vec3A::from(desk_arc(&hmd, 2, 3, 2.0, 2.5).translation);
        assert!(l.x < -1.5 && r.x > 1.5, "{l} {r}");
        assert!((l.length() - 2.5).abs() < 1e-4 && (r.length() - 2.5).abs() < 1e-4);
        // each faces the head: its -Z axis points away from the head
        let t = desk_arc(&hmd, 1, 3, 2.0, 2.5);
        let back = -Vec3A::from(t.matrix3.z_axis).normalize();
        assert!(back.dot(l.normalize()) > 0.99);
    }

    use super::*;

    fn check_neighbours(hmd: &Affine3A) {
        let main = desk_arc(hmd, 0, 1, 2.0, 2.5);
        let size = (2.0, 2.0 * 9.0 / 16.0);
        let pivot = Vec3A::from(hmd.translation);
        let mr = Vec3A::from(main.matrix3.x_axis).normalize();
        let mu = Vec3A::from(main.matrix3.y_axis).normalize();
        for (slot, gap) in [(Slot::Left, 0.05), (Slot::Right, 0.05), (Slot::Up, 0.05), (Slot::Left, 0.3)] {
            let n = desk_neighbour(&main, size, pivot, slot, gap);
            let c = Vec3A::from(n.translation);
            let rel = c - Vec3A::from(main.translation);
            assert!(((c - pivot).length() - 2.5).abs() < 1e-3, "{slot:?} distance");
            // faces the pivot: -Z points away from the head
            let back = -Vec3A::from(n.matrix3.z_axis).normalize();
            assert!(back.dot((c - pivot).normalize()) > 0.99, "{slot:?} facing");
            assert!((n.matrix3.x_axis.length() - 2.0).abs() < 1e-4, "{slot:?} scale");
            let (axis, ext) = match slot {
                Slot::Left | Slot::Right => (mr, size.0),
                Slot::Up => (mu, size.1),
            };
            let s = if slot == Slot::Left { -1.0 } else { 1.0 };
            assert!(rel.dot(axis) * s > 0.0, "{slot:?} side");
            let na = Vec3A::from(if slot == Slot::Up { n.matrix3.y_axis } else { n.matrix3.x_axis }).normalize();
            let e_main = Vec3A::from(main.translation) + axis * (s * ext * 0.5);
            let e_n = c - na * (s * ext * 0.5);
            let g = (e_n - e_main).length();
            assert!((g - gap).abs() < 0.03, "{slot:?} gap {g} vs {gap}");
        }
    }

    #[test]
    fn desk_neighbour_identity() {
        let hmd = Affine3A::IDENTITY;
        let main = desk_arc(&hmd, 0, 1, 2.0, 2.5);
        let size = (2.0, 1.125);
        let l = Vec3A::from(desk_neighbour(&main, size, Vec3A::ZERO, Slot::Left, 0.05).translation);
        let r = Vec3A::from(desk_neighbour(&main, size, Vec3A::ZERO, Slot::Right, 0.05).translation);
        let u = Vec3A::from(desk_neighbour(&main, size, Vec3A::ZERO, Slot::Up, 0.05).translation);
        assert!(l.x < -1.7 && r.x > 1.7 && u.y > 1.0, "{l} {r} {u}");
        check_neighbours(&hmd);
    }

    #[test]
    fn desk_neighbour_yawed_pitched_head() {
        let hmd = Affine3A::from_rotation_translation(
            Quat::from_rotation_y(0.9) * Quat::from_rotation_x(-0.3),
            Vec3::new(0.4, 1.6, -0.7),
        );
        check_neighbours(&hmd);
    }

    fn rot_scale(s: f32, yaw: f32, pos: Vec3) -> Affine3A {
        Affine3A::from_scale_rotation_translation(Vec3::ONE * s, Quat::from_rotation_y(yaw), pos)
    }

    #[test]
    fn push_pull_cycles_stay_uniform() {
        let lim = Limits::default();
        let pose = Affine3A::from_rotation_translation(Quat::from_rotation_x(0.3), Vec3::new(0.1, 1.2, -0.3));
        let mut t = rot_scale(1.5, 0.4, Vec3::new(0.2, 1.4, -2.0));
        let mut off = pose.inverse() * t;
        for i in 0..5000 {
            let y = if (i / 200) % 2 == 0 { 1.0 } else { -1.0 };
            let s = uniform_scale(&t);
            let f = push_pull(&mut off, s, y, 1.0 / 72.0, 0.8, true, &lim);
            t = pose * off;
            let _ = f;
            assert!(is_orthonormal_uniform(&t, 1e-3), "iter {i}");
            let s = uniform_scale(&t);
            assert!(s >= lim.size_min - 1e-3 && s <= lim.size_max + 1e-3);
            let d = off.translation.length();
            assert!(d >= lim.dist_min - 1e-3 && d <= lim.dist_max + 1e-3);
        }
    }

    #[test]
    fn two_hand_cycles_stay_uniform() {
        let lim = Limits::default();
        let t0 = rot_scale(1.5, 0.4, Vec3::new(0.2, 1.4, -2.0));
        let (v0, m0) = (Vec3A::new(0.4, 0.0, 0.0), Vec3A::new(0.0, 1.2, -0.4));
        for i in 0..2000 {
            let k = (i as f32 * 0.01).sin();
            let v = Vec3A::new(0.4 + 0.3 * k, 0.1 * k, 0.2 * k);
            let m = m0 + Vec3A::new(0.1 * k, 0.0, 0.05);
            let t = two_hand(&t0, v0, m0, v, m, &lim);
            assert!(is_orthonormal_uniform(&t, 1e-3));
            let s = uniform_scale(&t);
            assert!(s >= lim.size_min - 1e-3 && s <= lim.size_max + 1e-3);
        }
        // identity gesture keeps the pose
        let t = two_hand(&t0, v0, m0, v0, m0, &lim);
        assert!((t.translation - t0.translation).length() < 1e-4);
        assert!((uniform_scale(&t) - 1.5).abs() < 1e-4);
    }

    #[test]
    fn sanitize_fixes_skew_and_clamps() {
        let lim = Limits::default();
        // the bad pose from the live bug is valid rotation*scale 4.2 at 4.7 m
        let bad = Affine3A::from_cols(
            Vec3A::new(3.06697, 1.19e-7, -2.933),
            Vec3A::new(0.9405, 4.0196, 0.98346),
            Vec3A::new(2.7782, -1.3608, 2.905),
            Vec3A::new(-3.2723, 2.2723, -2.6725),
        );
        assert!(is_orthonormal_uniform(&bad, 0.05));
        let skew = Affine3A::from_cols(Vec3A::new(5.0, 0.0, 0.0), Vec3A::new(1.0, 2.0, 0.0), Vec3A::new(0.0, 0.0, 9.0), Vec3A::new(0.0, 0.0, -30.0));
        assert!(!is_orthonormal_uniform(&skew, 0.05));
        let fixed = sanitize_static(&skew, &lim).unwrap();
        assert!(is_orthonormal_uniform(&fixed, 1e-3));
        assert!(uniform_scale(&fixed) <= lim.size_max + 1e-4);
        assert!((fixed.translation.length() - lim.dist_max).abs() < 1e-3);
        let nan = Affine3A::from_translation(Vec3::new(f32::NAN, 0.0, 0.0));
        assert!(sanitize_static(&nan, &lim).is_none());
        assert!(invalid_reason(&nan, &Affine3A::IDENTITY, &lim).is_some());
    }

    #[test]
    fn default_front_is_valid_and_in_front() {
        let lim = Limits::default();
        let hmd = Affine3A::from_rotation_translation(Quat::from_rotation_y(1.0), Vec3::new(1.0, 1.6, 2.0));
        for slot in 0..3 {
            let t = default_front(&hmd, slot);
            assert!(invalid_reason(&t, &hmd, &lim).is_none());
        }
        assert!(view_angle_deg(&default_front(&hmd, 0), &hmd) < 10.0);
    }
}
