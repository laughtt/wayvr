//! Gaze lessons: a small correction of the gaze direction, learned from clicks.
//!
//! This is a Rust port of `LiveCorrection` / `PointerLessons` from Frametop (gaze/gazecal.py and gaze/ft-gazed,
//! https://github.com/Frametop/frametop). The method, the constants and the maths are theirs; the base correction
//! is zero here because Monado already applies the calibration.
//!
//! MIT License
//!
//! Copyright (c) 2026 DeeJanuz
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all
//! copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.
//!
//! Angles are head-relative degrees: yaw positive to the left, pitch positive up. A lesson says "the tracker put the
//! gaze at (hy, hp), the truth was (hy + dy, hp + dp)". Nothing in here can produce input: it only maps a direction
//! to a small correction and keeps the lessons on disk.

use std::{
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

/// Ridge strengths of the fit; the offset is held back (PointerLessons).
const RIDGE: [f64; 6] = [2.0, 0.5, 0.5, 0.5, 0.5, 0.5];
/// Degrees: how far a lesson's leftover reaches.
const KERNEL: f64 = 2.5;
/// Near one lesson half of its leftover is used, near several nearly all.
const SHRINK: f64 = 0.5;
/// Lessons: the weight of a lesson halves after this many newer ones.
const HALF_LIFE: f64 = 40.0;
/// Lessons kept.
const KEEP: usize = 150;
/// Degrees the features are extended past the range of the lessons.
const MARGIN: f64 = 3.0;
/// Lessons from before the headset went on again count this much.
const OLD_WEAR: f64 = 0.3;

/// One lesson.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lesson {
    /// Unix seconds.
    pub time: f64,
    pub hy: f64,
    pub hp: f64,
    /// The whole error there: truth minus raw.
    pub dy: f64,
    pub dp: f64,
    pub wy: f64,
    pub wp: f64,
    /// "refine", "laser" or "check".
    #[serde(default)]
    pub how: String,
}

/// Solve a small linear system (Gaussian elimination with pivoting); zeros if singular.
fn solve(m: &[[f64; 6]; 6], b: &[f64; 6]) -> [f64; 6] {
    const N: usize = 6;
    let mut a = [[0.0f64; N + 1]; N];
    for i in 0..N {
        a[i][..N].copy_from_slice(&m[i]);
        a[i][N] = b[i];
    }
    for c in 0..N {
        let mut piv = c;
        for r in c..N {
            if a[r][c].abs() > a[piv][c].abs() {
                piv = r;
            }
        }
        if a[piv][c].abs() < 1e-12 {
            return [0.0; N];
        }
        a.swap(c, piv);
        for r in 0..N {
            if r != c {
                let f = a[r][c] / a[c][c];
                for k in c..=N {
                    a[r][k] -= f * a[c][k];
                }
            }
        }
    }
    let mut out = [0.0; N];
    for i in 0..N {
        out[i] = a[i][N] / a[i][i];
    }
    out
}

/// The learned correction (Frametop `LiveCorrection` with the `PointerLessons` ridge, base correction zero).
#[derive(Debug, Clone, Default)]
pub struct Correction {
    samples: Vec<Lesson>,
    wear_time: Option<f64>,
    cy: [f64; 6],
    cp: [f64; 6],
    /// (hy, hp, leftover yaw, leftover pitch, wy, wp) with the decay already in the weights.
    left: Vec<(f64, f64, f64, f64, f64, f64)>,
    range: Option<[f64; 4]>,
}

impl Correction {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn lessons(&self) -> &Vec<Lesson> {
        &self.samples
    }

    pub const fn len(&self) -> usize {
        self.samples.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Lessons recorded before this time count `OLD_WEAR` as much.
    pub fn set_wear_time(&mut self, t: Option<f64>) {
        self.wear_time = t;
        self.refit();
    }

    pub fn set_lessons(&mut self, lessons: Vec<Lesson>) {
        let skip = lessons.len().saturating_sub(KEEP);
        self.samples = lessons.into_iter().skip(skip).collect();
        self.refit();
    }

    pub fn clear(&mut self) {
        self.samples.clear();
        self.refit();
    }

    pub fn add(&mut self, lesson: Lesson) {
        self.samples.push(lesson);
        let skip = self.samples.len().saturating_sub(KEEP);
        if skip > 0 {
            self.samples.drain(..skip);
        }
        self.refit();
    }

    fn features(&self, hy: f64, hp: f64) -> [f64; 6] {
        let (mut hy, mut hp) = (hy, hp);
        if let Some([y0, y1, p0, p1]) = self.range {
            hy = hy.clamp(y0 - MARGIN, y1 + MARGIN);
            hp = hp.clamp(p0 - MARGIN, p1 + MARGIN);
        }
        let (x, y) = (hy / 30.0, hp / 30.0);
        [1.0, x, y, x * y, x * x, y * y]
    }

    fn refit(&mut self) {
        self.cy = [0.0; 6];
        self.cp = [0.0; 6];
        self.left.clear();
        self.range = None;
        let n = self.samples.len();
        if n == 0 {
            return;
        }
        let mut range = [f64::MAX, f64::MIN, f64::MAX, f64::MIN];
        for s in &self.samples {
            range[0] = range[0].min(s.hy);
            range[1] = range[1].max(s.hy);
            range[2] = range[2].min(s.hp);
            range[3] = range[3].max(s.hp);
        }
        self.range = Some(range);

        // (lesson index, features, decay)
        let rows: Vec<([f64; 6], f64)> = self
            .samples
            .iter()
            .enumerate()
            .map(|(k, s)| {
                let mut decay = 0.5f64.powf((n - 1 - k) as f64 / HALF_LIFE);
                if self.wear_time.is_some_and(|w| s.time < w) {
                    decay *= OLD_WEAR;
                }
                (self.features(s.hy, s.hp), decay)
            })
            .collect();

        for axis in 0..2 {
            let mut m = [[0.0f64; 6]; 6];
            for (r, row) in m.iter_mut().enumerate() {
                row[r] = RIDGE[r];
            }
            let mut b = [0.0f64; 6];
            for (s, (f, decay)) in self.samples.iter().zip(&rows) {
                let (w, target) = if axis == 0 {
                    (decay * s.wy, s.dy)
                } else {
                    (decay * s.wp, s.dp)
                };
                for r in 0..6 {
                    b[r] += w * f[r] * target;
                    for c in 0..6 {
                        m[r][c] += w * f[r] * f[c];
                    }
                }
            }
            let sol = solve(&m, &b);
            if axis == 0 {
                self.cy = sol;
            } else {
                self.cp = sol;
            }
        }

        let mut left = Vec::with_capacity(n);
        for (s, (f, decay)) in self.samples.iter().zip(&rows) {
            let ly = s.dy - dot(&self.cy, f);
            let lp = s.dp - dot(&self.cp, f);
            left.push((s.hy, s.hp, ly, lp, s.wy * decay, s.wp * decay));
        }
        self.left = left;
    }

    /// The correction (yaw, pitch degrees) to add to a raw gaze at head-relative (hy, hp).
    pub fn get(&self, hy: f64, hp: f64) -> (f64, f64) {
        if self.samples.is_empty() {
            return (0.0, 0.0);
        }
        let f = self.features(hy, hp);
        let cy = dot(&self.cy, &f);
        let cp = dot(&self.cp, &f);
        let k2 = 2.0 * KERNEL * KERNEL;
        let (mut sy, mut sp, mut wy, mut wp) = (0.0, 0.0, 0.0, 0.0);
        for &(y, p, ly, lp, ay, ap) in &self.left {
            let d2 = (y - hy).powi(2) + (p - hp).powi(2);
            if d2 > 9.0 * k2 {
                continue;
            }
            let g = (-d2 / k2).exp();
            sy += g * ay * ly;
            wy += g * ay;
            sp += g * ap * lp;
            wp += g * ap;
        }
        (cy + sy / (wy + SHRINK), cp + sp / (wp + SHRINK))
    }

    /// The offset term of the fit (yaw, pitch degrees).
    pub const fn offset(&self) -> (f64, f64) {
        (self.cy[0], self.cp[0])
    }
}

fn dot(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------------------------------------------
// Persistence

#[derive(Debug, Serialize, Deserialize)]
struct FileData {
    version: u32,
    key: String,
    lessons: Vec<Lesson>,
    #[serde(default)]
    taken: u32,
    #[serde(default)]
    refused: u32,
    #[serde(default)]
    last_error_deg: Option<f64>,
}

/// The lessons, their counters and the file they live in. Every IO error is logged once and ignored.
#[derive(Debug, Default)]
pub struct LessonStore {
    pub correction: Correction,
    pub key: String,
    pub taken: u32,
    pub refused: u32,
    pub last_error_deg: Option<f64>,
    path: Option<PathBuf>,
    warned: bool,
}

pub fn unix_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// `$XDG_STATE_HOME/omarchyvr/gaze-lessons.json`, else `~/.local/state/omarchyvr/gaze-lessons.json`.
pub fn default_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_STATE_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .filter(|v| !v.is_empty())
                .map(|h| PathBuf::from(h).join(".local/state"))
        })?;
    Some(base.join("omarchyvr").join("gaze-lessons.json"))
}

impl LessonStore {
    /// No file at all (tests, and the state before the config is known).
    pub fn in_memory() -> Self {
        Self::default()
    }

    /// Load from `path`; a missing or corrupt file is an empty store. `wear_time` is when this process started.
    pub fn load(path: Option<PathBuf>, wear_time: f64) -> Self {
        let mut s = Self {
            path,
            ..Self::default()
        };
        s.correction.wear_time = Some(wear_time);
        if let Some(p) = s.path.clone() {
            match std::fs::read_to_string(&p) {
                Ok(text) => match serde_json::from_str::<FileData>(&text) {
                    Ok(d) if d.version == 1 => {
                        s.key = d.key;
                        s.taken = d.taken;
                        s.refused = d.refused;
                        s.last_error_deg = d.last_error_deg;
                        s.correction.set_lessons(d.lessons);
                    }
                    Ok(d) => s.warn(&format!("unknown lessons file version {}", d.version)),
                    Err(e) => s.warn(&format!("lessons file unreadable ({e}); starting empty")),
                },
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => s.warn(&format!("lessons file not read ({e}); starting empty")),
            }
        }
        s
    }

    fn warn(&mut self, msg: &str) {
        if !self.warned {
            self.warned = true;
            log::warn!("gaze lessons: {msg}");
        }
    }

    /// Stored lessons under another key are discarded (the file is kept as `.old`); the store starts empty.
    pub fn check_key(&mut self, key: &str) {
        if self.key == key {
            return;
        }
        if !self.correction.is_empty() || self.taken > 0 || self.refused > 0 {
            log::info!("gaze lessons: calibration key changed, the old lessons are set aside");
            if let Some(p) = self.path.clone() {
                let mut old = p.clone().into_os_string();
                old.push(".old");
                if let Err(e) = std::fs::rename(&p, PathBuf::from(old)) {
                    self.warn(&format!("old lessons not set aside ({e})"));
                }
            }
        }
        self.key = key.to_string();
        self.correction.clear();
        self.taken = 0;
        self.refused = 0;
        self.last_error_deg = None;
    }

    pub fn forget(&mut self) {
        self.correction.clear();
        self.taken = 0;
        self.refused = 0;
        self.last_error_deg = None;
        if let Some(p) = &self.path
            && let Err(e) = std::fs::remove_file(p)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            let msg = format!("lessons file not removed ({e})");
            self.warn(&msg);
        }
    }

    pub fn add(&mut self, lesson: Lesson, error_deg: f64) {
        self.correction.add(lesson);
        self.taken += 1;
        self.last_error_deg = Some(error_deg);
        self.save();
    }

    pub fn refuse(&mut self) {
        self.refused += 1;
        self.save();
    }

    /// Atomic tmp + rename write; never fails the caller.
    pub fn save(&mut self) {
        let Some(path) = self.path.clone() else {
            return;
        };
        let data = FileData {
            version: 1,
            key: self.key.clone(),
            lessons: self.correction.lessons().clone(),
            taken: self.taken,
            refused: self.refused,
            last_error_deg: self.last_error_deg,
        };
        if let Err(e) = write_atomic(&path, &data) {
            self.warn(&format!("lessons not saved ({e})"));
        }
    }
}

fn write_atomic(path: &Path, data: &FileData) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let json = serde_json::to_vec_pretty(data).map_err(std::io::Error::other)?;
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&json)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct Golden {
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        name: String,
        wear_time: Option<f64>,
        lessons: Vec<Lesson>,
        probes: Vec<[f64; 4]>,
    }

    #[test]
    fn matches_frametop_golden() {
        let g: Golden = serde_json::from_str(include_str!("gaze_lessons_golden.json")).unwrap();
        assert!(g.cases.len() >= 5);
        let mut worst = 0.0f64;
        let mut probes = 0;
        for c in &g.cases {
            let mut corr = Correction::new();
            corr.set_wear_time(c.wear_time);
            for l in &c.lessons {
                corr.add(l.clone());
            }
            for p in &c.probes {
                let (y, pi) = corr.get(p[0], p[1]);
                let e = (y - p[2]).abs().max((pi - p[3]).abs());
                assert!(e < 1e-6, "case {} probe {p:?}: got ({y}, {pi}), err {e}", c.name);
                worst = worst.max(e);
                probes += 1;
            }
        }
        println!("golden: {probes} probes, max error {worst:e}");
    }

    #[test]
    fn empty_is_zero_and_solve_singular_is_zero() {
        assert_eq!(Correction::new().get(3.0, 4.0), (0.0, 0.0));
        assert_eq!(solve(&[[0.0; 6]; 6], &[1.0; 6]), [0.0; 6]);
    }

    fn lesson(t: f64) -> Lesson {
        Lesson {
            time: t,
            hy: 5.0,
            hp: -3.0,
            dy: 1.0,
            dp: -0.5,
            wy: 1.0,
            wp: 1.0,
            how: "check".into(),
        }
    }

    #[test]
    fn store_round_trip_key_and_corruption() {
        let dir = std::env::temp_dir().join(format!("gaze-lessons-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("gaze-lessons.json");

        let mut s = LessonStore::load(Some(path.clone()), 0.0);
        s.check_key("k1");
        s.add(lesson(10.0), 1.5);
        s.refuse();
        let before = s.correction.get(5.0, -3.0);
        assert!(before.0 > 0.1);

        let s2 = LessonStore::load(Some(path.clone()), 0.0);
        assert_eq!((s2.key.as_str(), s2.taken, s2.refused), ("k1", 1, 1));
        assert_eq!(s2.correction.get(5.0, -3.0), before);

        // another key: old file set aside, start empty
        let mut s3 = LessonStore::load(Some(path.clone()), 0.0);
        s3.check_key("k2");
        assert!(s3.correction.is_empty() && !path.exists());
        assert!(dir.join("gaze-lessons.json.old").exists());

        // corrupt file: empty, no panic
        std::fs::write(&path, b"{ not json").unwrap();
        let s4 = LessonStore::load(Some(path.clone()), 0.0);
        assert!(s4.correction.is_empty());

        // forget removes the file
        let mut s5 = LessonStore::load(Some(path.clone()), 0.0);
        s5.add(lesson(1.0), 0.0);
        assert!(path.exists());
        s5.forget();
        assert!(!path.exists() && s5.correction.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn old_wear_counts_less() {
        let mut a = Correction::new();
        a.set_wear_time(Some(100.0));
        a.add(lesson(50.0));
        let mut b = Correction::new();
        b.set_wear_time(Some(0.0));
        b.add(lesson(50.0));
        assert!(a.get(5.0, -3.0).0 < b.get(5.0, -3.0).0);
    }
}
