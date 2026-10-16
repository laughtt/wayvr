//! Controller models (Work desk): the Oculus Touch v3 / Quest 2 glTF of the WebXR Input Profiles assets (MIT, see
//! `res/controllers/oculus-touch-v3/README.md`), drawn at each controller's OpenXR grip pose like SteamVR draws its
//! render model, with the trigger, squeeze, thumbstick and buttons moving the way the WebXR `motion-controllers`
//! library moves them (each response node interpolated between its min and max node by the input value).
//!
//! No OpenXR and no GPU in here (tests run without a runtime); `backend/openxr/controller_render.rs` draws it.
//! Also the pose math of the pointer ray and the debug axes ([`ray`], [`axes`]).

use std::collections::HashMap;

use anyhow::{Context, anyhow, bail};
use glam::{Affine3A, Quat, Vec3, Vec3A};

pub const LEFT_GLB: &[u8] = include_bytes!("../res/controllers/oculus-touch-v3/left.glb");
pub const RIGHT_GLB: &[u8] = include_bytes!("../res/controllers/oculus-touch-v3/right.glb");
pub const PROFILE_JSON: &str = include_str!("../res/controllers/oculus-touch-v3/profile.json");

/// Oculus Touch (Quest 2) aim -> grip, from the `openxr_aim` and `openxr_grip` components of SteamVR's
/// `oculus_quest2_controller_*` render model: `grip = aim * touch_aim_to_grip()` (60 deg pitch up, the grip 3.0 cm
/// below and 9.5 cm behind the ray origin). The Pimax driver uses the same transform (pimax-hands `touch.rs`); WayVR
/// only needs it for the tests and to explain the debug axes.
#[cfg_attr(not(test), allow(dead_code))] // tests and debugging
pub fn touch_aim_to_grip() -> Affine3A {
    let rot = |deg: f32| Quat::from_rotation_x(deg.to_radians());
    let aim = Affine3A::from_rotation_translation(
        rot(-39.4),
        Vec3::new(0.007, -0.038_947_66, 0.009_496_94),
    );
    let grip = Affine3A::from_rotation_translation(
        rot(20.6),
        Vec3::new(0.007, -0.001_829_41, 0.101_948_2),
    );
    aim.inverse() * grip
}

/// Where the WebXR model sits relative to the OpenXR grip, so it lands where SteamVR draws its own Quest 2 model.
/// Measured: with the WebXR model at SteamVR's `openxr_grip`, its bounding box in the Touch raw frame matches the
/// SteamVR model's in x and at the bottom but lies 2.45 cm further back (z 0.046..0.173 vs 0.020..0.150) and 2 mm
/// higher; this moves it forward by that much (raw frame offset rotated into the grip frame). Only box numbers of the
/// SteamVR model are used, not its data.
pub fn model_in_grip() -> Affine3A {
    let grip_from_raw = Quat::from_rotation_x((-20.6f32).to_radians());
    Affine3A::from_translation(grip_from_raw * Vec3::new(0.0, -0.002, -0.0245))
}

/// The pointer ray of a pose (OpenXR aim convention: origin at the pose, along its -Z).
#[cfg_attr(not(test), allow(dead_code))] // tests and debugging
pub fn ray(aim: &Affine3A) -> (Vec3A, Vec3A) {
    (
        aim.translation,
        aim.matrix3.mul_vec3a(Vec3A::NEG_Z).normalize(),
    )
}

/// Debug axes of a pose: (from, to, rgba) for +X red, +Y green, -Z blue (the ray direction), `len` metres long.
pub fn axes(pose: &Affine3A, len: f32) -> [(Vec3A, Vec3A, [f32; 4]); 3] {
    let o = pose.translation;
    let m = pose.matrix3;
    [
        (o, o + m.mul_vec3a(Vec3A::X) * len, [1.0, 0.15, 0.15, 1.0]),
        (o, o + m.mul_vec3a(Vec3A::Y) * len, [0.15, 1.0, 0.15, 1.0]),
        (o, o + m.mul_vec3a(Vec3A::NEG_Z) * len, [0.2, 0.4, 1.0, 1.0]),
    ]
}

/// Input values that move parts of the model.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct ControllerInputs {
    pub trigger: f32,
    pub squeeze: f32,
    /// OpenXR thumbstick (x right, y up), -1..1.
    pub stick: [f32; 2],
    pub stick_click: bool,
    /// A or X.
    pub primary: bool,
    /// B or Y.
    pub secondary: bool,
    pub menu: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputSel {
    Trigger,
    Squeeze,
    StickClick,
    StickX,
    StickY,
    Primary,
    Secondary,
    Menu,
    None,
}

impl InputSel {
    fn from_profile(component: &str, property: &str) -> Self {
        match (component, property) {
            ("xr-standard-trigger", "button") => Self::Trigger,
            ("xr-standard-squeeze", "button") => Self::Squeeze,
            ("xr-standard-thumbstick", "button") => Self::StickClick,
            ("xr-standard-thumbstick", "xAxis") => Self::StickX,
            ("xr-standard-thumbstick", "yAxis") => Self::StickY,
            ("a-button" | "x-button", "button") => Self::Primary,
            ("b-button" | "y-button", "button") => Self::Secondary,
            ("menu", "button") => Self::Menu,
            _ => Self::None,
        }
    }

    /// 0..1 like the WebXR library: buttons are their value, axes (v + 1) / 2 with the gamepad's y pointing down.
    fn value(self, i: &ControllerInputs) -> f32 {
        let b = |x: bool| if x { 1.0 } else { 0.0 };
        let v = match self {
            Self::Trigger => i.trigger,
            Self::Squeeze => i.squeeze,
            Self::StickClick => b(i.stick_click),
            Self::StickX => (i.stick[0] + 1.0) * 0.5,
            Self::StickY => (-i.stick[1] + 1.0) * 0.5,
            Self::Primary => b(i.primary),
            Self::Secondary => b(i.secondary),
            Self::Menu => b(i.menu),
            Self::None => 0.0,
        };
        if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
    }
}

#[derive(Debug, Clone, Copy)]
struct Trs {
    t: Vec3,
    r: Quat,
    s: Vec3,
}

impl Trs {
    fn affine(&self) -> Affine3A {
        Affine3A::from_scale_rotation_translation(self.s, self.r, self.t)
    }
}

#[derive(Debug, Clone)]
struct Node {
    name: String,
    rest: Trs,
    children: Vec<usize>,
    mesh: Option<usize>,
}

/// One drawable primitive (indexed triangles), positions/normals/uvs in its node's frame.
#[derive(Debug, Clone)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
struct Response {
    value: usize,
    min: usize,
    max: usize,
    input: InputSel,
}

/// A loaded controller model.
#[derive(Debug, Clone)]
pub struct ControllerModel {
    nodes: Vec<Node>,
    roots: Vec<usize>,
    pub meshes: Vec<MeshData>,
    /// The base colour texture as stored in the file (PNG), if any.
    pub base_color_png: Option<Vec<u8>>,
    responses: Vec<Response>,
}

/// A mesh to draw and its transform in grip space.
#[derive(Debug, Clone, Copy)]
pub struct DrawItem {
    pub mesh: usize,
    pub transform: Affine3A,
}

fn arr<const N: usize>(v: Option<&serde_json::Value>) -> Option<[f32; N]> {
    let a = v?.as_array()?;
    if a.len() != N {
        return None;
    }
    let mut out = [0f32; N];
    for (o, x) in out.iter_mut().zip(a) {
        *o = x.as_f64()? as f32;
    }
    Some(out)
}

struct Glb<'a> {
    json: serde_json::Value,
    bin: &'a [u8],
}

impl<'a> Glb<'a> {
    fn parse(b: &'a [u8]) -> anyhow::Result<Self> {
        let u32_at = |o: usize| -> anyhow::Result<u32> {
            Ok(u32::from_le_bytes(
                b.get(o..o + 4).context("glb: truncated")?.try_into()?,
            ))
        };
        if b.get(0..4) != Some(b"glTF".as_slice()) || u32_at(4)? != 2 {
            bail!("not a glTF 2 binary");
        }
        let mut off = 12;
        let (mut json, mut bin) = (None, None);
        while off + 8 <= b.len() {
            let len = u32_at(off)? as usize;
            let ty = u32_at(off + 4)?;
            let data = b.get(off + 8..off + 8 + len).context("glb: chunk past the end")?;
            match ty {
                0x4E4F_534A => json = Some(serde_json::from_slice(data)?),
                0x004E_4942 => bin = Some(data),
                _ => {}
            }
            off += 8 + len;
        }
        Ok(Self {
            json: json.context("glb: no JSON chunk")?,
            bin: bin.context("glb: no BIN chunk")?,
        })
    }

    fn view(&self, idx: usize) -> anyhow::Result<(&'a [u8], usize)> {
        let v = &self.json["bufferViews"][idx];
        let off = v["byteOffset"].as_u64().unwrap_or(0) as usize;
        let len = v["byteLength"].as_u64().context("bufferView without byteLength")? as usize;
        let stride = v["byteStride"].as_u64().unwrap_or(0) as usize;
        Ok((self.bin.get(off..off + len).context("bufferView past the BIN chunk")?, stride))
    }

    /// Accessor as f32 tuples of `n` components (FLOAT only).
    fn floats(&self, idx: usize, n: usize) -> anyhow::Result<Vec<f32>> {
        let a = &self.json["accessors"][idx];
        if a["componentType"].as_u64() != Some(5126) {
            bail!("accessor {idx}: want FLOAT");
        }
        let count = a["count"].as_u64().context("accessor count")? as usize;
        let (data, stride) = self.view(a["bufferView"].as_u64().context("sparse accessor")? as usize)?;
        let base = a["byteOffset"].as_u64().unwrap_or(0) as usize;
        let stride = if stride == 0 { 4 * n } else { stride };
        let mut out = Vec::with_capacity(count * n);
        for i in 0..count {
            for k in 0..n {
                let o = base + i * stride + 4 * k;
                let b = data.get(o..o + 4).context("accessor past its view")?;
                out.push(f32::from_le_bytes(b.try_into()?));
            }
        }
        Ok(out)
    }

    fn indices(&self, idx: usize) -> anyhow::Result<Vec<u32>> {
        let a = &self.json["accessors"][idx];
        let count = a["count"].as_u64().context("accessor count")? as usize;
        let (data, _) = self.view(a["bufferView"].as_u64().context("index view")? as usize)?;
        let base = a["byteOffset"].as_u64().unwrap_or(0) as usize;
        let size = match a["componentType"].as_u64() {
            Some(5121) => 1,
            Some(5123) => 2,
            Some(5125) => 4,
            t => bail!("index type {t:?}"),
        };
        (0..count)
            .map(|i| {
                let o = base + i * size;
                let b = data.get(o..o + size).context("indices past their view")?;
                Ok(match size {
                    1 => u32::from(b[0]),
                    2 => u32::from(u16::from_le_bytes([b[0], b[1]])),
                    _ => u32::from_le_bytes(b.try_into()?),
                })
            })
            .collect()
    }
}

impl ControllerModel {
    /// The built-in Touch v3 model of one hand.
    pub fn touch_v3(left: bool) -> anyhow::Result<Self> {
        Self::load(
            if left { LEFT_GLB } else { RIGHT_GLB },
            PROFILE_JSON,
            if left { "left" } else { "right" },
        )
    }

    /// A WebXR input profile model: the GLB and the profile JSON's layout for `hand`.
    pub fn load(glb: &[u8], profile_json: &str, hand: &str) -> anyhow::Result<Self> {
        let g = Glb::parse(glb)?;
        let j = &g.json;

        let mut nodes = Vec::new();
        for n in j["nodes"].as_array().context("no nodes")? {
            let (t, r, s) = if let Some(m) = n.get("matrix") {
                let m: [f32; 16] = arr(Some(m)).context("bad node matrix")?;
                let (s, r, t) = glam::Mat4::from_cols_array(&m).to_scale_rotation_translation();
                (t, r, s)
            } else {
                (
                    arr::<3>(n.get("translation")).map_or(Vec3::ZERO, Vec3::from),
                    arr::<4>(n.get("rotation")).map_or(Quat::IDENTITY, |q| {
                        Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize()
                    }),
                    arr::<3>(n.get("scale")).map_or(Vec3::ONE, Vec3::from),
                )
            };
            nodes.push(Node {
                name: n["name"].as_str().unwrap_or("").to_string(),
                rest: Trs { t, r, s },
                children: n["children"]
                    .as_array()
                    .map(|c| c.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect())
                    .unwrap_or_default(),
                mesh: n["mesh"].as_u64().map(|m| m as usize),
            });
        }
        if nodes.iter().flat_map(|n| &n.children).any(|&c| c >= nodes.len()) {
            bail!("node child out of range");
        }
        let scene = j["scene"].as_u64().unwrap_or(0) as usize;
        let roots: Vec<usize> = j["scenes"][scene]["nodes"]
            .as_array()
            .context("no scene")?
            .iter()
            .filter_map(|x| x.as_u64().map(|x| x as usize))
            .filter(|&x| x < nodes.len())
            .collect();

        // meshes: one primitive each in these assets; extra primitives are appended as their own meshes
        let mut meshes = Vec::new();
        let mut mesh_of = Vec::new(); // glTF mesh -> first MeshData, count
        for m in j["meshes"].as_array().context("no meshes")? {
            let first = meshes.len();
            for p in m["primitives"].as_array().context("no primitives")? {
                if p["mode"].as_u64().unwrap_or(4) != 4 {
                    continue; // triangles only
                }
                let at = &p["attributes"];
                let pos = g.floats(at["POSITION"].as_u64().context("no POSITION")? as usize, 3)?;
                let nv = pos.len() / 3;
                let nrm = match at["NORMAL"].as_u64() {
                    Some(i) => g.floats(i as usize, 3)?,
                    None => vec![0.0, 1.0, 0.0].repeat(nv),
                };
                let uv = match at["TEXCOORD_0"].as_u64() {
                    Some(i) => g.floats(i as usize, 2)?,
                    None => vec![0.0; nv * 2],
                };
                if nrm.len() != nv * 3 || uv.len() != nv * 2 {
                    bail!("attribute counts differ");
                }
                let indices = match p["indices"].as_u64() {
                    Some(i) => g.indices(i as usize)?,
                    None => (0..nv as u32).collect(),
                };
                if indices.iter().any(|&i| i as usize >= nv) {
                    bail!("index out of range");
                }
                meshes.push(MeshData {
                    positions: pos.chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
                    normals: nrm.chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
                    uvs: uv.chunks(2).map(|c| [c[0], c[1]]).collect(),
                    indices,
                });
            }
            mesh_of.push((first, meshes.len() - first));
        }
        // a node with a multi-primitive mesh: give it the first one and add siblings for the rest
        let mut extra = Vec::new();
        for (i, n) in nodes.iter_mut().enumerate() {
            if let Some(m) = n.mesh {
                let (first, count) = *mesh_of.get(m).context("node mesh out of range")?;
                n.mesh = (count > 0).then_some(first);
                for k in 1..count {
                    extra.push((i, first + k));
                }
            }
        }
        for (parent, mesh) in extra {
            let id = nodes.len();
            nodes.push(Node {
                name: String::new(),
                rest: Trs {
                    t: Vec3::ZERO,
                    r: Quat::IDENTITY,
                    s: Vec3::ONE,
                },
                children: vec![],
                mesh: Some(mesh),
            });
            nodes[parent].children.push(id);
        }

        // base colour texture (embedded PNG)
        let base_color_png = (|| {
            let tex = j["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"].as_u64()?;
            let img = j["textures"][tex as usize]["source"].as_u64()?;
            let im = &j["images"][img as usize];
            if im["mimeType"].as_str()? != "image/png" {
                return None;
            }
            let (data, _) = g.view(im["bufferView"].as_u64()? as usize).ok()?;
            Some(data.to_vec())
        })();

        // visual responses
        let by_name: HashMap<&str, usize> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.name.is_empty())
            .map(|(i, n)| (n.name.as_str(), i))
            .collect();
        let profile: serde_json::Value = serde_json::from_str(profile_json)?;
        let comps = profile["layouts"][hand]["components"]
            .as_object()
            .ok_or_else(|| anyhow!("profile has no {hand} layout"))?;
        let mut responses = Vec::new();
        for (cid, c) in comps {
            let Some(vr) = c["visualResponses"].as_object() else {
                continue;
            };
            for r in vr.values() {
                if r["valueNodeProperty"].as_str().unwrap_or("transform") != "transform" {
                    continue;
                }
                let input = InputSel::from_profile(cid, r["componentProperty"].as_str().unwrap_or(""));
                let find = |k: &str| r[k].as_str().and_then(|n| by_name.get(n).copied());
                if let (Some(value), Some(min), Some(max)) =
                    (find("valueNodeName"), find("minNodeName"), find("maxNodeName"))
                {
                    responses.push(Response {
                        value,
                        min,
                        max,
                        input,
                    });
                }
            }
        }

        Ok(Self {
            nodes,
            roots,
            meshes,
            base_color_png,
            responses,
        })
    }

    /// Number of animated parts found (trigger, squeeze, stick x/y/click, buttons).
    #[cfg_attr(not(test), allow(dead_code))] // tests and debugging
    pub fn response_count(&self) -> usize {
        self.responses.len()
    }

    /// The meshes to draw for these inputs, each with its transform in grip space.
    pub fn pose(&self, inputs: &ControllerInputs) -> Vec<DrawItem> {
        let mut local: Vec<Trs> = self.nodes.iter().map(|n| n.rest).collect();
        for r in &self.responses {
            let v = r.input.value(inputs);
            let (a, b) = (self.nodes[r.min].rest, self.nodes[r.max].rest);
            local[r.value] = Trs {
                t: a.t.lerp(b.t, v),
                r: a.r.slerp(b.r, v),
                s: a.s.lerp(b.s, v),
            };
        }
        let mut out = Vec::new();
        let mut stack: Vec<(usize, Affine3A, u32)> =
            self.roots.iter().map(|&r| (r, Affine3A::IDENTITY, 0)).collect();
        while let Some((i, parent, depth)) = stack.pop() {
            if depth > 64 {
                continue; // cycle guard
            }
            let world = parent * local[i].affine();
            if let Some(m) = self.nodes[i].mesh {
                out.push(DrawItem {
                    mesh: m,
                    transform: world,
                });
            }
            for &c in &self.nodes[i].children {
                stack.push((c, world, depth + 1));
            }
        }
        out
    }

    /// Bounding box (min, max) of the posed model in grip space.
    #[cfg_attr(not(test), allow(dead_code))] // tests and debugging
    pub fn bounds(&self, inputs: &ControllerInputs) -> (Vec3A, Vec3A) {
        let mut lo = Vec3A::splat(f32::INFINITY);
        let mut hi = Vec3A::splat(f32::NEG_INFINITY);
        for d in self.pose(inputs) {
            for p in &self.meshes[d.mesh].positions {
                let w = d.transform.transform_point3a(Vec3A::from(*p));
                lo = lo.min(w);
                hi = hi.max(w);
            }
        }
        (lo, hi)
    }
}

/// Column-major projection for one eye from an OpenXR field of view (tangent angles), Vulkan clip space (y down,
/// depth 0..1), like the OpenXR SDK's `XrMatrix4x4f_CreateProjectionFov` for Vulkan.
pub fn projection_fov(left: f32, right: f32, up: f32, down: f32, near: f32, far: f32) -> glam::Mat4 {
    let (tl, tr, tu, td) = (left.tan(), right.tan(), up.tan(), down.tan());
    let w = tr - tl;
    let h = td - tu;
    glam::Mat4::from_cols_array(&[
        2.0 / w,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0 / h,
        0.0,
        0.0,
        (tr + tl) / w,
        (tu + td) / h,
        -far / (far - near),
        -1.0,
        0.0,
        0.0,
        -(far * near) / (far - near),
        0.0,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn touch_aim_to_grip_matches_the_driver() {
        let t = touch_aim_to_grip();
        let fwd = t.matrix3.mul_vec3a(Vec3A::NEG_Z);
        let want = Vec3A::new(0.0, 60f32.to_radians().sin(), -60f32.to_radians().cos());
        assert!(fwd.distance(want) < 1e-5, "{fwd}");
        assert!(t.translation.distance(Vec3A::new(0.0, -0.030, 0.095)) < 5e-4, "{}", t.translation);
    }

    #[test]
    fn ray_goes_along_minus_z_from_the_pose() {
        let p = Affine3A::from_rotation_translation(
            Quat::from_rotation_y(0.5) * Quat::from_rotation_x(-0.3),
            Vec3::new(0.2, 1.1, -0.3),
        );
        let (o, d) = ray(&p);
        assert!(o.distance(Vec3A::new(0.2, 1.1, -0.3)) < 1e-6);
        assert!((d.length() - 1.0).abs() < 1e-5);
        // a point 1 m along the ray is 1 m along local -Z
        assert!((o + d).distance(p.transform_point3a(Vec3A::NEG_Z)) < 1e-5);
        let ax = axes(&p, 0.1);
        assert!(ax[2].1.distance(o + d * 0.1) < 1e-5);
        assert!((ax[0].1 - o).dot(d).abs() < 1e-5);
    }

    #[test]
    fn both_models_load_with_their_moving_parts() {
        for left in [true, false] {
            let m = ControllerModel::touch_v3(left).unwrap();
            assert_eq!(m.meshes.len(), 6, "left={left}");
            // trigger, squeeze, stick (click, x, y), 2 face buttons, thumbrest (+ menu on the left)
            assert_eq!(m.response_count(), 8, "left={left}");
            assert!(m.base_color_png.as_ref().is_some_and(|p| p.starts_with(b"\x89PNG")));
            assert_eq!(m.pose(&ControllerInputs::default()).len(), m.meshes.len());
        }
    }

    #[test]
    fn the_model_is_a_hand_sized_controller_around_the_grip() {
        for left in [true, false] {
            let m = ControllerModel::touch_v3(left).unwrap();
            let (lo, hi) = m.bounds(&ControllerInputs::default());
            let size = hi - lo;
            // a Quest 2 Touch is about 10-15 cm in its longest extent
            assert!(size.max_element() > 0.08 && size.max_element() < 0.2, "{size}");
            // the grip origin is inside the handle
            assert!(lo.cmplt(Vec3A::splat(0.01)).all() && hi.cmpgt(Vec3A::splat(-0.01)).all(), "{lo} {hi}");
        }
    }

    /// SteamVR's components in the Touch raw frame (right hand; x mirrored on the left).
    fn raw_from(deg: f32, x: f32, y: f32, z: f32) -> Affine3A {
        Affine3A::from_rotation_translation(Quat::from_rotation_x(deg.to_radians()), Vec3::new(x, y, z))
    }

    #[test]
    fn the_model_sits_where_steamvr_draws_its_own_and_the_ray_leaves_its_tip() {
        let m = ControllerModel::touch_v3(false).unwrap();
        // the right model, drawn at grip * model_in_grip, seen in the Touch raw frame (grip = raw * openxr_grip)
        let raw_from_grip = raw_from(20.6, -0.007, -0.001_829_41, 0.101_948_2);
        let mut lo = Vec3A::splat(f32::INFINITY);
        let mut hi = Vec3A::splat(f32::NEG_INFINITY);
        for it in m.pose(&ControllerInputs::default()) {
            for p in &m.meshes[it.mesh].positions {
                let w = (raw_from_grip * model_in_grip() * it.transform).transform_point3a(Vec3A::from(*p));
                lo = lo.min(w);
                hi = hi.max(w);
            }
        }
        // SteamVR's oculus_quest2_controller_right body box in the raw frame (measured once from its OBJ)
        let (s_lo, s_hi) = (
            Vec3A::new(-0.0601, -0.0376, 0.0195),
            Vec3A::new(0.0319, 0.0671, 0.1500),
        );
        assert!(lo.distance(s_lo) < 0.006 && hi.distance(s_hi) < 0.012, "{lo} {hi}");
        // the aim ray (driver aim = SteamVR's openxr_aim) starts at the front-bottom tip, at most 1.5 cm outside
        let aim = raw_from(-39.4, -0.007, -0.038_947_66, 0.009_496_94);
        let (o, d) = ray(&aim);
        let slack = Vec3A::splat(0.015);
        assert!(o.cmpge(lo - slack).all() && o.cmple(hi + slack).all(), "{o} not in {lo}..{hi}");
        assert!(o.z < lo.z + 0.1 * (hi.z - lo.z) && o.y < lo.y + 0.1 * (hi.y - lo.y), "{o}");
        // forward and 39.4 deg down from the raw frame's -Z, no yaw
        assert!(d.z < -0.7 && d.y < -0.6 && d.x.abs() < 1e-6, "{d}");
        // and the driver relation holds: grip = aim * touch_aim_to_grip
        let grip = aim * touch_aim_to_grip();
        assert!(grip.abs_diff_eq(raw_from_grip, 1e-5), "{grip:?}");
        // in grip space the ray tilts 60 deg down from the handle axis
        let (_, dg) = ray(&touch_aim_to_grip().inverse());
        assert!((dg.angle_between(Vec3A::NEG_Z).to_degrees() - 60.0).abs() < 0.01);
    }

    #[test]
    fn trigger_and_stick_move_their_parts() {
        let m = ControllerModel::touch_v3(false).unwrap();
        let rest = m.pose(&ControllerInputs::default());
        let pulled = m.pose(&ControllerInputs {
            trigger: 1.0,
            stick: [1.0, 0.0],
            ..Default::default()
        });
        let moved = rest
            .iter()
            .zip(&pulled)
            .filter(|(a, b)| !a.transform.abs_diff_eq(b.transform, 1e-5))
            .count();
        assert_eq!(moved, 2, "trigger and thumbstick");
        // nonsense input is clamped, not propagated
        let nan = m.pose(&ControllerInputs {
            trigger: f32::NAN,
            squeeze: 7.0,
            ..Default::default()
        });
        assert!(nan.iter().all(|d| d.transform.is_finite()));
    }

    #[test]
    fn projection_maps_the_fov_edges_to_the_clip_edges() {
        let (l, r, u, d) = (-0.8f32, 0.7f32, 0.75f32, -0.85f32);
        let p = projection_fov(l, r, u, d, 0.05, 50.0);
        let clip = |v: Vec3| {
            let c = p * v.extend(1.0);
            c.truncate() / c.w
        };
        let z = -2.0;
        let right = clip(Vec3::new(r.tan() * -z, 0.0, z));
        assert!((right.x - 1.0).abs() < 1e-4, "{right}");
        let left = clip(Vec3::new(l.tan() * -z, 0.0, z));
        assert!((left.x + 1.0).abs() < 1e-4);
        // Vulkan: up is -1
        let up = clip(Vec3::new(0.0, u.tan() * -z, z));
        assert!((up.y + 1.0).abs() < 1e-4, "{up}");
        let near = clip(Vec3::new(0.0, 0.0, -0.05));
        let far = clip(Vec3::new(0.0, 0.0, -50.0));
        assert!(near.z.abs() < 1e-4 && (far.z - 1.0).abs() < 1e-4, "{near} {far}");
    }
}
