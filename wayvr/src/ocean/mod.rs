//! Animated ocean background for Work mode (omarchyvr).
//!
//! The ocean is drawn into an equirectangular image with one fragment shader (`shaders/ocean.frag`); the OpenXR
//! backend shows that image on the same equirect layer the skybox pictures use (`backend/openxr/ocean_sky.rs`).
//! This module holds everything that does not need OpenXR: the quality presets, the wave table (a JONSWAP-like
//! sea sampled as a sum of directional waves), the uniform block and the renderer, so it can be tested and timed
//! headless.
//!
//! Selected with the skybox name `ocean:low|medium|high` (`wayvrctl skybox ocean:medium`); the Pimax app writes
//! that name from its Backgrounds page.
//!
//! References (clean-room, no code copied from other renderers):
//! * J. Tessendorf, "Simulating Ocean Water", SIGGRAPH course notes 2001 (statistical sea, Gerstner/Jacobian foam).
//! * W. Pierson, L. Moskowitz 1964 (fully developed sea); K. Hasselmann et al. 1973 (JONSWAP).
//! * C. Cox, W. Munk 1954 (slope statistics of the sea surface, used to check the roughness).
//! * T. Nishita et al. 1993 (single scattering sky); B. Walter et al. 2007 (GGX microfacet specular).

use std::sync::Arc;

use anyhow::Context;
use vulkano::{
    buffer::{
        Buffer, BufferContents, BufferCreateInfo, BufferUsage,
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
    },
    image::view::ImageView,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};
use wgui::gfx::{
    WGfx,
    cmd::{GfxCommandBuffer, WGfxClearMode},
    pipeline::{WGfxPipeline, WPipelineCreateInfo},
};

use crate::graphics::Vert2Uv;

#[cfg(test)]
mod tests;

pub mod frag_ocean {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/ocean.frag",
    }
}

/// Prefix of the skybox name that selects the ocean.
pub const SPEC_PREFIX: &str = "ocean";
/// The shader's wave table size.
pub const MAX_WAVES: usize = 64;
const GRAVITY: f64 = 9.81;
/// Surface tension cut-off wavenumber of water (rad/m): sqrt(rho g / sigma).
const K_CAPILLARY: f64 = 363.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Quality {
    Low,
    #[default]
    Medium,
    High,
}

impl Quality {
    pub const ALL: [Quality; 3] = [Quality::Low, Quality::Medium, Quality::High];

    pub fn name(self) -> &'static str {
        match self {
            Quality::Low => "low",
            Quality::Medium => "medium",
            Quality::High => "high",
        }
    }

    /// Size of the equirect image (2:1).
    pub fn extent(self) -> [u32; 2] {
        match self {
            Quality::Low => [2048, 1024],
            Quality::Medium => [4096, 2048],
            Quality::High => [6144, 3072],
        }
    }

    pub fn waves(self) -> usize {
        match self {
            Quality::Low => 20,
            Quality::Medium => 40,
            Quality::High => 64,
        }
    }

    /// Octaves of the cloud noise (0 = clear sky).
    pub fn cloud_octaves(self) -> u32 {
        match self {
            Quality::Low => 2,
            Quality::Medium => 4,
            Quality::High => 5,
        }
    }

    /// Draw a new picture every n-th frame (the layer keeps showing the last one in between).
    pub fn frame_divider(self) -> u32 {
        match self {
            Quality::Low => 3,
            Quality::Medium => 2,
            Quality::High => 2,
        }
    }
}

/// `ocean`, `ocean:low`, `ocean:medium`, `ocean:high` -> the quality. Anything else is not the ocean.
pub fn parse_spec(spec: &str) -> Option<Quality> {
    let s = spec.trim().to_ascii_lowercase();
    let rest = s.strip_prefix(SPEC_PREFIX)?;
    match rest.strip_prefix(':').unwrap_or(rest) {
        "" | "medium" | "med" | "mid" => Some(Quality::Medium),
        "low" => Some(Quality::Low),
        "high" => Some(Quality::High),
        _ => None,
    }
}

/// The scene: weather, light and the water's height under the user.
#[derive(Clone, Debug)]
pub struct OceanParams {
    /// m/s at 10 m. 7 is a lively but pleasant sea (about 1 m waves).
    pub wind_speed: f64,
    /// Degrees clockwise from straight ahead (-Z of the stage) that the wind blows towards.
    pub wind_dir_deg: f64,
    pub sun_azimuth_deg: f64,
    pub sun_elevation_deg: f64,
    pub sun_strength: f32,
    /// How far the water plane lies below the floor (m); the head is that much plus its own height above it.
    pub water_below_floor: f32,
    /// The horizon is shown this many degrees below eye level (a pier above a curved sea), so it clears the screens.
    pub horizon_dip_deg: f32,
    pub choppiness: f32,
    pub foam: f32,
    pub cloud_coverage: f32,
    pub exposure: f32,
    pub seed: u64,
}

impl Default for OceanParams {
    fn default() -> Self {
        OceanParams {
            wind_speed: 7.0,
            wind_dir_deg: 20.0,
            sun_azimuth_deg: 50.0,
            sun_elevation_deg: 11.0,
            sun_strength: 1.0,
            water_below_floor: 3.0,
            horizon_dip_deg: 4.0,
            choppiness: 0.75,
            foam: 1.0,
            cloud_coverage: 0.55,
            exposure: 1.0,
            seed: 0x0CEA_17,
        }
    }
}

/// One plane wave of the sea.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wave {
    /// Wave vector (rad/m) on the water plane (x, z).
    pub k: [f64; 2],
    /// Amplitude (m).
    pub amp: f64,
    /// Angular frequency (rad/s) from the dispersion relation.
    pub omega: f64,
    pub phase0: f64,
}

impl Wave {
    pub fn k_len(&self) -> f64 {
        self.k[0].hypot(self.k[1])
    }
    pub fn wavelength(&self) -> f64 {
        std::f64::consts::TAU / self.k_len()
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Inverse of the cos^2 direction distribution on (-pi/2, pi/2): u in (0,1) -> angle.
fn cos2_inverse_cdf(u: f64) -> f64 {
    use std::f64::consts::PI;
    // CDF(t) = 0.5 + (t + sin t cos t) / pi
    let (mut lo, mut hi) = (-PI / 2.0, PI / 2.0);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let cdf = 0.5 + (mid + mid.sin() * mid.cos()) / PI;
        if cdf < u {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Dispersion with the capillary term: omega^2 = g k (1 + (k/k_m)^2).
pub fn dispersion(k: f64) -> f64 {
    (GRAVITY * k * (1.0 + (k / K_CAPILLARY).powi(2))).sqrt()
}

/// Wavenumber spectrum S(k) of a fetch-limited sea: Phillips/Pierson-Moskowitz k^-3 tail with the low-k cut-off
/// at the peak `kp` and the JONSWAP peak enhancement (gamma 3.3).
pub fn spectrum(k: f64, kp: f64) -> f64 {
    let alpha = 0.0081;
    let pm = 0.5 * alpha * k.powi(-3) * (-1.25 * (kp / k).powi(2)).exp();
    let sigma = if k <= kp { 0.07 } else { 0.09 };
    let r = (-((k / kp).sqrt() - 1.0).powi(2) / (2.0 * sigma * sigma)).exp();
    pm * 3.3f64.powf(r)
}

/// Peak wavenumber of a fully developed sea at wind speed `u` (Pierson-Moskowitz: omega_p = 0.877 g / U).
pub fn peak_wavenumber(u: f64) -> f64 {
    let wp = 0.877 * GRAVITY / u;
    wp * wp / GRAVITY
}

/// The wave table: `n` waves, log-spaced in k from half the peak to 10 cm wavelength, directions drawn from a
/// cos^2 spreading that widens for the short waves, plus two long swells so the horizon has big slow shapes.
pub fn generate_waves(p: &OceanParams, n: usize) -> Vec<Wave> {
    use std::f64::consts::TAU;
    let n = n.clamp(2, MAX_WAVES);
    let mut rng = Rng(p.seed);
    let kp = peak_wavenumber(p.wind_speed.max(1.0));
    let k_min = kp * 0.5;
    let k_max = TAU / 0.10;
    let dln = (k_max / k_min).ln() / n as f64;
    let wind = p.wind_dir_deg.to_radians();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let k = k_min * ((i as f64 + 0.35 + 0.3 * rng.next()) * dln).exp();
        // energy of the band this wave stands for: S(k) dk = S k dln
        let amp = (2.0 * spectrum(k, kp) * k * dln).sqrt();
        // golden-ratio sequence for the direction so neighbours in k do not line up
        let u = (0.123 + i as f64 * 0.618_033_988_75).fract().clamp(0.02, 0.98);
        let widen = 1.0 + 0.8 * (k / (6.0 * kp)).min(1.0);
        let dir = wind + cos2_inverse_cdf(u) * widen;
        let amp = amp.min(0.14 / k); // slope limit: a wave this steep would break
        out.push(Wave { k: [dir.sin() * k, -dir.cos() * k], amp, omega: dispersion(k), phase0: rng.next() * TAU });
    }
    // two swells: long, low and nearly sinusoidal (a different direction from the wind sea)
    for (j, (lam, amp, deg)) in [(90.0, 0.30, -38.0), (62.0, 0.18, 27.0)].into_iter().enumerate() {
        if n <= 6 {
            break;
        }
        let k = TAU / lam;
        let dir = wind + f64::to_radians(deg);
        out[j] = Wave { k: [dir.sin() * k, -dir.cos() * k], amp, omega: dispersion(k), phase0: rng.next() * TAU };
    }
    out
}

/// Root-mean-square slope of the table (both axes together): the Cox-Munk number to check the roughness against.
pub fn rms_slope(waves: &[Wave]) -> f64 {
    waves.iter().map(|w| 0.5 * (w.amp * w.k_len()).powi(2)).sum::<f64>().sqrt()
}

/// Significant wave height (4 sigma of the surface height).
pub fn significant_height(waves: &[Wave]) -> f64 {
    4.0 * waves.iter().map(|w| 0.5 * w.amp * w.amp).sum::<f64>().sqrt()
}

/// The shader's uniform block (std140: everything is a vec4).
#[repr(C)]
#[derive(BufferContents, Clone, Copy, Debug, PartialEq)]
pub struct Uniforms {
    /// time (clouds), eye height, horizon dip (tan), wave count
    pub a: [f32; 4],
    /// image width, exposure, choppiness, cloud octaves
    pub b: [f32; 4],
    /// sun direction, strength
    pub sun: [f32; 4],
    /// head x, head z, foam, cloud coverage
    pub c: [f32; 4],
    /// wave amplitude scale, haze distance, cloud drift x, z
    pub d: [f32; 4],
    pub waves: [[f32; 4]; MAX_WAVES],
}

/// What changes every frame.
#[derive(Clone, Copy, Debug)]
pub struct FrameInput {
    /// Seconds since the ocean started.
    pub time: f64,
    /// Head position in the stage space (m).
    pub head: [f32; 3],
}

pub fn sun_direction(p: &OceanParams) -> [f32; 3] {
    let az = p.sun_azimuth_deg.to_radians();
    let el = p.sun_elevation_deg.to_radians();
    [(az.sin() * el.cos()) as f32, el.sin() as f32, (-az.cos() * el.cos()) as f32]
}

/// The height of the head above the water plane: the water is fixed in the stage, `water_below_floor` under
/// the floor, so standing up lifts the horizon view and sitting down lowers it.
pub fn eye_height(p: &OceanParams, head_y: f32) -> f32 {
    head_y.clamp(0.3, 2.4) + p.water_below_floor
}

pub fn make_uniforms(waves: &[Wave], p: &OceanParams, q: Quality, f: &FrameInput) -> Uniforms {
    let mut w = [[0f32; 4]; MAX_WAVES];
    for (i, wave) in waves.iter().take(MAX_WAVES).enumerate() {
        // time is folded into the phase in f64 so the shader never sees a big number
        let ph = (wave.phase0 - wave.omega * f.time).rem_euclid(std::f64::consts::TAU);
        w[i] = [wave.k[0] as f32, wave.k[1] as f32, wave.amp as f32, ph as f32];
    }
    let s = sun_direction(p);
    let drift = 0.004f32; // clouds drift slowly with the wind
    let wd = p.wind_dir_deg.to_radians();
    Uniforms {
        a: [(f.time % 100_000.0) as f32, eye_height(p, f.head[1]), p.horizon_dip_deg.to_radians().tan(), waves.len().min(MAX_WAVES) as f32],
        b: [q.extent()[0] as f32, p.exposure, p.choppiness, q.cloud_octaves() as f32],
        sun: [s[0], s[1], s[2], p.sun_strength],
        c: [f.head[0], f.head[2], p.foam, p.cloud_coverage],
        d: [1.0, 5500.0, wd.sin() as f32 * drift, -(wd.cos() as f32) * drift],
        waves: w,
    }
}

/// Draws the ocean into any colour image (R8G8B8A8_SRGB).
pub struct OceanRenderer {
    pub quality: Quality,
    pub params: OceanParams,
    waves: Vec<Wave>,
    pipeline: Arc<WGfxPipeline<Vert2Uv>>,
    quad: vulkano::buffer::Subbuffer<[Vert2Uv]>,
    ubo: SubbufferAllocator,
}

impl OceanRenderer {
    pub fn new(gfx: &Arc<WGfx>, quality: Quality, params: OceanParams) -> anyhow::Result<Self> {
        let vert = crate::shaders::vert_quad::load(gfx.device.clone()).context("quad vertex shader")?;
        let frag = frag_ocean::load(gfx.device.clone()).context("ocean fragment shader")?;
        let pipeline = gfx.create_pipeline::<Vert2Uv>(&vert, &frag, WPipelineCreateInfo::new(gfx.surface_format))?;
        let verts = [[0., 0.], [1., 0.], [0., 1.], [1., 1.]].map(|p| Vert2Uv { in_pos: p, in_uv: p });
        let quad = Buffer::from_iter(
            gfx.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::VERTEX_BUFFER, ..Default::default() },
            AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, ..Default::default() },
            verts,
        )?;
        let ubo = SubbufferAllocator::new(
            gfx.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );
        let waves = generate_waves(&params, quality.waves());
        Ok(Self { quality, params, waves, pipeline, quad, ubo })
    }

    pub fn waves(&self) -> &[Wave] {
        &self.waves
    }

    /// Record one picture into `target` (the whole image).
    pub fn draw(&self, cmd: &mut GfxCommandBuffer, target: Arc<ImageView>, frame: &FrameInput) -> anyhow::Result<()> {
        let u = make_uniforms(&self.waves, &self.params, self.quality, frame);
        let buf = self.ubo.allocate_slice::<Uniforms>(1)?;
        buf.write()?[0] = u;
        let set = self.pipeline.buffer(0, buf)?;
        let ext = target.image().extent();
        let pass = self.pipeline.create_pass([ext[0] as f32, ext[1] as f32], [0.0, 0.0], self.quad.clone(), 0..4, 0..1, vec![set], &Default::default())?;
        cmd.begin_rendering(target, WGfxClearMode::DontCare)?;
        cmd.run_ref(&pass)?;
        cmd.end_rendering()?;
        Ok(())
    }
}
