use super::*;

// ---------------------------------------------------------------------------------------------------------------
// CPU tests (always run)

#[test]
fn spec_names() {
    assert_eq!(parse_spec("ocean"), Some(Quality::Medium));
    assert_eq!(parse_spec("ocean:low"), Some(Quality::Low));
    assert_eq!(parse_spec(" Ocean:HIGH "), Some(Quality::High));
    assert_eq!(parse_spec("ocean:medium"), Some(Quality::Medium));
    assert_eq!(parse_spec("ocean:ultra"), None);
    assert_eq!(parse_spec("oceanic.dds"), None, "a file that merely starts with the word is a picture");
    assert_eq!(parse_spec("/home/x/ocean/umhlanga_sunrise_8k.dds"), None);
    assert_eq!(parse_spec(""), None);
    for q in Quality::ALL {
        assert_eq!(parse_spec(&format!("ocean:{}", q.name())), Some(q));
    }
}

#[test]
fn quality_ladder_is_monotonic_and_fits_the_shader() {
    let l = Quality::ALL;
    for w in l.windows(2) {
        assert!(w[0].extent()[0] < w[1].extent()[0]);
        assert!(w[0].waves() < w[1].waves());
        assert!(w[0].cloud_octaves() <= w[1].cloud_octaves());
    }
    for q in l {
        let e = q.extent();
        assert_eq!(e[0], 2 * e[1], "equirect is 2:1");
        assert!(q.waves() <= MAX_WAVES);
        assert!(q.frame_divider() >= 1);
    }
}

#[test]
fn waves_are_deterministic_and_physical() {
    let p = OceanParams::default();
    let a = generate_waves(&p, 40);
    let b = generate_waves(&p, 40);
    assert_eq!(a, b, "same seed, same sea");
    let c = generate_waves(&OceanParams { seed: 5, ..p.clone() }, 40);
    assert_ne!(a, c);
    assert_eq!(a.len(), 40);
    for w in &a {
        assert!(w.amp.is_finite() && w.amp > 0.0);
        assert!(w.omega.is_finite() && w.omega > 0.0);
        assert!((0.09..=200.0).contains(&w.wavelength()), "wavelength {}", w.wavelength());
        // no wave steeper than the breaking limit
        assert!(w.amp * w.k_len() <= 0.30 * 1.01, "slope {}", w.amp * w.k_len());
    }
    // deep water: a 100 m wave travels at sqrt(g/k) and the capillary term is negligible there
    let k = std::f64::consts::TAU / 100.0;
    assert!((dispersion(k) - (9.81 * k).sqrt()).abs() / dispersion(k) < 1e-3);
    // short ripples are faster than pure gravity waves (surface tension)
    let k = std::f64::consts::TAU / 0.02;
    assert!(dispersion(k) > (9.81 * k).sqrt() * 1.25);
}

#[test]
fn sea_statistics_match_the_literature() {
    // Cox-Munk 1954: total mean-square slope ~ 0.003 + 0.00512 U10 (about 0.04 at 7 m/s, i.e. rms 0.2)
    let p = OceanParams::default();
    let w = generate_waves(&p, 64);
    let rms = rms_slope(&w);
    let cm = (0.003 + 0.00512 * p.wind_speed).sqrt();
    assert!(rms > 0.45 * cm && rms < 1.6 * cm, "rms slope {rms:.3} vs Cox-Munk {cm:.3}");
    // significant wave height of a fully developed sea: 0.21 U^2 / g (about 1.0 m at 7 m/s); the table starts at
    // half the peak wavenumber, so it carries a bit less
    let hs = significant_height(&w);
    let pm = 0.21 * p.wind_speed * p.wind_speed / 9.81;
    assert!(hs > 0.4 * pm && hs < 2.0 * pm, "Hs {hs:.2} vs Pierson-Moskowitz {pm:.2}");
    // a stronger wind makes a bigger sea
    let storm = generate_waves(&OceanParams { wind_speed: 14.0, ..p }, 64);
    assert!(significant_height(&storm) > 2.0 * hs);
    // fewer waves keep the same energy (the quality switch must not change the weather)
    let few = generate_waves(&OceanParams::default(), 20);
    let r20 = rms_slope(&few);
    assert!(r20 > 0.5 * rms && r20 < 1.6 * rms, "20 waves {r20:.3} vs 64 waves {rms:.3}");
}

#[test]
fn uniform_block_matches_std140_layout() {
    assert_eq!(std::mem::size_of::<Uniforms>(), (5 + MAX_WAVES) * 16);
    let p = OceanParams::default();
    let w = generate_waves(&p, 40);
    let u = make_uniforms(&w, &p, Quality::Medium, &FrameInput { time: 12.5, head: [1.0, 1.2, -0.5] });
    assert_eq!(u.a[3], 40.0);
    assert!((u.a[1] - (1.2 + p.water_below_floor)).abs() < 1e-5, "eye height above the water");
    assert!((u.a[2] - (4.0f32).to_radians().tan()).abs() < 1e-6);
    assert_eq!(u.b[0], 4096.0);
    assert_eq!(u.c[0], 1.0);
    assert_eq!(u.c[1], -0.5);
    let s = u.sun;
    let len = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
    assert!((len - 1.0).abs() < 1e-5, "unit sun direction");
    assert!(s[1] > 0.0, "sun above the horizon");
    for (i, row) in u.waves.iter().enumerate() {
        if i < 40 {
            assert!(row[2] > 0.0);
            assert!((0.0..std::f32::consts::TAU + 1e-3).contains(&row[3]), "phase wrapped");
        } else {
            assert_eq!(*row, [0.0; 4], "unused rows stay empty");
        }
    }
}

#[test]
fn waves_move_with_time_at_the_dispersion_speed() {
    let p = OceanParams::default();
    let w = generate_waves(&p, 40);
    let f0 = FrameInput { time: 0.0, head: [0.0; 3] };
    let f1 = FrameInput { time: 1.0, head: [0.0; 3] };
    let u0 = make_uniforms(&w, &p, Quality::Medium, &f0);
    let u1 = make_uniforms(&w, &p, Quality::Medium, &f1);
    for i in 0..40 {
        let dphi = (u0.waves[i][3] - u1.waves[i][3]).rem_euclid(std::f32::consts::TAU);
        let expect = (w[i].omega as f32).rem_euclid(std::f32::consts::TAU);
        let err = (dphi - expect).abs().min((dphi - expect + std::f32::consts::TAU).abs()).min((dphi - expect - std::f32::consts::TAU).abs());
        assert!(err < 1e-3, "wave {i}: phase advanced {dphi}, omega {expect}");
    }
    // a long time does not lose phase accuracy (the table is folded in f64)
    let far = make_uniforms(&w, &p, Quality::Medium, &FrameInput { time: 3.0e6, head: [0.0; 3] });
    assert!(far.waves[5][3].is_finite() && far.a[0] < 100_000.0);
}

#[test]
fn eye_height_clamps_to_a_person() {
    let p = OceanParams::default();
    assert!((eye_height(&p, 1.2) - 4.2).abs() < 1e-6);
    assert_eq!(eye_height(&p, -5.0), 0.3 + p.water_below_floor);
    assert_eq!(eye_height(&p, 50.0), 2.4 + p.water_below_floor);
}

#[test]
fn direction_sampling_is_a_proper_inverse_cdf() {
    use std::f64::consts::PI;
    assert!(cos2_inverse_cdf(0.5).abs() < 1e-9);
    assert!((cos2_inverse_cdf(0.02) + PI / 2.0).abs() < 0.5);
    let mut last = -10.0;
    for i in 1..100 {
        let a = cos2_inverse_cdf(i as f64 / 100.0);
        assert!(a > last);
        last = a;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// GPU test: renders offscreen, writes PNGs and measures the cost. It skips itself (with a message) when a VR session
// is running (the GPU is the headset's), when there is no Vulkan device, or when OMARCHYVR_OCEAN_NOGPU is set.
// Output folder: $OMARCHYVR_OCEAN_OUT (default: the temp dir). Frames to time: $OMARCHYVR_OCEAN_FRAMES (default 60).

mod gpu {
    use super::*;
    use std::time::Instant;
    use vulkano::{
        VulkanLibrary,
        buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
        command_buffer::{CopyImageToBufferInfo, PrimaryCommandBufferAbstract},
        device::{Device, DeviceCreateInfo, DeviceFeatures, QueueCreateInfo, QueueFlags, physical::PhysicalDeviceType},
        format::Format,
        image::ImageUsage,
        instance::{Instance, InstanceCreateInfo},
        query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType},
        sync::{GpuFuture, PipelineStage},
    };
    use wgui::gfx::WGfx;

    fn vr_running() -> bool {
        // a process named monado-service, vrserver or wayvr (its comm is cut to 15 chars) owns the GPU
        let Ok(rd) = std::fs::read_dir("/proc") else { return false };
        rd.flatten().any(|e| {
            let name = e.file_name();
            let Some(pid) = name.to_str().filter(|s| s.bytes().all(|b| b.is_ascii_digit())) else { return false };
            let comm = std::fs::read_to_string(format!("/proc/{pid}/comm")).unwrap_or_default();
            matches!(comm.trim(), "monado-service" | "vrserver" | "vrcompositor")
        })
    }

    fn headless_gfx() -> Option<Arc<WGfx>> {
        let lib = VulkanLibrary::new().ok()?;
        let instance = Instance::new(lib, InstanceCreateInfo { max_api_version: Some(vulkano::Version::V1_3), ..Default::default() }).ok()?;
        let mut devs: Vec<_> = instance.enumerate_physical_devices().ok()?.collect();
        devs.sort_by_key(|d| match d.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            _ => 2,
        });
        let phys = devs.into_iter().find(|d| d.supported_features().dynamic_rendering)?;
        let qf = phys.queue_family_properties().iter().position(|q| q.queue_flags.contains(QueueFlags::GRAPHICS))? as u32;
        let (device, mut queues) = Device::new(
            phys.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo { queue_family_index: qf, ..Default::default() }],
                enabled_features: DeviceFeatures { dynamic_rendering: true, ..DeviceFeatures::empty() },
                ..Default::default()
            },
        )
        .ok()?;
        let q = queues.next()?;
        eprintln!("ocean gpu test: {}", phys.properties().device_name);
        Some(WGfx::new_from_raw(instance, device, q.clone(), q, Format::R8G8B8A8_SRGB))
    }

    fn out_dir() -> std::path::PathBuf {
        let d = std::env::var_os("OMARCHYVR_OCEAN_OUT").map(std::path::PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("omarchyvr-ocean"));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    struct Rendered {
        rgba: Vec<u8>,
        w: u32,
        h: u32,
        gpu_ms: Vec<f64>,
        wall_ms: Vec<f64>,
    }

    /// Render `frames` pictures at `time0 + i/90 s`, return the last one and the per-frame GPU times.
    fn render(gfx: &Arc<WGfx>, q: Quality, params: OceanParams, time0: f64, frames: usize, head: [f32; 3]) -> Rendered {
        let r = OceanRenderer::new(gfx, q, params).expect("renderer");
        let [w, h] = q.extent();
        let image = gfx.new_image(w, h, Format::R8G8B8A8_SRGB, ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC).unwrap();
        let view = ImageView::new_default(image.clone()).unwrap();
        let readback: Subbuffer<[u8]> = Buffer::new_slice(
            gfx.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::TRANSFER_DST, ..Default::default() },
            AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS, ..Default::default() },
            (w * h * 4) as u64,
        )
        .unwrap();
        let pool = QueryPool::new(gfx.device.clone(), QueryPoolCreateInfo { query_count: 2, ..QueryPoolCreateInfo::query_type(QueryType::Timestamp) }).unwrap();
        let period = f64::from(gfx.device.physical_device().properties().timestamp_period) / 1.0e6; // ticks -> ms

        let mut gpu_ms = Vec::new();
        let mut wall_ms = Vec::new();
        for i in 0..frames.max(1) {
            let last = i + 1 == frames.max(1);
            let mut cmd = gfx.create_gfx_command_buffer(vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit).unwrap();
            unsafe {
                cmd.command_buffer.reset_query_pool(pool.clone(), 0..2).unwrap();
                cmd.command_buffer.write_timestamp(pool.clone(), 0, PipelineStage::AllCommands).unwrap();
            }
            r.draw(&mut cmd, view.clone(), &FrameInput { time: time0 + i as f64 / 90.0, head }).unwrap();
            unsafe {
                cmd.command_buffer.write_timestamp(pool.clone(), 1, PipelineStage::AllCommands).unwrap();
            }
            if last {
                cmd.command_buffer.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(image.clone(), readback.clone())).unwrap();
            }
            let queue = cmd.queue.clone();
            let cb = cmd.build().unwrap();
            let t = Instant::now();
            let f = cb.execute(queue).unwrap().then_signal_fence_and_flush().unwrap();
            f.wait(None).unwrap();
            wall_ms.push(t.elapsed().as_secs_f64() * 1e3);
            let mut res = [0u64; 2];
            pool.get_results(0..2, &mut res, QueryResultFlags::WAIT).unwrap();
            gpu_ms.push((res[1] - res[0]) as f64 * period);
        }
        let rgba = readback.read().unwrap().to_vec();
        Rendered { rgba, w, h, gpu_ms, wall_ms }
    }

    fn median(v: &[f64]) -> f64 {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    }

    /// A rectilinear view out of the equirect picture (bilinear), for looking at it.
    fn view(img: &Rendered, yaw_deg: f64, pitch_deg: f64, fov_deg: f64, out_w: u32, out_h: u32) -> Vec<u8> {
        let (yaw, pitch) = (yaw_deg.to_radians(), pitch_deg.to_radians());
        let f = 0.5 * out_w as f64 / (0.5 * fov_deg.to_radians()).tan();
        let mut out = vec![0u8; (out_w * out_h * 4) as usize];
        for y in 0..out_h {
            for x in 0..out_w {
                let (px, py) = (x as f64 + 0.5 - 0.5 * out_w as f64, 0.5 * out_h as f64 - y as f64 - 0.5);
                // camera looks along -Z; rotate by pitch (x axis) then yaw (y axis, clockwise seen from above)
                let (cx, cy, cz) = (px, py, -f);
                let (cy2, cz2) = (cy * pitch.cos() - cz * pitch.sin(), cy * pitch.sin() + cz * pitch.cos());
                let (dx, dz) = (cx * yaw.cos() - cz2 * yaw.sin(), cx * yaw.sin() + cz2 * yaw.cos());
                let dy = cy2;
                let n = (dx * dx + dy * dy + dz * dz).sqrt();
                let (dx, dy, dz) = (dx / n, dy / n, dz / n);
                let az = dx.atan2(-dz);
                let el = dy.asin();
                let u = (az / std::f64::consts::TAU + 0.5) * img.w as f64 - 0.5;
                let v = (0.5 - el / std::f64::consts::PI) * img.h as f64 - 0.5;
                let (u0, v0) = (u.floor(), v.floor());
                let (fu, fv) = (u - u0, v - v0);
                let at = |uu: f64, vv: f64, c: usize| {
                    let ux = (uu.rem_euclid(img.w as f64)) as usize;
                    let vy = (vv.clamp(0.0, img.h as f64 - 1.0)) as usize;
                    f64::from(img.rgba[(vy * img.w as usize + ux) * 4 + c])
                };
                for c in 0..3 {
                    let a = at(u0, v0, c) * (1.0 - fu) + at(u0 + 1.0, v0, c) * fu;
                    let b = at(u0, v0 + 1.0, c) * (1.0 - fu) + at(u0 + 1.0, v0 + 1.0, c) * fu;
                    out[((y * out_w + x) * 4) as usize + c] = (a * (1.0 - fv) + b * fv).round() as u8;
                }
                out[((y * out_w + x) * 4) as usize + 3] = 255;
            }
        }
        out
    }

    fn save(path: &std::path::Path, rgba: &[u8], w: u32, h: u32) {
        image::save_buffer(path, rgba, w, h, image::ColorType::Rgba8).unwrap();
        eprintln!("wrote {}", path.display());
    }

    fn mean_luma(img: &Rendered, y0: u32, y1: u32) -> f64 {
        let mut s = 0.0;
        let mut n = 0.0;
        for y in y0..y1 {
            for x in 0..img.w {
                let i = ((y * img.w + x) * 4) as usize;
                s += 0.2126 * f64::from(img.rgba[i]) + 0.7152 * f64::from(img.rgba[i + 1]) + 0.0722 * f64::from(img.rgba[i + 2]);
                n += 1.0;
            }
        }
        s / n
    }

    #[test]
    fn offscreen_render_png_and_gpu_time() {
        if std::env::var_os("OMARCHYVR_OCEAN_NOGPU").is_some() {
            eprintln!("skipped: OMARCHYVR_OCEAN_NOGPU set");
            return;
        }
        if vr_running() {
            eprintln!("skipped: a VR session is running (the GPU is the headset's)");
            return;
        }
        let Some(gfx) = headless_gfx() else {
            eprintln!("skipped: no Vulkan device with dynamic rendering");
            return;
        };
        let frames: usize = std::env::var("OMARCHYVR_OCEAN_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
        let dir = out_dir();
        let mut report = String::from("quality,width,height,waves,gpu_ms_median,gpu_ms_p95,wall_ms_median\n");
        for q in Quality::ALL {
            if vr_running() {
                panic!("a VR session started during the test: stopping");
            }
            let _warm = render(&gfx, q, OceanParams::default(), 0.0, 3, [0.0, 1.2, 0.0]);
            let img = render(&gfx, q, OceanParams::default(), 20.0, frames, [0.0, 1.2, 0.0]);
            let mut sorted = img.gpu_ms.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p95 = sorted[(sorted.len() * 95 / 100).min(sorted.len() - 1)];
            let med = median(&img.gpu_ms);
            eprintln!("ocean {:<6} {}x{} {} waves: gpu median {:.3} ms, p95 {:.3} ms, wall median {:.3} ms", q.name(), img.w, img.h, q.waves(), med, p95, median(&img.wall_ms));
            report += &format!("{},{},{},{},{:.4},{:.4},{:.4}\n", q.name(), img.w, img.h, q.waves(), med, p95, median(&img.wall_ms));

            // sanity of the picture: not blank, sky brighter than... and a horizon where the dip puts it
            let top = mean_luma(&img, 0, img.h / 4);
            let bottom = mean_luma(&img, img.h * 3 / 4, img.h);
            assert!(top > 5.0 && bottom > 2.0, "{}: picture is black (top {top:.1}, bottom {bottom:.1})", q.name());
            assert!(img.rgba.iter().skip(3).step_by(4).all(|a| *a == 255), "opaque");
            let distinct: std::collections::HashSet<[u8; 3]> = img.rgba.chunks(4).step_by(97).map(|p| [p[0], p[1], p[2]]).collect();
            assert!(distinct.len() > 500, "{}: only {} distinct colours", q.name(), distinct.len());

            save(&dir.join(format!("ocean-{}-equirect.png", q.name())), &img.rgba, img.w, img.h);
            if q == Quality::Medium || q == Quality::High {
                let sun = OceanParams::default().sun_azimuth_deg;
                let toward = view(&img, sun, 2.0, 100.0, 1600, 900);
                save(&dir.join(format!("ocean-{}-view-sun.png", q.name())), &toward, 1600, 900);
                let away = view(&img, sun + 160.0, -3.0, 100.0, 1600, 900);
                save(&dir.join(format!("ocean-{}-view-away.png", q.name())), &away, 1600, 900);
                let down = view(&img, 0.0, -45.0, 100.0, 1600, 900);
                save(&dir.join(format!("ocean-{}-view-down.png", q.name())), &down, 1600, 900);
            }
        }
        // the animation moves: two pictures a second apart differ
        let a = render(&gfx, Quality::Low, OceanParams::default(), 5.0, 1, [0.0, 1.2, 0.0]);
        let b = render(&gfx, Quality::Low, OceanParams::default(), 6.0, 1, [0.0, 1.2, 0.0]);
        let diff = a.rgba.iter().zip(&b.rgba).filter(|(x, y)| x != y).count();
        assert!(diff > a.rgba.len() / 20, "the sea does not move: {diff} bytes differ");
        // and the same time gives the same picture
        let c = render(&gfx, Quality::Low, OceanParams::default(), 5.0, 1, [0.0, 1.2, 0.0]);
        assert_eq!(a.rgba, c.rgba, "deterministic");
        std::fs::write(dir.join("ocean-gpu-time.csv"), report).unwrap();
    }
}
