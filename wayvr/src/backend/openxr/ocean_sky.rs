//! The animated ocean as the skybox of Work mode: an equirect swapchain that is redrawn every n-th frame by
//! `crate::ocean`. Used by `skybox.rs` when the skybox name is `ocean:low|medium|high`.

use std::time::Instant;

use vulkano::command_buffer::CommandBufferUsage;

use crate::{
    graphics::GpuFutures,
    ocean::{FrameInput, OceanParams, OceanRenderer, Quality},
    state::AppState,
};

use super::{
    XrState,
    swapchain::{SwapchainOpts, WlxSwapchain, create_swapchain},
};

pub(super) struct OceanSky {
    pub(super) swapchain: WlxSwapchain,
    renderer: OceanRenderer,
    started: Instant,
    frame: u64,
    divider: u64,
}

impl OceanSky {
    pub(super) fn new(xr: &XrState, app: &AppState, quality: Quality) -> anyhow::Result<Self> {
        let swapchain = create_swapchain(xr, app.gfx.clone(), quality.extent(), 1, SwapchainOpts::new())?;
        let renderer = OceanRenderer::new(&app.gfx, quality, OceanParams::default())?;
        log::info!("ocean background: {} ({}x{}, {} waves, every {} frames)", quality.name(), quality.extent()[0], quality.extent()[1], quality.waves(), quality.frame_divider());
        Ok(Self { swapchain, renderer, started: Instant::now(), frame: 0, divider: u64::from(quality.frame_divider()) })
    }

    /// Draw this frame's picture (every `divider`-th frame; in between the layer keeps the last released image).
    pub(super) fn render(&mut self, app: &AppState, futures: &mut GpuFutures) -> anyhow::Result<()> {
        let due = self.frame % self.divider == 0 || !self.swapchain.ever_acquired;
        self.frame += 1;
        if !due {
            return Ok(());
        }
        let target = self.swapchain.acquire_wait_image()?.views.into_iter().next().unwrap(); // want panic
        let head = app.input_state.hmd.translation;
        let frame = FrameInput { time: self.started.elapsed().as_secs_f64(), head: [head.x, head.y, head.z] };
        let mut cmd = app.gfx.create_gfx_command_buffer(CommandBufferUsage::OneTimeSubmit)?;
        self.renderer.draw(&mut cmd, target, &frame)?;
        futures.execute(cmd.queue.clone(), cmd.build()?)?;
        Ok(())
    }
}
