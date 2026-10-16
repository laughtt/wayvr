//! Draws the controller models (`backend/controller_model.rs`) into a stereo projection layer: one 2-layer swapchain
//! (left, right eye), a depth buffer, the views and fields of view the runtime gives for this frame. Only submitted
//! while at least one controller model is shown, so with hands only (the Work default) it costs nothing.

use std::sync::Arc;

use glam::{Affine3A, Mat4, Vec3A};
use openxr as xr;
use smallvec::smallvec;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, RenderingAttachmentInfo, RenderingInfo,
        SubpassContents,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    format::{ClearValue, Format},
    image::{
        Image, ImageCreateInfo, ImageType, ImageUsage,
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Scissor, Viewport},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
};

use crate::{
    backend::controller_model::{ControllerInputs, ControllerModel, model_in_grip, projection_fov},
    graphics::GpuFutures,
    shaders::{frag_model, vert_model},
    state::AppState,
};

use super::{
    CompositionLayer, XrState,
    helpers::posef_to_transform,
    swapchain::{SwapchainOpts, WlxSwapchain, create_swapchain},
};

/// Eye resolution relative to the runtime's recommendation: the models are small and close, sharp enough at this.
const RESOLUTION_SCALE: f32 = 0.6;
const NEAR: f32 = 0.02;
const FAR: f32 = 20.0;
const DEPTH_FORMAT: Format = Format::D32_SFLOAT;

#[repr(C)]
#[derive(BufferContents, Vertex, Copy, Clone, Debug)]
struct ModelVert {
    #[format(R32G32B32_SFLOAT)]
    in_pos: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    in_normal: [f32; 3],
    #[format(R32G32_SFLOAT)]
    in_uv: [f32; 2],
}

#[repr(C)]
#[derive(BufferContents, Copy, Clone, Debug)]
struct PushConstants {
    mvp: [[f32; 4]; 4],
    m0: [f32; 4],
    m1: [f32; 4],
    m2: [f32; 4],
    tint: [f32; 4],
}

struct GpuMesh {
    vertices: Subbuffer<[ModelVert]>,
    indices: Subbuffer<[u32]>,
    count: u32,
}

struct GpuModel {
    model: ControllerModel,
    meshes: Vec<GpuMesh>,
    texture: Arc<DescriptorSet>,
}

/// What to draw this frame for one hand.
#[derive(Debug, Clone, Copy)]
pub(super) struct ShownController {
    /// Grip pose, stage space.
    pub grip: Affine3A,
    pub inputs: ControllerInputs,
}

pub(super) struct ControllerRenderer {
    pipeline: Arc<GraphicsPipeline>,
    models: [GpuModel; 2],
    swapchain: WlxSwapchain,
    depth: Arc<ImageView>,
    extent: [u32; 2],
    /// Views used for this frame's image (pose, fov per eye); None = nothing drawn, no layer.
    drawn: Option<[xr::View; 2]>,
}

fn upload<T: BufferContents + Copy>(
    app: &AppState,
    usage: BufferUsage,
    data: Vec<T>,
) -> anyhow::Result<Subbuffer<[T]>> {
    Ok(Buffer::from_iter(
        app.gfx.memory_allocator.clone(),
        BufferCreateInfo {
            usage,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data,
    )?)
}

impl ControllerRenderer {
    pub(super) fn new(
        xr: &XrState,
        app: &AppState,
        system: xr::SystemId,
    ) -> anyhow::Result<Self> {
        let device = app.gfx.device.clone();
        let vs = vert_model::load(device.clone())?.entry_point("main").unwrap(); // want panic
        let fs = frag_model::load(device.clone())?.entry_point("main").unwrap(); // want panic
        let vertex_input_state = ModelVert::per_vertex().definition(&vs)?;
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())?,
        )?;
        let pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(Default::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    1,
                    ColorBlendAttachmentState::default(),
                )),
                dynamic_state: [DynamicState::Viewport, DynamicState::Scissor]
                    .into_iter()
                    .collect(),
                subpass: Some(
                    PipelineRenderingCreateInfo {
                        color_attachment_formats: vec![Some(app.gfx.surface_format)],
                        depth_attachment_format: Some(DEPTH_FORMAT),
                        ..Default::default()
                    }
                    .into(),
                ),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )?;

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )?;
        let set_layout = pipeline.layout().set_layouts()[0].clone();
        let load = |left: bool| -> anyhow::Result<GpuModel> {
            let model = ControllerModel::touch_v3(left)?;
            let mut meshes = Vec::with_capacity(model.meshes.len());
            for m in &model.meshes {
                let verts = (0..m.positions.len())
                    .map(|i| ModelVert {
                        in_pos: m.positions[i],
                        in_normal: m.normals[i],
                        in_uv: m.uvs[i],
                    })
                    .collect::<Vec<_>>();
                meshes.push(GpuMesh {
                    vertices: upload(app, BufferUsage::VERTEX_BUFFER, verts)?,
                    indices: upload(app, BufferUsage::INDEX_BUFFER, m.indices.clone())?,
                    count: m.indices.len() as u32,
                });
            }
            // base colour (sRGB); a mid grey 1x1 if the file has none
            let (w, h, rgba) = match model
                .base_color_png
                .as_deref()
                .map(|png| image::load_from_memory_with_format(png, image::ImageFormat::Png))
            {
                Some(Ok(img)) => {
                    let img = img.to_rgba8();
                    (img.width(), img.height(), img.into_raw())
                }
                _ => (1, 1, vec![60, 60, 64, 255]),
            };
            let mut xfer = app
                .gfx
                .create_xfer_command_buffer(CommandBufferUsage::OneTimeSubmit)?;
            let image = xfer.upload_image(w, h, Format::R8G8B8A8_SRGB, &rgba)?;
            xfer.build_and_execute_now()?;
            let texture = DescriptorSet::new(
                app.gfx.descriptor_set_allocator.clone(),
                set_layout.clone(),
                [WriteDescriptorSet::image_view_sampler(
                    0,
                    ImageView::new_default(image)?,
                    sampler.clone(),
                )],
                [],
            )?;
            Ok(GpuModel {
                model,
                meshes,
                texture,
            })
        };
        let models = [load(true)?, load(false)?];

        let rec = xr
            .instance
            .enumerate_view_configuration_views(system, super::VIEW_TYPE)?;
        let base = rec.first().map_or([1600, 1600], |v| {
            [
                v.recommended_image_rect_width,
                v.recommended_image_rect_height,
            ]
        });
        let extent = base.map(|x| ((x as f32 * RESOLUTION_SCALE) as u32).clamp(256, 4096));
        let swapchain = create_swapchain(xr, app.gfx.clone(), extent, 2, SwapchainOpts::new())?;
        let depth = ImageView::new_default(Image::new(
            app.gfx.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: DEPTH_FORMAT,
                extent: [extent[0], extent[1], 1],
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )?)?;
        log::info!(
            "controller models: Touch v3 (WebXR input profiles, MIT), {}x{} per eye",
            extent[0],
            extent[1]
        );
        Ok(Self {
            pipeline,
            models,
            swapchain,
            depth,
            extent,
            drawn: None,
        })
    }

    /// Draw the shown controllers for both eyes (nothing, and no layer, when none is shown).
    pub(super) fn render(
        &mut self,
        app: &AppState,
        views: &[xr::View],
        shown: &[Option<ShownController>; 2],
        futures: &mut GpuFutures,
    ) -> anyhow::Result<()> {
        self.drawn = None;
        if views.len() < 2 || shown.iter().all(Option::is_none) {
            return Ok(());
        }
        let wsi = self.swapchain.acquire_wait_image()?;
        let mut cmd = AutoCommandBufferBuilder::primary(
            app.gfx.command_buffer_allocator.clone(),
            app.gfx.queue_gfx.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        let extent_f = [self.extent[0] as f32, self.extent[1] as f32];
        for eye in 0..2 {
            let v = &views[eye];
            let view = posef_to_transform(&v.pose).inverse();
            let proj = projection_fov(
                v.fov.angle_left,
                v.fov.angle_right,
                v.fov.angle_up,
                v.fov.angle_down,
                NEAR,
                FAR,
            );
            let vp = proj * Mat4::from(view);
            cmd.begin_rendering(RenderingInfo {
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),
                    ..RenderingAttachmentInfo::image_view(wsi.views[eye].clone())
                })],
                depth_attachment: Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::DontCare,
                    clear_value: Some(ClearValue::Depth(1.0)),
                    ..RenderingAttachmentInfo::image_view(self.depth.clone())
                }),
                contents: SubpassContents::Inline,
                ..Default::default()
            })?;
            cmd.set_viewport(
                0,
                smallvec![Viewport {
                    offset: [0.0, 0.0],
                    extent: extent_f,
                    depth_range: 0.0..=1.0,
                }],
            )?
            .set_scissor(
                0,
                smallvec![Scissor {
                    offset: [0, 0],
                    extent: self.extent,
                }],
            )?
            .bind_pipeline_graphics(self.pipeline.clone())?;
            for (hand, s) in shown.iter().enumerate() {
                let Some(s) = s else { continue };
                let gm = &self.models[hand];
                cmd.bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    gm.texture.clone(),
                )?;
                let place = s.grip * model_in_grip();
                for item in gm.model.pose(&s.inputs) {
                    let model = place * item.transform;
                    let mvp = vp * Mat4::from(model);
                    let r = model.matrix3;
                    let col = |c: Vec3A| [c.x, c.y, c.z, 0.0];
                    let pc = PushConstants {
                        mvp: mvp.to_cols_array_2d(),
                        m0: col(r.x_axis),
                        m1: col(r.y_axis),
                        m2: col(r.z_axis),
                        tint: [1.0, 1.0, 1.0, 1.0],
                    };
                    let mesh = &gm.meshes[item.mesh];
                    cmd.push_constants(self.pipeline.layout().clone(), 0, pc)?
                        .bind_vertex_buffers(0, mesh.vertices.clone())?
                        .bind_index_buffer(mesh.indices.clone())?;
                    // SAFETY: indices were checked against the vertex count at load.
                    unsafe { cmd.draw_indexed(mesh.count, 1, 0, 0, 0)? };
                }
            }
            cmd.end_rendering()?;
        }
        futures.execute(app.gfx.queue_gfx.clone(), cmd.build()?)?;
        self.drawn = Some([views[0], views[1]]);
        Ok(())
    }

    /// The projection layer for this frame (None when nothing was drawn). `views_out` holds the per-eye views
    /// the layer points at.
    pub(super) fn present<'a>(
        &'a mut self,
        xr: &'a XrState,
        views_out: &'a mut Vec<xr::CompositionLayerProjectionView<'a, xr::Vulkan>>,
    ) -> anyhow::Result<Option<CompositionLayer<'a>>> {
        self.swapchain.ensure_image_released()?;
        let Some(drawn) = self.drawn.take() else {
            return Ok(None);
        };
        let sc: &'a WlxSwapchain = &self.swapchain;
        views_out.clear();
        for (eye, v) in drawn.iter().enumerate() {
            views_out.push(
                xr::CompositionLayerProjectionView::new()
                    .pose(v.pose)
                    .fov(v.fov)
                    .sub_image(sc.get_subimage(eye as u32)),
            );
        }
        let views: &'a [xr::CompositionLayerProjectionView<'a, xr::Vulkan>] = views_out;
        Ok(Some(CompositionLayer::Projection(
            xr::CompositionLayerProjection::new()
                .layer_flags(xr::CompositionLayerFlags::BLEND_TEXTURE_SOURCE_ALPHA)
                .space(&xr.stage)
                .views(views),
        )))
    }
}
