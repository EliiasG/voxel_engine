mod camera;
mod chunk;
mod render;
mod shadow;

use std::collections::HashSet;

use bevy_ecs::prelude::*;
use modul_asset::Assets;
use modul_core::{
    run_app, DeviceRes, EventBuffer, GraphicsInitializer, GraphicsInitializerResult, Init,
    MainWindow, QueueRes, Redraw, WindowComponent,
};
use modul_render::{
    InitialSurfaceConfig, RenderPipelineManager, RenderPlugin, RenderTargetColorConfig,
    RenderTargetSource, RunningSequenceQueue, Sequence, SequenceBuilder, SequenceQueue,
    SurfaceRenderTargetConfig, RenderTargetDepthStencilConfig, RenderSystemSet, Synchronize,
};
use modul_util::ExitPlugin;
use wgpu::{
    Backends, Color, DeviceDescriptor, Features, Instance, InstanceDescriptor,
    PowerPreference, PresentMode, RequestAdapterOptions, TextureFormat,
    TextureUsages,
};
use winit::event::{DeviceEvent, ElementState, Event, KeyEvent, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, WindowAttributes};

use camera::FlyCamera;

#[derive(Resource)]
struct Camera(FlyCamera);

#[derive(Resource)]
struct FrameCount(u64);

#[derive(Resource)]
struct FpsCounter {
    last_instant: std::time::Instant,
    frame_count: u32,
    fps: f32,
}

impl Default for FpsCounter {
    fn default() -> Self {
        Self {
            last_instant: std::time::Instant::now(),
            frame_count: 0,
            fps: 0.0,
        }
    }
}

#[derive(Resource)]
struct InputState {
    keys: HashSet<KeyCode>,
    mouse_dx: f64,
    mouse_dy: f64,
    captured: bool,
    last_instant: std::time::Instant,
    dt: f32,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            keys: HashSet::new(),
            mouse_dx: 0.0,
            mouse_dy: 0.0,
            captured: false,
            last_instant: std::time::Instant::now(),
            dt: 1.0 / 60.0,
        }
    }
}

pub struct FrozenCulling {
    pub chunk_pos: glam::IVec3,
    pub planes: [[f32; 4]; 6],
    pub camera_world: [f64; 3],
}

#[derive(Resource, Default)]
pub struct DebugMode {
    pub frozen: Option<FrozenCulling>,
}

struct VoxelGraphicsInitializer;

impl GraphicsInitializer for VoxelGraphicsInitializer {
    fn initialize(
        self,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) -> GraphicsInitializerResult {
        env_logger::init();
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let window = std::sync::Arc::new(
            event_loop
                .create_window(WindowAttributes::default().with_title("Voxel Engine v1"))
                .expect("failed to create window"),
        );

        let surface = instance
            .create_surface(window.clone())
            .expect("no surface?");

        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }))
        .expect("no adapter?");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: None,
                required_features: Features::POLYGON_MODE_LINE | Features::MULTI_DRAW_INDIRECT,
                ..Default::default()
            },
            None,
        ))
        .expect("no device?");

        let surface_format = surface
            .get_capabilities(&adapter)
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .expect("SRGB not supported");

        GraphicsInitializerResult {
            window,
            surface,
            instance,
            adapter,
            device,
            queue,
            window_attribs: WindowAttributes::default().with_title("Voxel Engine v1"),
            surface_format,
        }
    }
}

fn main() {
    let load_config = chunk::loading::LoadConfig::default();
    let lod_count = load_config.lod_count as usize;

    run_app(VoxelGraphicsInitializer, |app| {
        app.add_plugins((RenderPlugin, ExitPlugin));

        // Resources
        app.insert_resource(chunk::LodChunkMaps::new(lod_count));
        app.insert_resource(chunk::ChunkChangedQueue::default());
        app.insert_resource(chunk::LoadedChunkIndex::default());
        app.insert_resource(render::ChunkRenderData::default());
        app.insert_resource(chunk::loading::Loader::default());
        app.insert_resource(shadow::grid::ShadowGrid::new(&load_config));
        app.insert_resource(shadow::grid::BitmaskPool::new());
        app.insert_resource(load_config);
        app.insert_resource(chunk::generation::GenPool::new());
        app.insert_resource(chunk::meshing::MeshPool::new());
        app.insert_resource(InputState::default());
        app.insert_resource(FpsCounter::default());

        // Init
        app.add_systems(Init, init);

        // Gameplay systems (before render). process_input runs first so debug
        // mode and camera state are up-to-date before the loader sees them.
        app.add_systems(
            Redraw,
            (
                process_input,
                chunk::loading::update_loader,
                apply_deferred,
                chunk::generation::poll_generation,
                chunk::generation::start_generation,
                apply_deferred,
                chunk::meshing::resolve_changes,
                apply_deferred,
                chunk::meshing::poll_meshing,
                chunk::meshing::start_meshing,
            )
                .chain()
                .before(RenderSystemSet),
        );

        // GPU synchronization
        app.add_systems(
            Synchronize,
            (
                render::synchronize_gpu,
                shadow::gpu::synchronize_shadow_buffers,
                update_camera,
                shadow::pass::update_previous_frame_data,
            ),
        );
    });
}

fn init(
    mut commands: Commands,
    device: Res<DeviceRes>,
    main_window: Query<Entity, With<MainWindow>>,
    mut shaders: ResMut<Assets<wgpu::ShaderModule>>,
    mut layouts: ResMut<Assets<wgpu::PipelineLayout>>,
    mut pipelines: ResMut<Assets<RenderPipelineManager>>,
    mut sequences: ResMut<Assets<Sequence>>,
    shadow_grid: Res<shadow::grid::ShadowGrid>,
) {
    let window_entity = main_window.single();

    commands
        .entity(window_entity)
        .insert(InitialSurfaceConfig(SurfaceRenderTargetConfig {
            color_config: RenderTargetColorConfig {
                multisample_config: None,
                clear_color: Color {
                    r: 0.1,
                    g: 0.15,
                    b: 0.25,
                    a: 1.0,
                },
                usages: TextureUsages::RENDER_ATTACHMENT,
                format_override: None,
            },
            depth_stencil_config: Some(RenderTargetDepthStencilConfig {
                clear_depth: 0.0,
                clear_stencil: 0,
                usages: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                format: TextureFormat::Depth32Float,
            }),
            desired_maximum_frame_latency: 2,
            present_mode: PresentMode::AutoVsync,
            backup_present_mode: None,
        }));

    let gpu_buffers = render::create_gpu_buffers(&device.0);
    let camera_bg = render::create_camera_bind_group(&device.0);
    let shadow_gpu = shadow::gpu::ShadowGpuBuffers::new(&device.0, &shadow_grid);

    let voxel_pipeline = render::init_pipelines(
        &device.0,
        &mut pipelines,
        &mut shaders,
        &mut layouts,
    );

    let shadow_pass_res = shadow::pass::ShadowPassResources::new(
        &device.0, &shadow_gpu, 800, 600, 1, // initial size, full res for debugging
    );

    let render_target = RenderTargetSource::Surface(window_entity);
    let mut builder = SequenceBuilder::new();
    builder
        .add(render::ClearAll { render_target })
        .add(shadow::pass::ShadowTraceOperationBuilder)
        .add(render::VoxelDrawOperationBuilder {
            target: render_target,
        })
        .add(shadow::pass::ShadowDebugOverlayBuilder {
            target: render_target,
        });
    let sequence = builder.finish(&mut sequences);

    let mut cam = FlyCamera::new([0.0, 300.0, 200.0]);
    cam.pitch = -0.3;
    cam.speed = 100.0;
    cam.far = 50000.0;

    commands.insert_resource(Camera(cam));
    commands.insert_resource(FrameCount(0));
    commands.insert_resource(gpu_buffers);
    commands.insert_resource(camera_bg);
    commands.insert_resource(shadow_gpu);
    commands.insert_resource(shadow_pass_res);
    commands.insert_resource(shadow::pass::ShadowConfig::default());
    commands.insert_resource(shadow::pass::SunDirection::default());
    commands.insert_resource(shadow::pass::PreviousFrameData::default());
    commands.insert_resource(voxel_pipeline);
    commands.insert_resource(render::PageAllocator::new());
    commands.insert_resource(render::Wireframe(false));
    commands.insert_resource(DebugMode::default());
    commands.insert_resource(RunningSequenceQueue(SequenceQueue(vec![sequence])));
}

fn set_cursor_captured(window: &winit::window::Window, captured: bool) {
    if captured {
        if window.set_cursor_grab(CursorGrabMode::Locked).is_err() {
            let _ = window.set_cursor_grab(CursorGrabMode::Confined);
        }
        window.set_cursor_visible(false);
    } else {
        let _ = window.set_cursor_grab(CursorGrabMode::None);
        window.set_cursor_visible(true);
    }
}

fn process_input(
    events: Res<EventBuffer>,
    mut input: ResMut<InputState>,
    mut wireframe: ResMut<render::Wireframe>,
    mut debug: ResMut<DebugMode>,
    mut camera: ResMut<Camera>,
    render_data: Res<render::ChunkRenderData>,
    gpu: Res<render::GpuBuffers>,
    frame_count: Res<FrameCount>,
    loaded_index: Res<chunk::LoadedChunkIndex>,
    lod_maps: Res<chunk::LodChunkMaps>,
    config: Res<chunk::loading::LoadConfig>,
    chunk_data_q: Query<(), With<chunk::ChunkData>>,
    needs_gen_q: Query<(), With<chunk::NeedsGeneration>>,
    needs_remesh_q: Query<(), With<chunk::meshing::NeedsRemesh>>,
    has_faces_q: Query<(), With<chunk::meshing::ChunkFaces>>,
    window_query: Query<&WindowComponent, With<MainWindow>>,
) {
    let now = std::time::Instant::now();
    input.dt = now.duration_since(input.last_instant).as_secs_f32();
    input.last_instant = now;
    input.mouse_dx = 0.0;
    input.mouse_dy = 0.0;

    for event in events.events() {
        match event {
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                physical_key: PhysicalKey::Code(key),
                                state,
                                repeat: false,
                                ..
                            },
                        ..
                    },
                ..
            } => {
                match state {
                    ElementState::Pressed => { input.keys.insert(*key); }
                    ElementState::Released => { input.keys.remove(key); }
                }
                if *state == ElementState::Pressed {
                    match key {
                        KeyCode::Escape => {
                            input.captured = !input.captured;
                            if let Ok(wc) = window_query.get_single() {
                                set_cursor_captured(&wc.window, input.captured);
                            }
                        }
                        KeyCode::KeyF => {
                            wireframe.0 = !wireframe.0;
                            println!("Wireframe: {}", wireframe.0);
                        }
                        KeyCode::Tab => {
                            if debug.frozen.is_some() {
                                debug.frozen = None;
                                println!("Debug mode OFF");
                            } else {
                                let u = camera.0.uniform();
                                debug.frozen = Some(FrozenCulling {
                                    chunk_pos: glam::IVec3::from_array(u.chunk_offset),
                                    planes: camera::extract_frustum_planes(&u.view_proj),
                                    camera_world: camera.0.position,
                                });
                                println!("Debug mode ON - frustum & loading frozen, fly freely to inspect");
                            }
                        }
                        KeyCode::F12 => {
                            let pos = camera.0.position;
                            let cp = camera.0.chunk_pos();
                            println!("=== DEBUG (frame {}) ===", frame_count.0);
                            println!("  Pos: ({:.1}, {:.1}, {:.1})", pos[0], pos[1], pos[2]);
                            println!("  Chunk: ({}, {}, {})", cp.x, cp.y, cp.z);
                            println!("  Pages: {}/{} ({} slabs)",
                                render::PageAllocator::total_used(&gpu),
                                render::PageAllocator::total_capacity(&gpu),
                                gpu.slabs.len());
                            for lod in 0..config.lod_count {
                                let in_map = lod_maps.maps[lod as usize].iter().count();
                                let mut has_data = 0u32;
                                let mut waiting_gen = 0u32;
                                let mut waiting_mesh = 0u32;
                                let mut has_faces = 0u32;
                                let in_loaded_idx = loaded_index.0.iter()
                                    .filter(|(_, l)| *l == lod as u8).count();
                                for (_, &entity) in lod_maps.maps[lod as usize].iter() {
                                    if chunk_data_q.get(entity).is_ok() { has_data += 1; }
                                    if needs_gen_q.get(entity).is_ok() { waiting_gen += 1; }
                                    if needs_remesh_q.get(entity).is_ok() { waiting_mesh += 1; }
                                    if has_faces_q.get(entity).is_ok() { has_faces += 1; }
                                }
                                println!(
                                    "  LOD {}: {} map, {} data, {} gen-wait, {} mesh-wait, {} faces, {} uploaded",
                                    lod, in_map, has_data, waiting_gen, waiting_mesh, has_faces, in_loaded_idx
                                );
                            }
                            let mut total_faces = 0u32;
                            let mut standard_faces = 0u32;
                            let mut page_count = 0u32;
                            let mut page_face_sum = 0u32;
                            for entry in render_data.entries.values() {
                                for dp in &entry.directions {
                                    total_faces += dp.total_faces;
                                    standard_faces += dp.standard_faces;
                                    for page in &dp.pages {
                                        page_count += 1;
                                        page_face_sum += page.face_count;
                                    }
                                }
                            }
                            let avg_fill = if page_count > 0 {
                                page_face_sum as f32 / page_count as f32 / render::PAGE_SIZE as f32 * 100.0
                            } else { 0.0 };
                            let total_draws: u32 = gpu.draws.iter().map(|d| d.count).sum();
                            println!("  Faces: {} standard + {} border = {} total",
                                standard_faces, total_faces - standard_faces, total_faces);
                            println!("  Page fill: {:.1}% avg ({} pages)", avg_fill, page_count);
                            println!("  Total draws: {}", total_draws);
                            println!("  Frustum culled: {} chunks", gpu.frustum_culled);
                            if debug.frozen.is_some() {
                                println!("  [DEBUG MODE ACTIVE]");
                            }
                            println!("========================");
                        }
                        _ => {}
                    }
                }
            }

            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta: (dx, dy) },
                ..
            } => {
                if input.captured {
                    input.mouse_dx += dx;
                    input.mouse_dy += dy;
                }
            }

            Event::WindowEvent {
                event: WindowEvent::MouseInput { state: ElementState::Pressed, .. },
                ..
            } => {
                if !input.captured {
                    input.captured = true;
                    if let Ok(wc) = window_query.get_single() {
                        set_cursor_captured(&wc.window, true);
                    }
                }
            }

            Event::WindowEvent {
                event: WindowEvent::Focused(false),
                ..
            } => {
                if input.captured {
                    input.captured = false;
                    if let Ok(wc) = window_query.get_single() {
                        set_cursor_captured(&wc.window, false);
                    }
                }
            }

            Event::WindowEvent {
                event: WindowEvent::MouseWheel { delta, .. },
                ..
            } => {
                let y = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y as f64,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y / 30.0,
                };
                let factor = 1.15f32.powf(y as f32);
                camera.0.speed = (camera.0.speed * factor).clamp(1.0, 1000.0);
            }

            _ => {}
        }
    }

    if input.mouse_dx != 0.0 || input.mouse_dy != 0.0 {
        camera.0.rotate(input.mouse_dx, input.mouse_dy);
    }

    let dt = input.dt.min(0.1);
    let mut forward = 0.0f32;
    let mut right = 0.0f32;
    let mut up = 0.0f32;

    if input.keys.contains(&KeyCode::KeyW) { forward += 1.0; }
    if input.keys.contains(&KeyCode::KeyS) { forward -= 1.0; }
    if input.keys.contains(&KeyCode::KeyD) { right += 1.0; }
    if input.keys.contains(&KeyCode::KeyA) { right -= 1.0; }
    if input.keys.contains(&KeyCode::Space) { up += 1.0; }
    if input.keys.contains(&KeyCode::ShiftLeft) || input.keys.contains(&KeyCode::ShiftRight) {
        up -= 1.0;
    }
    if input.keys.contains(&KeyCode::ControlLeft) || input.keys.contains(&KeyCode::ControlRight) {
        forward *= 3.0;
        right *= 3.0;
        up *= 3.0;
    }

    if forward != 0.0 || right != 0.0 || up != 0.0 {
        camera.0.move_dir(forward, right, up, dt);
    }
}

fn update_camera(
    mut camera: ResMut<Camera>,
    mut frame_count: ResMut<FrameCount>,
    mut fps: ResMut<FpsCounter>,
    debug: Res<DebugMode>,
    queue: Res<QueueRes>,
    camera_bg: Res<render::CameraBindGroup>,
    rt_query: Query<&modul_render::SurfaceRenderTarget, With<MainWindow>>,
    window_query: Query<&WindowComponent, With<MainWindow>>,
) {
    frame_count.0 += 1;
    fps.frame_count += 1;

    let elapsed = fps.last_instant.elapsed();
    if elapsed.as_secs_f32() >= 0.5 {
        fps.fps = fps.frame_count as f32 / elapsed.as_secs_f32();
        fps.frame_count = 0;
        fps.last_instant = std::time::Instant::now();

        if let Ok(wc) = window_query.get_single() {
            let title = if debug.frozen.is_some() {
                format!("Voxel Engine \u{2014} {:.0} FPS [DEBUG]", fps.fps)
            } else {
                format!("Voxel Engine \u{2014} {:.0} FPS", fps.fps)
            };
            wc.window.set_title(&title);
        }
    }

    if let Ok(rt) = rt_query.get_single() {
        let (w, h) = modul_render::RenderTarget::size(rt);
        if w > 0 && h > 0 {
            camera.0.aspect = w as f32 / h as f32;
        }
    }

    let uniform = camera.0.uniform();
    queue
        .0
        .write_buffer(&camera_bg.buffer, 0, bytemuck::bytes_of(&uniform));
}
