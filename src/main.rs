mod camera;
mod chunk;
mod render;

use std::collections::HashSet;

use bevy_ecs::prelude::*;
use modul_asset::Assets;
use modul_core::{
    run_app, EventBuffer, GraphicsInitializer, GraphicsInitializerResult, Init,
    MainWindow, QueueRes, Redraw, WindowComponent,
};
use modul_render::{
    InitialSurfaceConfig, RenderPlugin, RenderTargetColorConfig,
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

use camera::{FlyCamera, Position, Rotation, CameraConfig, MainCamera};
use chunk::demand::{ChunkSource, ChunkLoadList};

#[derive(Resource)]
struct FrameCount(u64);

use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

pub static TIMING_DEMAND_US: AtomicU32 = AtomicU32::new(0);
pub static TIMING_LOADING_US: AtomicU32 = AtomicU32::new(0);
pub static TIMING_SYNC_UPLOAD_US: AtomicU32 = AtomicU32::new(0);
pub static TIMING_SYNC_DRAWS_US: AtomicU32 = AtomicU32::new(0);

/// Drop guard that writes elapsed microseconds to an atomic target on drop.
pub struct SysTimer {
    start: std::time::Instant,
    target: &'static AtomicU32,
}

impl SysTimer {
    pub fn new(target: &'static AtomicU32) -> Self {
        Self { start: std::time::Instant::now(), target }
    }
}

impl Drop for SysTimer {
    fn drop(&mut self) {
        self.target.store(self.start.elapsed().as_micros() as u32, Relaxed);
    }
}

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
struct DayCycle {
    angle: f32, // radians, 0 = sunrise east, π/2 = noon overhead
    paused: bool,
}

impl Default for DayCycle {
    fn default() -> Self {
        // Start at ~25° above horizon to match old default
        Self { angle: 0.44, paused: false }
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
                required_limits: wgpu::Limits {
                    max_bind_groups: 5,
                    ..wgpu::Limits::default()
                },
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
    let source = ChunkSource::default();
    let lod_count = source.lod_count as usize;
    let end_radius = source.end_radius;

    run_app(VoxelGraphicsInitializer, |app| {
        app.add_plugins((RenderPlugin, ExitPlugin));

        // Resources
        app.insert_resource(chunk::LodChunkMaps::new(lod_count));
        app.insert_resource(chunk::ChunkChangedQueue::default());
        app.insert_resource(chunk::ChunkUnloadQueue::default());
        app.insert_resource(chunk::LoadedChunkIndex::default());
        app.insert_resource(render::ChunkRenderData::default());
        app.insert_resource(render::DrawCache::default());
        app.insert_resource(render::TransparentDrawCache::default());
        app.insert_resource(chunk::loading::ChunkLoader::default());
        app.insert_resource(render::shadow::grid::ShadowGrid::new(end_radius, lod_count as u32));
        app.insert_resource(render::shadow::grid::BitmaskPool::new());
        app.insert_resource(chunk::generation::GenPool::new());
        app.insert_resource(chunk::meshing::MeshPool::new());
        app.insert_resource(InputState::default());
        app.insert_resource(FpsCounter::default());

        // Init: subsystem init systems run in dependency order
        app.add_systems(Init, (
            init_window,
            init_gameplay,
            render::init_render,
            render::shadow::init_shadow,
            render::taa::init_taa,
            render::atmosphere::init_atmosphere,
            render::wboit::init_wboit,
        ).chain());

        // Gameplay systems (before render). process_input runs first so debug
        // mode and camera state are up-to-date before the demand/loader sees them.
        app.add_systems(
            Redraw,
            (
                (
                    process_input,
                    chunk::demand::update_chunk_demand,
                    chunk::loading::update_chunk_loading,
                    apply_deferred,
                ).chain(),
                (
                    chunk::meshing::resolve_changes,
                    apply_deferred,
                    chunk::meshing::poll_meshing,
                    chunk::meshing::start_meshing,
                ).chain(),
            )
                .chain()
                .before(RenderSystemSet),
        );

        // GPU synchronization
        app.add_systems(
            Synchronize,
            (
                render::cleanup_unloaded_chunks.before(render::synchronize_gpu),
                render::shadow::grid::process_chunk_bitmasks
                    .before(render::shadow::gpu::synchronize_shadow_buffers),
                render::shadow::grid::update_shadow_grid_origins
                    .before(render::shadow::gpu::synchronize_shadow_buffers),
                render::synchronize_gpu,
                render::shadow::gpu::synchronize_shadow_buffers,
                update_day_cycle,
                render::atmosphere::update_atmosphere,
                update_camera.before(render::shadow::pass::update_previous_frame_data),
                render::shadow::pass::update_previous_frame_data,
                chunk::clear_chunk_changed_queue
                    .after(render::shadow::grid::process_chunk_bitmasks),
            ),
        );
    });
}

fn init_window(
    mut commands: Commands,
    main_window: Query<Entity, With<MainWindow>>,
    mut sequences: ResMut<Assets<Sequence>>,
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

    // Render sequence
    let render_target = RenderTargetSource::Surface(window_entity);
    let mut builder = SequenceBuilder::new();
    builder
        .add(render::ClearAll { render_target })
        .add(render::shadow::pass::ShadowDepthOperationBuilder)
        .add(render::shadow::pass::ShadowTraceOperationBuilder)
        .add(render::taa::TaaVoxelDrawOperationBuilder)
        .add(render::atmosphere::SkyPassOperationBuilder)
        .add(render::wboit::TransparentDrawOperationBuilder)
        .add(render::wboit::WboitResolveOperationBuilder)
        .add(render::taa::TaaResolveOperationBuilder {
            surface_entity: window_entity,
        })
        .add(render::shadow::pass::ShadowDebugOverlayBuilder {
            target: render_target,
        });
    let sequence = builder.finish(&mut sequences);
    commands.insert_resource(RunningSequenceQueue(SequenceQueue(vec![sequence])));
}

fn init_gameplay(mut commands: Commands) {
    let mut fly = FlyCamera::default();
    fly.pitch = -0.3;
    fly.speed = 100.0;

    let pos = Position([0.0, 300.0, 200.0]);
    let rot = Rotation(fly.rotation());
    let config = CameraConfig {
        fov_y: 70.0f32.to_radians(),
        near: 0.1,
        far: 50000.0,
    };
    let cam = camera::compute_camera(&pos, &rot, &config, 16.0 / 9.0);

    commands.spawn((pos, rot, config, cam, fly, MainCamera, ChunkSource::default(), ChunkLoadList::default()));

    commands.insert_resource(FrameCount(0));
    commands.insert_resource(DayCycle::default());
    commands.insert_resource(DebugMode::default());
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


/// DDA raycast through LOD-0 chunks. Returns (hit_pos, face_normal) in world block coords.
/// `max_dist` is in blocks.
fn raycast_blocks(
    origin: [f64; 3],
    dir: [f32; 3],
    max_dist: f32,
    lod_maps: &chunk::LodChunkMaps,
    chunk_data_q: &Query<&mut chunk::ChunkData>,
) -> Option<([i32; 3], [i32; 3])> {
    let cs = chunk::CHUNK_SIZE as i32;

    // Current voxel position
    let mut vx = origin[0].floor() as i32;
    let mut vy = origin[1].floor() as i32;
    let mut vz = origin[2].floor() as i32;

    let step_x = if dir[0] >= 0.0 { 1i32 } else { -1 };
    let step_y = if dir[1] >= 0.0 { 1i32 } else { -1 };
    let step_z = if dir[2] >= 0.0 { 1i32 } else { -1 };

    let inv_dx = if dir[0] != 0.0 { 1.0 / dir[0] as f64 } else { f64::MAX };
    let inv_dy = if dir[1] != 0.0 { 1.0 / dir[1] as f64 } else { f64::MAX };
    let inv_dz = if dir[2] != 0.0 { 1.0 / dir[2] as f64 } else { f64::MAX };

    let mut t_max_x = ((if step_x > 0 { vx + 1 } else { vx }) as f64 - origin[0]) * inv_dx;
    let mut t_max_y = ((if step_y > 0 { vy + 1 } else { vy }) as f64 - origin[1]) * inv_dy;
    let mut t_max_z = ((if step_z > 0 { vz + 1 } else { vz }) as f64 - origin[2]) * inv_dz;

    let t_delta_x = (step_x as f64 * inv_dx).abs();
    let t_delta_y = (step_y as f64 * inv_dy).abs();
    let t_delta_z = (step_z as f64 * inv_dz).abs();

    let mut face = [0i32; 3];
    let max_steps = (max_dist * 2.0) as usize;

    for _ in 0..max_steps {
        // Look up chunk and local coords
        let cx = vx.div_euclid(cs);
        let cy = vy.div_euclid(cs);
        let cz = vz.div_euclid(cs);
        let lx = vx.rem_euclid(cs) as usize;
        let ly = vy.rem_euclid(cs) as usize;
        let lz = vz.rem_euclid(cs) as usize;

        let chunk_pos = glam::IVec3::new(cx, cy, cz);
        if let Some(&entity) = lod_maps.maps[0].get(&chunk_pos) {
            if let Ok(data) = chunk_data_q.get(entity) {
                let block = data.0.get(lx, ly, lz);
                if block != chunk::AIR {
                    return Some(([vx, vy, vz], face));
                }
            }
        }


        // Step to next voxel
        if t_max_x < t_max_y && t_max_x < t_max_z {
            if t_max_x > max_dist as f64 { break; }
            vx += step_x;
            t_max_x += t_delta_x;
            face = [-step_x, 0, 0];
        } else if t_max_y < t_max_z {
            if t_max_y > max_dist as f64 { break; }
            vy += step_y;
            t_max_y += t_delta_y;
            face = [0, -step_y, 0];
        } else {
            if t_max_z > max_dist as f64 { break; }
            vz += step_z;
            t_max_z += t_delta_z;
            face = [0, 0, -step_z];
        }
    }

    None
}

fn process_input(
    events: Res<EventBuffer>,
    mut input: ResMut<InputState>,
    mut wireframe: ResMut<render::Wireframe>,
    mut debug: ResMut<DebugMode>,
    mut taa_enabled: ResMut<render::taa::TaaEnabled>,
    mut day_cycle: ResMut<DayCycle>,
    render_data: Res<render::ChunkRenderData>,
    gpu: Res<render::GpuBuffers>,
    frame_count: Res<FrameCount>,
    loaded_index: Res<chunk::LoadedChunkIndex>,
    lod_maps: Res<chunk::LodChunkMaps>,
    mut chunk_data_q: Query<&mut chunk::ChunkData>,
    needs_remesh_q: Query<(), With<chunk::meshing::NeedsRemesh>>,
    window_query: Query<&WindowComponent, With<MainWindow>>,
    mut cam_query: Query<(&mut Position, &mut FlyCamera, &CameraConfig), With<MainCamera>>,
    mut changed_queue: ResMut<chunk::ChunkChangedQueue>,
) {
    let now = std::time::Instant::now();
    input.dt = now.duration_since(input.last_instant).as_secs_f32();
    input.last_instant = now;
    input.mouse_dx = 0.0;
    input.mouse_dy = 0.0;

    let Ok((mut cam_pos, mut fly_cam, cam_config)) = cam_query.get_single_mut() else { return };

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
                        KeyCode::KeyT => {
                            taa_enabled.0 = !taa_enabled.0;
                            println!("TAA: {}", if taa_enabled.0 { "ON" } else { "OFF" });
                        }
                        KeyCode::KeyP => {
                            day_cycle.paused = !day_cycle.paused;
                            println!("Day/night: {}", if day_cycle.paused { "PAUSED" } else { "RUNNING" });
                        }
                        KeyCode::Tab => {
                            if debug.frozen.is_some() {
                                debug.frozen = None;
                                println!("Debug mode OFF");
                            } else {
                                let cp = camera::chunk_pos(&cam_pos);
                                // We need view_proj for frustum planes — compute it temporarily
                                let cam = camera::compute_camera(&cam_pos, &Rotation(fly_cam.rotation()), cam_config, 16.0/9.0);
                                debug.frozen = Some(FrozenCulling {
                                    chunk_pos: cp,
                                    planes: camera::extract_frustum_planes(&cam.view_proj),
                                    camera_world: cam_pos.0,
                                });
                                println!("Debug mode ON - frustum & loading frozen, fly freely to inspect");
                            }
                        }
                        KeyCode::F12 => {
                            let pos = cam_pos.0;
                            let cp = camera::chunk_pos(&cam_pos);
                            println!("=== DEBUG (frame {}) ===", frame_count.0);
                            println!("  Pos: ({:.1}, {:.1}, {:.1})", pos[0], pos[1], pos[2]);
                            println!("  Chunk: ({}, {}, {})", cp.x, cp.y, cp.z);
                            println!("  Pages: {}/{} ({} slabs)",
                                render::PageAllocator::total_used(&gpu),
                                render::PageAllocator::total_capacity(&gpu),
                                gpu.slabs.len());
                            for lod in 0..lod_maps.maps.len() {
                                let in_map = lod_maps.maps[lod].iter().count();
                                let mut has_data = 0u32;
                                let mut waiting_mesh = 0u32;
                                let in_loaded_idx = loaded_index.0.iter()
                                    .filter(|(_, l)| *l == lod as u8).count();
                                for (_, &entity) in lod_maps.maps[lod].iter() {
                                    if chunk_data_q.get(entity).is_ok() { has_data += 1; }
                                    if needs_remesh_q.get(entity).is_ok() { waiting_mesh += 1; }
                                }
                                let gen_pending = in_map - has_data as usize;
                                println!(
                                    "  LOD {}: {} map, {} data, {} gen-wait, {} mesh-wait, {} uploaded",
                                    lod, in_map, has_data, gen_pending, waiting_mesh, in_loaded_idx
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
                event: WindowEvent::MouseInput { state: ElementState::Pressed, button, .. },
                ..
            } => {
                if !input.captured {
                    input.captured = true;
                    if let Ok(wc) = window_query.get_single() {
                        set_cursor_captured(&wc.window, true);
                    }
                } else {
                    let dir = fly_cam.look_dir();
                    let dir_arr = [dir.x, dir.y, dir.z];
                    let ray_hit = raycast_blocks(
                        cam_pos.0, dir_arr, 100.0, &lod_maps, &chunk_data_q,
                    );
                    if let Some((hit, face)) = ray_hit {
                        let cs = chunk::CHUNK_SIZE as i32;
                        let (target, block) = match button {
                            winit::event::MouseButton::Left => (hit, chunk::AIR),
                            winit::event::MouseButton::Right => {
                                ([hit[0] + face[0], hit[1] + face[1], hit[2] + face[2]], chunk::GLASS)
                            }
                            _ => { continue; }
                        };
                        let cx = target[0].div_euclid(cs);
                        let cy = target[1].div_euclid(cs);
                        let cz = target[2].div_euclid(cs);
                        let chunk_pos = glam::IVec3::new(cx, cy, cz);
                        if let Some(&entity) = lod_maps.maps[0].get(&chunk_pos) {
                            if let Ok(mut data) = chunk_data_q.get_mut(entity) {
                                let lx = target[0].rem_euclid(cs) as usize;
                                let ly = target[1].rem_euclid(cs) as usize;
                                let lz = target[2].rem_euclid(cs) as usize;
                                std::sync::Arc::make_mut(&mut data.0).set(lx, ly, lz, block);
                                changed_queue.0.push(chunk::ChunkChange {
                                    entity,
                                    pos: chunk_pos,
                                    lod: 0,
                                });
                            }
                        }
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
                fly_cam.speed = (fly_cam.speed * factor).clamp(1.0, 1000.0);
            }

            _ => {}
        }
    }

    if input.mouse_dx != 0.0 || input.mouse_dy != 0.0 {
        fly_cam.rotate(input.mouse_dx, input.mouse_dy);
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
        fly_cam.move_dir(&mut cam_pos, forward, right, up, dt);
    }
}

fn update_day_cycle(
    input: Res<InputState>,
    mut cycle: ResMut<DayCycle>,
    mut sun_dir: ResMut<render::shadow::pass::SunDirection>,
) {
    if cycle.paused {
        sun_dir.0 = render::atmosphere::sun_direction_at_angle(cycle.angle);
        return;
    }
    let dt = input.dt.min(0.1);
    let day_length = 300.0; // seconds per full rotation
    cycle.angle += dt * std::f32::consts::TAU / day_length;
    cycle.angle %= std::f32::consts::TAU;

    sun_dir.0 = render::atmosphere::sun_direction_at_angle(cycle.angle);
}

fn update_camera(
    mut frame_count: ResMut<FrameCount>,
    mut fps: ResMut<FpsCounter>,
    mut taa_res: ResMut<render::taa::TaaResources>,
    taa_enabled: Res<render::taa::TaaEnabled>,
    debug: Res<DebugMode>,
    queue: Res<QueueRes>,
    camera_bg: Res<render::CameraBindGroup>,
    rt_query: Query<&modul_render::SurfaceRenderTarget, With<MainWindow>>,
    window_query: Query<&WindowComponent, With<MainWindow>>,
    mut cam_query: Query<(&Position, &FlyCamera, &CameraConfig, &mut camera::Camera), With<MainCamera>>,
) {
    frame_count.0 += 1;
    fps.frame_count += 1;

    let elapsed = fps.last_instant.elapsed();
    if elapsed.as_secs_f32() >= 0.5 {
        fps.fps = fps.frame_count as f32 / elapsed.as_secs_f32();
        fps.frame_count = 0;
        fps.last_instant = std::time::Instant::now();

        if let Ok(wc) = window_query.get_single() {
            let title = format!(
                "Voxel Engine \u{2014} {:.0} FPS | demand {}us load {}us sync_up {}us sync_draw {}us{}",
                fps.fps,
                TIMING_DEMAND_US.load(Relaxed),
                TIMING_LOADING_US.load(Relaxed),
                TIMING_SYNC_UPLOAD_US.load(Relaxed),
                TIMING_SYNC_DRAWS_US.load(Relaxed),
                if debug.frozen.is_some() { " [DEBUG]" } else { "" },
            );
            wc.window.set_title(&title);
        }
    }

    let Ok((cam_pos, fly_cam, cam_config, mut cam_component)) = cam_query.get_single_mut() else { return };

    let mut aspect = 16.0 / 9.0;
    if let Ok(rt) = rt_query.get_single() {
        let (w, h) = modul_render::RenderTarget::size(rt);
        if w > 0 && h > 0 {
            aspect = w as f32 / h as f32;
        }
    }

    // Compute and store Camera component so other systems can read it
    *cam_component = camera::compute_camera(cam_pos, &Rotation(fly_cam.rotation()), cam_config, aspect);
    let mut uniform = camera::CameraUniform::from_camera(&cam_component);
    if let Ok(rt) = rt_query.get_single() {
        let (w, h) = modul_render::RenderTarget::size(rt);
        uniform.screen_size = [w as f32, h as f32];
    }

    uniform.frame_index = (frame_count.0 % 16) as u32;

    // Previous frame data — needed by both TAA resolve and shadow temporal reprojection
    if taa_res.prev_valid {
        uniform.prev_jittered_view_proj = taa_res.prev_jittered_view_proj;
        uniform.prev_chunk_offset = taa_res.prev_chunk_offset;
    } else {
        uniform.prev_jittered_view_proj = uniform.view_proj;
        uniform.prev_chunk_offset = uniform.chunk_offset;
    }

    // Sub-pixel jitter (TAA only)
    if taa_enabled.0 {
        let (jx, jy) = camera::taa_jitter(uniform.frame_index);
        uniform.jitter_offset = [jx, -jy];
        camera::apply_jitter(
            &mut uniform.view_proj,
            jx,
            jy,
            uniform.screen_size[0],
            uniform.screen_size[1],
        );
    }

    // Store this frame's data for next frame's reprojection
    taa_res.prev_jittered_view_proj = uniform.view_proj;
    taa_res.prev_chunk_offset = uniform.chunk_offset;
    taa_res.prev_valid = true;

    // Inverse of (possibly jittered) VP for depth reconstruction
    uniform.inv_view_proj = camera::invert_mat4(&uniform.view_proj);

    queue
        .0
        .write_buffer(&camera_bg.buffer, 0, bytemuck::bytes_of(&uniform));
}
