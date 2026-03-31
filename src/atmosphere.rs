use bevy_ecs::prelude::*;
use bytemuck::{Pod, Zeroable};
use modul_render::{
    BindGroupLayoutProvider, Operation, OperationBuilder, RenderTarget, RenderTargetSource,
};
use wgpu::{
    BufferDescriptor, BufferUsages, CommandEncoder, Device, TextureFormat, TextureUsages,
};

use crate::render;
use crate::shadow::pass::SunDirection;

// --- Atmosphere scattering constants (Earth-like) ---

const PLANET_RADIUS: f64 = 6371e3;
const ATMOSPHERE_RADIUS: f64 = PLANET_RADIUS + 100e3;
const RAYLEIGH_SCALE_HEIGHT: f64 = 8500.0;
const MIE_SCALE_HEIGHT: f64 = 1200.0;
const RAYLEIGH_COEFF: [f64; 3] = [5.5e-6, 13.0e-6, 22.4e-6];
const MIE_COEFF: f64 = 21e-6;
const MIE_EXTINCTION: f64 = MIE_COEFF / 0.9;
const MIE_G: f64 = 0.76;
const NUM_SCATTER_STEPS: usize = 32;
const NUM_OPTICAL_DEPTH_STEPS: usize = 8;
const SUN_IRRADIANCE: f64 = 10.0;
const LUT_SLICES: u32 = 64;

// --- Resources ---

#[derive(Resource)]
pub struct AtmosphereConfig {
    pub fog_density: f32,
    pub sun_angular_radius: f32,
    pub sun_intensity: f32,
    pub lut_width: u32,
    pub lut_height: u32,
}

impl Default for AtmosphereConfig {
    fn default() -> Self {
        Self {
            fog_density: 0.0005,
            sun_angular_radius: 0.00935, // ~0.535 degrees
            sun_intensity: 20.0,
            lut_width: 64,
            lut_height: 64,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct AtmosphereUniform {
    pub sun_direction: [f32; 3],
    pub sun_angular_radius: f32,
    pub fog_density: f32,
    pub sun_intensity: f32,
    pub lut_w: f32,
    pub _pad: f32,
}

#[derive(Resource)]
pub struct AtmosphereResources {
    pub lut_texture: wgpu::Texture,
    pub lut_view: wgpu::TextureView,
    pub fog_lut_texture: wgpu::Texture,
    pub fog_lut_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub sky_pipeline: wgpu::RenderPipeline,
}

// --- Bind group layout ---

pub struct AtmosphereBGLayout;

impl BindGroupLayoutProvider for AtmosphereBGLayout {
    fn layout(&self) -> wgpu::BindGroupLayoutDescriptor<'_> {
        static ENTRIES: [wgpu::BindGroupLayoutEntry; 4] = [
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZero::new(
                        std::mem::size_of::<AtmosphereUniform>() as u64,
                    ),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
        ];
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Atmosphere BG Layout"),
            entries: &ENTRIES,
        }
    }

    fn library(&self) -> &str {
        "\
struct AtmosphereUniform {
    sun_direction: vec3<f32>,
    sun_angular_radius: f32,
    fog_density: f32,
    sun_intensity: f32,
    lut_w: f32,
    _pad: f32,
};

@group(#BIND_GROUP) @binding(0)
var<uniform> atmosphere: AtmosphereUniform;
@group(#BIND_GROUP) @binding(1)
var sky_lut: texture_3d<f32>;
@group(#BIND_GROUP) @binding(2)
var sky_sampler: sampler;
@group(#BIND_GROUP) @binding(3)
var fog_lut: texture_3d<f32>;
"
    }
}

// --- Sun orbit ---

/// Compute sun direction for a given orbit angle.
/// angle=0: east horizon, angle=π/2: zenith, angle=π: west horizon, etc.
pub fn sun_direction_at_angle(angle: f32) -> [f32; 3] {
    let x = angle.cos();
    let y = angle.sin();
    let z = 0.3f32;
    let len = (x * x + y * y + z * z).sqrt();
    [x / len, y / len, z / len]
}

// --- Sky LUT baking (CPU-side Rayleigh/Mie single scattering) ---

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length3(v: [f64; 3]) -> f64 {
    dot3(v, v).sqrt()
}

fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Returns the far intersection distance of a ray with a sphere centered at origin.
/// Returns None if no intersection or only behind the ray.
fn ray_sphere_exit(origin: [f64; 3], dir: [f64; 3], radius: f64) -> Option<f64> {
    let b = dot3(origin, dir);
    let c = dot3(origin, origin) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = -b + disc.sqrt();
    if t > 0.0 { Some(t) } else { None }
}

/// Returns the near intersection distance (entering the sphere). None if no hit or behind ray.
fn ray_sphere_enter(origin: [f64; 3], dir: [f64; 3], radius: f64) -> Option<f64> {
    let b = dot3(origin, dir);
    let c = dot3(origin, origin) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    if t > 0.0 { Some(t) } else { None }
}

/// Integrate atmospheric density along a ray for optical depth calculation.
fn integrate_optical_depth(
    origin: [f64; 3],
    dir: [f64; 3],
    length: f64,
) -> (f64, f64) {
    let step = length / NUM_OPTICAL_DEPTH_STEPS as f64;
    let mut od_r = 0.0;
    let mut od_m = 0.0;
    for i in 0..NUM_OPTICAL_DEPTH_STEPS {
        let t = (i as f64 + 0.5) * step;
        let pos = add3(origin, scale3(dir, t));
        let height = length3(pos) - PLANET_RADIUS;
        if height < 0.0 {
            continue;
        }
        od_r += (-height / RAYLEIGH_SCALE_HEIGHT).exp() * step;
        od_m += (-height / MIE_SCALE_HEIGHT).exp() * step;
    }
    (od_r, od_m)
}

fn rayleigh_phase(cos_theta: f64) -> f64 {
    3.0 / (16.0 * std::f64::consts::PI) * (1.0 + cos_theta * cos_theta)
}

fn mie_phase(cos_theta: f64, g: f64) -> f64 {
    let g2 = g * g;
    let num = (1.0 - g2) * (1.0 + cos_theta * cos_theta);
    let denom = (2.0 + g2) * (1.0 + g2 - 2.0 * g * cos_theta).powf(1.5);
    3.0 / (8.0 * std::f64::consts::PI) * num / denom
}

/// Compute sky color for a given view direction and sun direction.
/// `mie_g` controls Mie asymmetry: 0.76 for full forward scattering, 0.0 for isotropic (no glow).
fn compute_sky_color(view_dir: [f64; 3], sun_dir: [f64; 3], mie_g: f64) -> [f64; 3] {
    let cam_pos = [0.0, PLANET_RADIUS + 1.0, 0.0];

    let atm_dist = match ray_sphere_exit(cam_pos, view_dir, ATMOSPHERE_RADIUS) {
        Some(t) => t,
        None => return [0.0; 3],
    };

    // Check if view ray hits the ground
    let max_dist = if let Some(t) = ray_sphere_enter(cam_pos, view_dir, PLANET_RADIUS) {
        if t > 0.0 { t } else { atm_dist }
    } else {
        atm_dist
    };

    let cos_theta = dot3(view_dir, sun_dir);
    let phase_r = rayleigh_phase(cos_theta);
    let phase_m = mie_phase(cos_theta, mie_g);

    let step = max_dist / NUM_SCATTER_STEPS as f64;
    let mut sum_r = [0.0f64; 3];
    let mut sum_m = [0.0f64; 3];
    let mut od_r = 0.0;
    let mut od_m = 0.0;

    for i in 0..NUM_SCATTER_STEPS {
        let t = (i as f64 + 0.5) * step;
        let pos = add3(cam_pos, scale3(view_dir, t));
        let height = length3(pos) - PLANET_RADIUS;
        if height < 0.0 {
            break;
        }

        let hr = (-height / RAYLEIGH_SCALE_HEIGHT).exp() * step;
        let hm = (-height / MIE_SCALE_HEIGHT).exp() * step;

        od_r += hr;
        od_m += hm;

        // Sun ray from scatter point
        let sun_dist = match ray_sphere_exit(pos, sun_dir, ATMOSPHERE_RADIUS) {
            Some(t) => t,
            None => continue,
        };
        // Check if sun is occluded by the planet from this point
        if ray_sphere_enter(pos, sun_dir, PLANET_RADIUS).is_some_and(|t| t > 0.0) {
            continue;
        }

        let (sun_od_r, sun_od_m) = integrate_optical_depth(pos, sun_dir, sun_dist);

        for c in 0..3 {
            let tau = RAYLEIGH_COEFF[c] * (od_r + sun_od_r)
                + MIE_EXTINCTION * (od_m + sun_od_m);
            let atten = (-tau).exp();
            sum_r[c] += hr * atten;
            sum_m[c] += hm * atten;
        }
    }

    let mut color = [0.0; 3];
    for c in 0..3 {
        color[c] = SUN_IRRADIANCE
            * (sum_r[c] * RAYLEIGH_COEFF[c] * phase_r
                + sum_m[c] * MIE_COEFF * phase_m);
    }
    color
}

/// Convert f32 to IEEE 754 half-precision (f16) stored as u16.
fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    if exp == 0xFF {
        return (sign | 0x7C00 | if mantissa != 0 { 0x200 } else { 0 }) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        return sign as u16;
    }

    (sign | ((new_exp as u32) << 10) | (mantissa >> 13)) as u16
}

/// Bake a single 2D LUT slice for a given sun direction and Mie asymmetry.
fn bake_lut(sun_dir: [f32; 3], width: u32, height: u32, mie_g: f64) -> Vec<[u16; 4]> {
    let sun = [sun_dir[0] as f64, sun_dir[1] as f64, sun_dir[2] as f64];
    let sun_yaw = sun[0].atan2(sun[2]);

    let mut data = vec![[0u16; 4]; (width * height) as usize];

    for y in 0..height {
        let v = (y as f64 + 0.5) / height as f64;
        let elevation = (v - 0.5) * std::f64::consts::PI; // [-π/2, π/2]
        let ce = elevation.cos();
        let se = elevation.sin();

        for x in 0..width {
            let u = (x as f64 + 0.5) / width as f64;
            let rel_yaw = (u - 0.5) * 2.0 * std::f64::consts::PI; // [-π, π]

            let view_yaw = sun_yaw + rel_yaw;
            let view_dir = [view_yaw.sin() * ce, se, view_yaw.cos() * ce];

            let color = compute_sky_color(view_dir, sun, mie_g);

            let idx = (y * width + x) as usize;
            data[idx] = [
                f32_to_f16(color[0] as f32),
                f32_to_f16(color[1] as f32),
                f32_to_f16(color[2] as f32),
                f32_to_f16(1.0),
            ];
        }
    }

    data
}

/// Prebake all LUT slices for every sun angle around the orbit.
fn bake_all_slices(width: u32, height: u32, mie_g: f64) -> Vec<[u16; 4]> {
    let mut data = Vec::with_capacity((width * height * LUT_SLICES) as usize);
    for i in 0..LUT_SLICES {
        let angle = i as f32 * std::f32::consts::TAU / LUT_SLICES as f32;
        let sun_dir = sun_direction_at_angle(angle);
        data.extend_from_slice(&bake_lut(sun_dir, width, height, mie_g));
    }
    data
}

// --- GPU resource creation ---

fn create_lut_texture_3d(
    device: &Device,
    width: u32,
    height: u32,
    depth: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Sky LUT 3D"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: depth,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

impl AtmosphereResources {
    pub fn new(
        device: &Device,
        queue: &wgpu::Queue,
        sun_dir: &SunDirection,
        config: &AtmosphereConfig,
        surface_format: TextureFormat,
    ) -> Self {
        let w = config.lut_width;
        let h = config.lut_height;

        let (lut_texture, lut_view) = create_lut_texture_3d(device, w, h, LUT_SLICES);
        let (fog_lut_texture, fog_lut_view) = create_lut_texture_3d(device, w, h, LUT_SLICES);

        // Prebake all slices
        let sky_data = bake_all_slices(w, h, MIE_G);
        let fog_data = bake_all_slices(w, h, 0.0);

        let extent = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: LUT_SLICES,
        };
        let layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 8),
            rows_per_image: Some(h),
        };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &lut_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&sky_data),
            layout,
            extent,
        );
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &fog_lut_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&fog_data),
            layout,
            extent,
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Sky LUT sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat, // yaw wraps
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::Repeat, // day cycle wraps
            ..Default::default()
        });

        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Atmosphere uniform"),
            size: std::mem::size_of::<AtmosphereUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg_layout = device.create_bind_group_layout(&AtmosphereBGLayout.layout());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Atmosphere BG"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&fog_lut_view),
                },
            ],
        });

        // Sky pipeline
        let camera_layout = device.create_bind_group_layout(&render::CameraBGLayout.layout());
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky pipeline layout"),
            bind_group_layouts: &[&camera_layout, &bg_layout],
            push_constant_ranges: &[],
        });

        let camera_wgsl = render::CameraBGLayout.library().replace("#BIND_GROUP", "0");
        let atmo_wgsl = AtmosphereBGLayout.library().replace("#BIND_GROUP", "1");
        let sky_shader_src = include_str!("sky.wgsl");
        let full_source = format!("{camera_wgsl}\n{atmo_wgsl}\n{sky_shader_src}");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sky shader"),
            source: wgpu::ShaderSource::Wgsl(full_source.into()),
        });

        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_sky"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_sky"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        // Upload initial uniform
        let lut_w = sun_dir.0[1].atan2(sun_dir.0[0]).rem_euclid(std::f32::consts::TAU)
            / std::f32::consts::TAU;
        let uniform = AtmosphereUniform {
            sun_direction: sun_dir.0,
            sun_angular_radius: config.sun_angular_radius,
            fog_density: config.fog_density,
            sun_intensity: config.sun_intensity,
            lut_w,
            _pad: 0.0,
        };
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniform));

        Self {
            lut_texture,
            lut_view,
            fog_lut_texture,
            fog_lut_view,
            sampler,
            uniform_buffer,
            bind_group,
            sky_pipeline,
        }
    }

    pub fn update(
        &self,
        queue: &wgpu::Queue,
        sun_dir: &SunDirection,
        config: &AtmosphereConfig,
    ) {
        let lut_w = sun_dir.0[1].atan2(sun_dir.0[0]).rem_euclid(std::f32::consts::TAU)
            / std::f32::consts::TAU;
        let uniform = AtmosphereUniform {
            sun_direction: sun_dir.0,
            sun_angular_radius: config.sun_angular_radius,
            fog_density: config.fog_density,
            sun_intensity: config.sun_intensity,
            lut_w,
            _pad: 0.0,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }
}

// --- Systems ---

pub fn update_atmosphere(
    sun_dir: Res<SunDirection>,
    config: Res<AtmosphereConfig>,
    atmo_res: Res<AtmosphereResources>,
    queue: Res<modul_core::QueueRes>,
) {
    atmo_res.update(&queue.0, &sun_dir, &config);
}

// --- Sky Pass Operation ---

pub struct SkyPassOperation;

impl Operation for SkyPassOperation {
    fn run(&mut self, world: &mut World, command_encoder: &mut CommandEncoder) {
        // Get scene texture view and depth view from TaaResources and surface
        let scene_view_ptr: *const wgpu::TextureView;
        let depth_view_ptr: *const wgpu::TextureView;
        let pipeline_ptr: *const wgpu::RenderPipeline;
        let atmo_bg_ptr: *const wgpu::BindGroup;
        {
            let taa_res = world.resource::<crate::taa::TaaResources>();
            scene_view_ptr = &taa_res.scene_view as *const _;
        }
        {
            let main_window = world
                .query_filtered::<bevy_ecs::prelude::Entity, bevy_ecs::prelude::With<modul_core::MainWindow>>()
                .single(world);
            let surface_rt = world
                .get::<modul_render::SurfaceRenderTarget>(main_window)
                .unwrap();
            depth_view_ptr = surface_rt.depth_stencil_view().unwrap() as *const _;
        }
        {
            let atmo_res = world.resource::<AtmosphereResources>();
            pipeline_ptr = &atmo_res.sky_pipeline as *const _;
            atmo_bg_ptr = &atmo_res.bind_group as *const _;
        }

        // SAFETY: All views/resources live in World storage, stable during run().
        let scene_view = unsafe { &*scene_view_ptr };
        let depth_view = unsafe { &*depth_view_ptr };
        let sky_pipeline = unsafe { &*pipeline_ptr };
        let atmo_bg = unsafe { &*atmo_bg_ptr };
        let camera_bg = &world.resource::<render::CameraBindGroup>().bind_group;

        let mut pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Sky pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: scene_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // preserve voxel geometry
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load, // preserve depth from voxel pass
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(sky_pipeline);
        pass.set_bind_group(0, camera_bg, &[]);
        pass.set_bind_group(1, atmo_bg, &[]);
        pass.draw(0..3, 0..1); // fullscreen triangle
    }
}

pub struct SkyPassOperationBuilder;

impl OperationBuilder for SkyPassOperationBuilder {
    fn reading(&self) -> Vec<RenderTargetSource> {
        Vec::new()
    }
    fn writing(&self) -> Vec<RenderTargetSource> {
        Vec::new()
    }
    fn finish(self, _world: &World, _device: &Device) -> impl Operation + 'static {
        SkyPassOperation
    }
}
