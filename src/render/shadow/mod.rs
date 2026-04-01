pub mod bitmask;
pub mod gpu;
pub mod grid;
pub mod pass;

use bevy_ecs::prelude::*;
use modul_core::DeviceRes;

/// Initializes shadow GPU buffers, pass resources, and default config.
pub fn init_shadow(
    mut commands: Commands,
    device: Res<DeviceRes>,
    shadow_grid: Res<grid::ShadowGrid>,
) {
    let shadow_gpu = gpu::ShadowGpuBuffers::new(&device.0, &shadow_grid);
    let shadow_pass_res = pass::ShadowPassResources::new(
        &device.0, &shadow_gpu, 800, 600, 1,
    );

    commands.insert_resource(shadow_gpu);
    commands.insert_resource(shadow_pass_res);
    commands.insert_resource(pass::ShadowConfig::default());
    commands.insert_resource(pass::SunDirection::default());
    commands.insert_resource(pass::PreviousFrameData::default());
}
