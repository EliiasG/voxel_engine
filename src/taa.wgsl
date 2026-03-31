const CHUNK_SIZE: i32 = 32;

struct TaaVaryings {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct TaaOutput {
    @location(0) display: vec4<f32>,
    @location(1) history: vec4<f32>,
};

@vertex
fn vs_taa(@builtin(vertex_index) vi: u32) -> TaaVaryings {
    var out: TaaVaryings;
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

fn clip_aabb(box_min: vec3<f32>, box_max: vec3<f32>, history: vec3<f32>) -> vec3<f32> {
    let center = (box_min + box_max) * 0.5;
    let half_ext = (box_max - box_min) * 0.5 + vec3<f32>(0.0001);
    let d = history - center;
    let ratio = abs(d) / half_ext;
    let max_ratio = max(ratio.x, max(ratio.y, ratio.z));
    if max_ratio <= 1.0 {
        return history;
    }
    return center + d / max_ratio;
}

@fragment
fn fs_taa(in: TaaVaryings) -> TaaOutput {
    var out: TaaOutput;
    let pixel = vec2<i32>(in.position.xy);
    let tex_size = vec2<f32>(textureDimensions(scene_color));
    let tex_size_i = vec2<i32>(tex_size);

    // 1. Sample current scene at exact pixel
    let current_color = textureLoad(scene_color, pixel, 0).rgb;

    // 2. Neighborhood variance clipping
    var m1 = current_color;
    var m2 = current_color * current_color;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if dx == 0 && dy == 0 { continue; }
            let np = clamp(pixel + vec2<i32>(dx, dy), vec2<i32>(0), tex_size_i - 1);
            let neighbor = textureLoad(scene_color, np, 0).rgb;
            m1 += neighbor;
            m2 += neighbor * neighbor;
        }
    }
    let avg = m1 / 9.0;
    let variance = m2 / 9.0 - avg * avg;
    let sigma = sqrt(max(variance, vec3<f32>(0.0)));
    let color_min = avg - sigma * 1.5;
    let color_max = avg + sigma * 1.5;

    // 3. Reproject to previous frame
    let uv = in.uv;
    let depth = textureLoad(depth_tex, pixel, 0);

    var history_uv = uv;
    var reprojection_valid = false;

    if depth > 0.0 {
        let ndc = vec4<f32>(
            uv.x * 2.0 - 1.0,
            (1.0 - uv.y) * 2.0 - 1.0,
            depth,
            1.0,
        );
        let clip = camera.inv_view_proj * ndc;
        let cam_rel = clip.xyz / clip.w;

        let chunk_shift = vec3<f32>((camera.chunk_offset - camera.prev_chunk_offset) * CHUNK_SIZE);
        let prev_cam_rel = cam_rel + chunk_shift;

        let prev_clip = camera.prev_jittered_view_proj * vec4<f32>(prev_cam_rel, 1.0);
        let prev_ndc = prev_clip.xyz / prev_clip.w;
        history_uv = vec2<f32>(
            prev_ndc.x * 0.5 + 0.5,
            1.0 - (prev_ndc.y * 0.5 + 0.5),
        );

        reprojection_valid = all(history_uv >= vec2<f32>(0.0)) && all(history_uv <= vec2<f32>(1.0));
    }

    // 4. Sample history, clip, blend
    var result = current_color;
    if reprojection_valid {
        // Detect sub-pixel motion: if history maps to roughly the same pixel, use
        // textureLoad (exact value, no bilinear blur). Otherwise bilinear for smooth motion.
        let motion_pixels = length((history_uv - uv) * tex_size);
        var history_color: vec3<f32>;
        if motion_pixels < 1.0 {
            // Stationary or near-stationary: read exact history pixel (no blur)
            history_color = textureLoad(history_tex, pixel, 0).rgb;
        } else {
            // Moving: bilinear for smooth reprojection
            history_color = textureSample(history_tex, taa_sampler, history_uv).rgb;
        }

        let clipped = clip_aabb(color_min, color_max, history_color);
        let alpha = select(0.05, 0.2, motion_pixels >= 1.0);
        result = mix(clipped, current_color, alpha);
    }

    out.display = vec4<f32>(result, 1.0);
    out.history = vec4<f32>(result, 1.0);
    return out;
}
