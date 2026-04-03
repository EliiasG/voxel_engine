// WBOIT resolve: fullscreen triangle compositing transparent over opaque.

@group(0) @binding(0) var accum_tex: texture_2d<f32>;
@group(0) @binding(1) var revealage_tex: texture_2d<f32>;

struct ResolveOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_resolve(@builtin(vertex_index) vi: u32) -> ResolveOutput {
    // Fullscreen triangle
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: ResolveOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@fragment
fn fs_resolve(in: ResolveOutput) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(in.position.xy);
    let accum = textureLoad(accum_tex, coords, 0);
    let revealage = textureLoad(revealage_tex, coords, 0).r;

    // No transparent fragments here — keep opaque color
    if accum.a < 1e-4 {
        discard;
    }

    // WBOIT resolve: premultiplied-alpha weighted average
    let avg_color = accum.rgb / max(accum.a, 1e-4);

    // Composite: transparent over whatever is behind
    // revealage = product of (1-alpha) of all fragments... but with additive blend
    // we stored alpha directly and use ONE_MINUS_SRC_ALPHA on the revealage channel.
    // revealage here is the alpha sum, so (1 - revealage) = how much opaque shows through.
    return vec4<f32>(avg_color, 1.0 - revealage);
}
