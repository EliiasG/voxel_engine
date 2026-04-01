// Shared normal-output fragment shader. Only requires VertexOutput with a normal field.

@fragment
fn fs_normal(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.normal * 0.5 + 0.5, 1.0);
}
