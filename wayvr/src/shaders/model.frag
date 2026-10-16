#version 450

// Controller model: base colour texture, one soft key light from above plus ambient (no PBR, like SteamVR's
// simple render model shading), opaque.
layout(set = 0, binding = 0) uniform sampler2D base_color;

layout(push_constant) uniform Pc {
    mat4 mvp;
    vec4 m0;
    vec4 m1;
    vec4 m2;
    vec4 tint;
} pc;

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

void main() {
    vec3 base = texture(base_color, in_uv).rgb * pc.tint.rgb;
    vec3 n = normalize(in_normal);
    float key = max(dot(n, normalize(vec3(0.3, 1.0, 0.45))), 0.0);
    float fill = max(dot(n, normalize(vec3(-0.4, -0.2, 0.6))), 0.0);
    vec3 c = base * (0.35 + 0.75 * key + 0.25 * fill);
    out_color = vec4(c, 1.0);
}
