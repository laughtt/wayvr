#version 450

// Controller model (backend/openxr/controller_render.rs): one eye, mesh in grip space -> clip space.
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(push_constant) uniform Pc {
    mat4 mvp;
    vec4 m0; // model rotation columns (world space normals)
    vec4 m1;
    vec4 m2;
    vec4 tint;
} pc;

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec2 out_uv;

void main() {
    gl_Position = pc.mvp * vec4(in_pos, 1.0);
    out_normal = mat3(pc.m0.xyz, pc.m1.xyz, pc.m2.xyz) * in_normal;
    out_uv = in_uv;
}
