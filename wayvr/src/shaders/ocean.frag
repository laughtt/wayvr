// Animated ocean background for WayVR's Work mode (omarchyvr).
//
// Drawn into an equirectangular swapchain (the same layer the skybox pictures use), so it follows the head
// position and stays world-locked in orientation. One fragment per texel of the 360 image:
//   direction -> ray hits the water plane -> sum of directional waves (statistical spectral synthesis of a
//   JONSWAP-like sea, Tessendorf 2001 / Pierson-Moskowitz 1964 / Hasselmann 1973) -> normal + foam (Jacobian of
//   the Gerstner displacement) -> Fresnel sky reflection + sun glitter (GGX) + body colour -> haze -> tone map.
// The sky is a closed-form single-scattering gradient (Rayleigh + Mie phase, Nishita 1993 style, no tables)
// with an optional cheap cloud layer. Clean-room: written from those papers, no code from other renderers.
#version 450

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

const int MAX_WAVES = 64;
const float PI = 3.14159265359;

layout(set = 0, binding = 0) uniform Ocean {
    // x: time (s, for the clouds), y: eye height above the water (m), z: horizon dip (tan), w: wave count
    vec4 a;
    // x: image width (px), y: exposure, z: choppiness, w: cloud octaves (0 = no clouds)
    vec4 b;
    // xyz: direction to the sun, w: sun strength
    vec4 sun;
    // xy: head position on the water plane (m), z: foam amount, w: cloud coverage
    vec4 c;
    // x: wave amplitude scale, y: haze distance (m), zw: cloud drift (per second)
    vec4 d;
    // per wave: kx, kz (rad/m), amplitude (m), phase (rad, time already folded in on the CPU)
    vec4 waves[MAX_WAVES];
} u;

// ---------------------------------------------------------------------------------------------------------------
// noise
float hash21(vec2 p) {
    // integer hash (PCG-like): no sin(), no precision trouble at large coordinates
    uvec2 v = uvec2(ivec2(floor(p))) * uvec2(1597334673u, 3812015801u);
    uint h = (v.x ^ v.y) * 1597334673u;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    return float(h) * (1.0 / 4294967295.0);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a0 = hash21(i);
    float b0 = hash21(i + vec2(1.0, 0.0));
    float c0 = hash21(i + vec2(0.0, 1.0));
    float d0 = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a0, b0, f.x), mix(c0, d0, f.x), f.y);
}

float fbm(vec2 p, int octaves) {
    float s = 0.0;
    float amp = 0.5;
    float norm = 0.0;
    for (int i = 0; i < octaves; ++i) {
        s += amp * vnoise(p);
        norm += amp;
        p = p * 2.03 + vec2(17.1, 9.3);
        amp *= 0.5;
    }
    return s / max(norm, 1e-4);
}

// ---------------------------------------------------------------------------------------------------------------
// sky
vec3 sunColour(float sunY) {
    // transmittance of the sun's light through the atmosphere: reddens when low
    float air = 1.0 / (max(sunY, 0.0) + 0.045);
    vec3 beta = vec3(0.0058, 0.0135, 0.0331) * 0.55 + 0.0021;
    return exp(-beta * air * 7.0);
}

float cloudDensity(vec2 p, int oct, float coverage) {
    float n = fbm(p, oct);
    return smoothstep(1.0 - coverage, 1.0 - coverage + 0.28, n);
}

vec3 skyColour(vec3 dir, int cloudOct) {
    vec3 S = u.sun.xyz;
    float y = max(dir.y, 0.0);
    float mu = clamp(dot(dir, S), -1.0, 1.0);

    // single scattering, closed form along the view ray
    vec3 betaR = vec3(0.0058, 0.0135, 0.0331);
    float betaM = 0.0030;
    vec3 beta = betaR * 0.55 + betaM;
    float L = 1.0 / (y * 0.85 + 0.06);
    float phR = 0.75 * (1.0 + mu * mu);
    float g = 0.76;
    float phM = (1.0 - g * g) / (4.0 * PI * pow(1.0 + g * g - 2.0 * g * mu, 1.5)) * 4.0 * PI * 0.25;
    vec3 sc = sunColour(S.y) * u.sun.w;
    vec3 inscatter = sc * (betaR * 0.55 * phR + betaM * phM) / beta * (1.0 - exp(-beta * L * 1.6));
    // the part of the sky not directly sunlit (earth shadow, multiple scattering): a soft blue floor
    vec3 amb = vec3(0.045, 0.085, 0.17) * (0.35 + 0.65 * smoothstep(-0.12, 0.35, S.y)) * (1.0 - 0.55 * exp(-y * 5.0));
    vec3 col = inscatter * 0.55 + amb * u.sun.w * 0.45;

    // sun disc
    float disc = smoothstep(0.99996, 0.99999, mu);
    col += disc * sc * 25.0 * step(0.0, dir.y);

    // clouds: one flat layer, lit from the sun side
    if (cloudOct > 0 && dir.y > 0.01) {
        float fade = smoothstep(0.01, 0.16, dir.y);
        vec2 p = dir.xz / (dir.y + 0.18) * 1.6 + u.d.zw * u.a.x;
        float cov = u.c.w;
        float dens = cloudDensity(p, cloudOct, cov);
        if (dens > 0.002) {
            float dsun = cloudDensity(p + S.xz * 0.12 / (S.y + 0.2) * 0.35, max(cloudOct - 1, 1), cov);
            float lit = clamp(0.55 + (dens - dsun) * 1.6, 0.0, 1.0);
            vec3 lightC = sc * 1.35 + vec3(0.04);
            vec3 shadeC = vec3(0.10, 0.12, 0.18) * u.sun.w * 0.55 + sc * 0.08;
            vec3 cc = mix(shadeC, lightC, lit);
            // silver lining near the sun
            cc += sc * pow(max(mu, 0.0), 8.0) * (1.0 - dens) * 0.9;
            col = mix(col, cc, dens * fade * 0.92);
        }
    }
    return col;
}

// ---------------------------------------------------------------------------------------------------------------
vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    // equirect2: texture centre looks down -Z, u grows to +X, v = 0 is straight up
    float az = (vUV.x - 0.5) * 2.0 * PI;
    float el = (0.5 - vUV.y) * PI;
    vec3 dir = vec3(sin(az) * cos(el), sin(el), -cos(az) * cos(el));

    int cloudOct = int(u.b.w);
    float H = u.a.y;
    float dip = u.a.z;

    // the horizon sits `dip` below eye level: the rays are turned up before they meet the (flat) water
    vec3 de = normalize(dir + vec3(0.0, dip, 0.0));

    vec3 col;
    if (de.y >= -0.0004) {
        col = skyColour(dir, cloudOct);
    } else {
        float t = H / -de.y;                       // distance along the ray to the water
        vec2 xz = de.xz * t + u.c.xy;              // world position on the water plane
        // one texel of the 360 image seen on the water: how big a wave must be to be drawn
        float texAng = 2.0 * PI / u.b.x;
        float fp = max(t * texAng / max(-de.y, 0.02), 1e-3);

        int n = int(u.a.w);
        float h = 0.0;
        vec2 slope = vec2(0.0);
        float lostVar = 0.0;
        float jxx = 0.0, jzz = 0.0, jxz = 0.0;
        float ch = u.b.z;
        for (int i = 0; i < MAX_WAVES; ++i) {
            if (i >= n) break;
            vec4 w = u.waves[i];
            float k = length(w.xy);
            float lam = 2.0 * PI / k;
            float amp = w.z * u.d.x;
            float vis = smoothstep(2.6 * fp, 7.0 * fp, lam);
            float sl = amp * k;
            lostVar += 0.5 * sl * sl * (1.0 - vis);
            if (vis <= 0.0) continue;
            float th = dot(w.xy, xz) + w.w;
            float cs = cos(th);
            float sn = sin(th);
            h += vis * amp * cs;
            slope -= vis * amp * sn * w.xy;
            float q = ch * vis * amp / k;          // Gerstner: x' = x - ch * a * khat * sin
            vec2 kh = w.xy;
            jxx -= q * kh.x * kh.x * cs;
            jzz -= q * kh.y * kh.y * cs;
            jxz -= q * kh.x * kh.y * cs;
        }
        vec3 N = normalize(vec3(-slope.x, 1.0, -slope.y));
        vec3 V = -de;
        float NoV = max(dot(N, V), 0.02);

        // sky reflection (rays that would dive under the sea are mirrored back up)
        vec3 R = reflect(de, N);
        R.y = abs(R.y) + 0.003;
        R = normalize(R);
        int reflOct = max(cloudOct - 2, 0);
        vec3 refl = skyColour(R, reflOct);

        // Fresnel; rough water reflects less at grazing angles
        float rough = clamp(sqrt(lostVar) * 1.6, 0.0, 1.0);
        float fr = pow(1.0 - NoV, 5.0);
        float F = 0.02 + (max(1.0 - rough * 0.9, 0.02) - 0.02) * fr;

        // water body: dark teal that brightens on lit crests (cheap subsurface)
        vec3 L = u.sun.xyz;
        vec3 sc = sunColour(L.y) * u.sun.w;
        float crest = clamp(h * 0.9 + 0.4 + length(slope) * 1.2, 0.0, 1.0);
        float sss = pow(clamp(dot(normalize(de.xz), normalize(L.xz)) * 0.5 + 0.5, 0.0, 1.0), 3.0) * crest;
        vec3 deep = vec3(0.020, 0.070, 0.095);
        vec3 shallow = vec3(0.020, 0.150, 0.130);
        vec3 amb = skyColour(vec3(0.0, 1.0, 0.0), 0) * 0.18;
        vec3 body = deep * (amb * 2.0 + sc * 0.25 * max(L.y, 0.0)) + shallow * sss * sc * 0.9;

        // sun glitter: GGX with the roughness of the waves too small to draw
        vec3 Hh = normalize(V + L);
        float NoH = max(dot(N, Hh), 0.0);
        float NoL = max(dot(N, L), 0.0);
        float alpha = clamp(sqrt(2.0 * lostVar + 0.0022 * 0.0022), 0.003, 0.6);
        float a2 = alpha * alpha;
        float dd = NoH * NoH * (a2 - 1.0) + 1.0;
        float D = a2 / (PI * dd * dd);
        float VoH = max(dot(V, Hh), 0.0);
        float Fs = 0.02 + 0.98 * pow(1.0 - VoH, 5.0);
        float Gv = NoV + sqrt(a2 + (1.0 - a2) * NoV * NoV);
        float Gl = NoL + sqrt(a2 + (1.0 - a2) * NoL * NoL);
        float spec = D * Fs / max(Gv * Gl, 1e-4);
        vec3 specCol = sc * spec * NoL * 2.2;

        col = body * (1.0 - F) + refl * F + specCol;

        // whitecaps where the surface folds (Jacobian of the Gerstner displacement < 1)
        float J = (1.0 + jxx) * (1.0 + jzz) - jxz * jxz;
        float fold = smoothstep(0.62, 0.20, J) * u.c.z;
        if (fold > 0.001) {
            float fn = fbm(xz * 1.7, 3) * 0.6 + fbm(xz * 6.0 + 3.0, 2) * 0.4;
            float foam = clamp(fold * (0.55 + 0.9 * fn) - 0.12, 0.0, 1.0);
            foam = smoothstep(0.0, 0.55, foam);
            vec3 foamC = (sc * max(L.y, 0.05) * 0.55 + amb * 1.2) * vec3(0.95, 0.97, 1.0);
            col = mix(col, foamC, foam * 0.85 * (1.0 - 0.6 * smoothstep(0.5, 4.0, fp)));
        }

        // aerial perspective: far water settles into the horizon colour
        float fog = 1.0 - exp(-t / u.d.y);
        vec3 horizonC = skyColour(normalize(vec3(dir.x, 0.0, dir.z) + vec3(0.0, 0.02, 0.0)), 0);
        col = mix(col, horizonC, fog);
    }

    col *= u.b.y;
    col = aces(col);
    // blue-noise-ish dither against 8-bit banding in the smooth sky
    float dn = hash21(gl_FragCoord.xy) - 0.5;
    col += dn / 255.0;
    outColor = vec4(col, 1.0);
}
