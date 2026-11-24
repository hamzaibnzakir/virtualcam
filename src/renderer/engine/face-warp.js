// Draws the latest AI-swapped face onto the live picture, bent from where the face was when it was
// swapped to where it is now, using the face tracker's mesh. The AI may take a few hundred ms per
// face; this runs every camera frame, so the face follows your head at full speed.
//
// Sharpness: the AI's face is 128 px, stretched to ~300 px on screen. It's upscaled with a sharp
// (Catmull-Rom) filter, and the camera's own fine detail (skin texture, edges of the eyes and lips:
// brightness only, finer than the AI's face can hold) is added on top. Who you look like lives in the
// coarser shapes and colours, which stay the AI's.
import Delaunator from 'delaunator';
import { FACE_OVAL, INNER_LIPS, MESH_POINT_COUNT } from './face-points.js';

const RING_SCALE = 1.12; // a ring this far outside the face outline, where the face fades out
const POINTS = MESH_POINT_COUNT + FACE_OVAL.length;
const INNER = new Set(INNER_LIPS);
const FADE_MS = 120; // a new face blends in over this long

const VERT = `#version 300 es
in vec2 aPos;   // output pixels
in vec2 aUV;    // face picture, 0..1
in float aAlpha;
uniform vec2 uViewport;
out vec2 vUV;
out float vAlpha;
void main() {
  vUV = aUV;
  vAlpha = aAlpha;
  vec2 clip = aPos / uViewport * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
in float vAlpha;
uniform sampler2D uFace;
uniform sampler2D uCam;
uniform vec2 uViewport;
uniform float uOpacity;
uniform float uDetail;   // 0..1: how much of the camera's fine detail to add
uniform float uRadius;   // detail finer than this many screen pixels comes from the camera
uniform float uFaceSize; // face picture size in pixels
out vec4 color;

// Catmull-Rom upscaling with 9 bilinear taps (sharper than plain bilinear).
vec4 sharpSample(sampler2D tex, vec2 uv, float size) {
  vec2 p = uv * size - 0.5;
  vec2 f = fract(p);
  vec2 base = (floor(p) + 0.5) / size;
  vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  vec2 w3 = f * f * (-0.5 + 0.5 * f);
  vec2 w12 = w1 + w2;
  vec2 o12 = w2 / w12;
  vec2 t0 = base - 1.0 / size;
  vec2 t3 = base + 2.0 / size;
  vec2 t12 = base + o12 / size;
  vec4 r = texture(tex, vec2(t0.x, t0.y)) * w0.x * w0.y + texture(tex, vec2(t12.x, t0.y)) * w12.x * w0.y + texture(tex, vec2(t3.x, t0.y)) * w3.x * w0.y
         + texture(tex, vec2(t0.x, t12.y)) * w0.x * w12.y + texture(tex, vec2(t12.x, t12.y)) * w12.x * w12.y + texture(tex, vec2(t3.x, t12.y)) * w3.x * w12.y
         + texture(tex, vec2(t0.x, t3.y)) * w0.x * w3.y + texture(tex, vec2(t12.x, t3.y)) * w12.x * w3.y + texture(tex, vec2(t3.x, t3.y)) * w3.x * w3.y;
  return max(r, vec4(0.0));
}

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main() {
  vec2 inside = step(vec2(0.0), vUV) * step(vUV, vec2(1.0));
  vec4 face = clamp(sharpSample(uFace, vUV, uFaceSize), 0.0, 1.0);
  float a = face.a * vAlpha * uOpacity * inside.x * inside.y;
  vec3 rgb = face.rgb;
  if (uDetail > 0.0) {
    vec2 px = vec2(gl_FragCoord.x, uViewport.y - gl_FragCoord.y);
    vec2 uv = px / uViewport;
    vec2 d = uRadius / uViewport;
    float c = luma(texture(uCam, uv).rgb);
    // A soft blur of the camera around this pixel (centre + 8 around it).
    float b = c * 0.2
      + (luma(texture(uCam, uv + vec2(d.x, 0.0)).rgb) + luma(texture(uCam, uv - vec2(d.x, 0.0)).rgb)
       + luma(texture(uCam, uv + vec2(0.0, d.y)).rgb) + luma(texture(uCam, uv - vec2(0.0, d.y)).rgb)) * 0.125
      + (luma(texture(uCam, uv + d).rgb) + luma(texture(uCam, uv - d).rgb)
       + luma(texture(uCam, uv + vec2(d.x, -d.y)).rgb) + luma(texture(uCam, uv + vec2(-d.x, d.y)).rgb)) * 0.075;
    float detail = clamp(c - b, -0.15, 0.15);
    rgb = clamp(rgb + detail * uDetail, 0.0, 1.0);
  }
  color = vec4(rgb * a, a);
}`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

// Adds the fade-out ring around the face outline (pushed out from the outline's centre).
function withRing(points, out) {
  out.set(points.subarray(0, MESH_POINT_COUNT * 2));
  let cx = 0;
  let cy = 0;
  for (const i of FACE_OVAL) {
    cx += points[i * 2] / FACE_OVAL.length;
    cy += points[i * 2 + 1] / FACE_OVAL.length;
  }
  FACE_OVAL.forEach((i, k) => {
    out[(MESH_POINT_COUNT + k) * 2] = cx + (points[i * 2] - cx) * RING_SCALE;
    out[(MESH_POINT_COUNT + k) * 2 + 1] = cy + (points[i * 2 + 1] - cy) * RING_SCALE;
  });
  return out;
}

export class FaceWarp {
  constructor() {
    this.canvas = document.createElement('canvas');
    const gl = this.canvas.getContext('webgl2', { premultipliedAlpha: true, alpha: true, antialias: false });
    if (!gl) throw new Error('No WebGL 2');
    this.gl = gl;
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    this.program = program;
    this.loc = {
      pos: gl.getAttribLocation(program, 'aPos'),
      uv: gl.getAttribLocation(program, 'aUV'),
      alpha: gl.getAttribLocation(program, 'aAlpha'),
      viewport: gl.getUniformLocation(program, 'uViewport'),
      face: gl.getUniformLocation(program, 'uFace'),
      cam: gl.getUniformLocation(program, 'uCam'),
      opacity: gl.getUniformLocation(program, 'uOpacity'),
      detail: gl.getUniformLocation(program, 'uDetail'),
      radius: gl.getUniformLocation(program, 'uRadius'),
      faceSize: gl.getUniformLocation(program, 'uFaceSize'),
    };
    this.vao = gl.createVertexArray();
    this.buffer = gl.createBuffer();
    this.index = gl.createBuffer();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    const stride = 5 * 4;
    gl.enableVertexAttribArray(this.loc.pos);
    gl.vertexAttribPointer(this.loc.pos, 2, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(this.loc.uv);
    gl.vertexAttribPointer(this.loc.uv, 2, gl.FLOAT, false, stride, 8);
    gl.enableVertexAttribArray(this.loc.alpha);
    gl.vertexAttribPointer(this.loc.alpha, 1, gl.FLOAT, false, stride, 16);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.index);
    gl.bindVertexArray(null);
    this.vertices = new Float32Array(POINTS * 5);
    this.nowRing = new Float32Array(POINTS * 2);
    this.faces = []; // newest last: { texture, uv: Float32Array(POINTS*2), tris: Uint16Array, at, size }
    this.camTexture = gl.createTexture();
    this.detail = 0; // 0..1, set by the engine
  }

  // A new swapped face. rgba/size: its picture (alpha = soft edge). points: the tracked points of the
  // frame it was made from (output pixels). toFace: affine from output pixels to the face picture's pixels.
  addFace(rgba, size, points, toFace, now) {
    const gl = this.gl;
    const old = this.faces.length > 1 ? this.faces.shift() : null;
    const texture = old ? old.texture : gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, size, size, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const then = withRing(points, new Float32Array(POINTS * 2));
    const uv = new Float32Array(POINTS * 2);
    for (let i = 0; i < POINTS; i += 1) {
      const x = then[i * 2];
      const y = then[i * 2 + 1];
      uv[i * 2] = (toFace[0] * x + toFace[1] * y + toFace[2]) / size;
      uv[i * 2 + 1] = (toFace[3] * x + toFace[4] * y + toFace[5]) / size;
    }
    // Triangles over the face as it was (so none are folded there); the mouth opening is left out,
    // so your own mouth shows there, live.
    const all = new Delaunator(then).triangles;
    const tris = [];
    for (let t = 0; t < all.length; t += 3) {
      const a = all[t];
      const b = all[t + 1];
      const c = all[t + 2];
      if (INNER.has(a) && INNER.has(b) && INNER.has(c)) continue;
      // Triangles only between ring points lie outside the face.
      if (a >= MESH_POINT_COUNT && b >= MESH_POINT_COUNT && c >= MESH_POINT_COUNT) continue;
      tris.push(a, b, c);
    }
    this.faces.push({ texture, uv, tris: Uint16Array.from(tris), at: now, size });
  }

  clear() {
    this.faces = [];
  }

  get hasFace() {
    return this.faces.length > 0;
  }

  // Draws the face(s) at the current points into this.canvas (transparent elsewhere).
  // camera: the live picture (same size), for its fine detail; optional.
  draw(width, height, points, now, camera = null) {
    const gl = this.gl;
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!this.faces.length || !points) return;
    gl.useProgram(this.program);
    gl.uniform2f(this.loc.viewport, width, height);
    gl.uniform1i(this.loc.face, 0);
    gl.uniform1i(this.loc.cam, 1);
    const detail = camera && this.detail > 0 ? this.detail : 0;
    gl.uniform1f(this.loc.detail, detail);
    if (detail > 0) {
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.camTexture);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, camera);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      // Detail finer than one AI pixel, as it lands on screen, is taken from the camera.
      let x0 = Infinity;
      let x1 = -Infinity;
      for (const i of FACE_OVAL) {
        x0 = Math.min(x0, points[i * 2]);
        x1 = Math.max(x1, points[i * 2]);
      }
      const size = this.faces[this.faces.length - 1].size || 128;
      gl.uniform1f(this.loc.radius, Math.max(1, Math.min(6, (x1 - x0) / (size * 0.62))));
    }
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(this.vao);
    const pos = withRing(points, this.nowRing);
    const newest = this.faces[this.faces.length - 1];
    const fade = Math.min(1, (now - newest.at) / FADE_MS);
    this.faces.forEach((face, k) => {
      const opacity = k === this.faces.length - 1 && this.faces.length > 1 ? fade : 1;
      if (opacity <= 0) return;
      const v = this.vertices;
      for (let i = 0; i < POINTS; i += 1) {
        v[i * 5] = pos[i * 2];
        v[i * 5 + 1] = pos[i * 2 + 1];
        v[i * 5 + 2] = face.uv[i * 2];
        v[i * 5 + 3] = face.uv[i * 2 + 1];
        v[i * 5 + 4] = i < MESH_POINT_COUNT ? 1 : 0;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
      gl.bufferData(gl.ARRAY_BUFFER, v, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.index);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, face.tris, gl.DYNAMIC_DRAW);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, face.texture);
      gl.uniform1f(this.loc.opacity, opacity);
      gl.uniform1f(this.loc.faceSize, face.size || 128);
      gl.drawElements(gl.TRIANGLES, face.tris.length, gl.UNSIGNED_SHORT, 0);
    });
    gl.bindVertexArray(null);
    // Once the newest face is fully in, the older one isn't needed.
    if (fade >= 1 && this.faces.length > 1) {
      const old = this.faces.shift();
      gl.deleteTexture(old.texture);
    }
  }
}
