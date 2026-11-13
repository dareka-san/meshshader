(() => {
  const canvas = document.querySelector('#drawing');
  const status = document.querySelector('#status');
  const motionToggle = document.querySelector('#motion-toggle');
  const floorDepthScale = 1 / Math.sqrt(3);
  const heightProjection = Math.sqrt(1 - floorDepthScale * floorDepthScale);
  const waveHeightScale = 2;
  const simulationStep = 1 / 120;
  const pointerSmoothing = 1 - Math.exp(-18 * simulationStep);
  const windDelay = 12;
  const windInterval = 24;
  const windDuration = 10;
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let motionEnabled = !reducedMotion.matches;

  function readDensity(value) {
    const number = Number(value);
    return Number.isInteger(number) && number >= 1 && number <= 64 ? number : null;
  }

  const requestedDensity = readDensity(new URLSearchParams(location.search).get('n'));
  const defaultDensity = readDensity(canvas.dataset.n) ?? 24;

  function meshDensity(width) {
    return requestedDensity ?? Math.min(defaultDensity, Math.max(1, Math.floor(width / 10)));
  }

  // RGBA8 uses two bytes each for signed height and velocity.
  const packedWave = `
    precision highp float;

    float decodeWaveValue(vec2 bytes) {
      return (dot(bytes, vec2(65280.0, 255.0)) - 32768.0) / 200.0;
    }

    vec2 encodeWaveValue(float value) {
      float code = clamp(floor(value * 200.0 + 32768.5), 0.0, 65535.0);
      float highByte = floor(code / 256.0);
      return vec2(highByte, code - highByte * 256.0) / 255.0;
    }
  `;

  const waveTexture = `
    vec2 readWave(highp sampler2D wave, ivec2 pixel, ivec2 lastPixel) {
      vec4 sampleValue = texelFetch(wave, clamp(pixel, ivec2(0), lastPixel), 0);
      #ifdef PACKED_WAVE
        return vec2(decodeWaveValue(sampleValue.rg), decodeWaveValue(sampleValue.ba));
      #else
        return sampleValue.rg;
      #endif
    }
  `;

  // 1. 波動シミュレーション用シェーダー（FBOピンポン）
  // R: 現在の高さ, G: 速度
  const simVertex = `#version 300 es
    layout(location = 0) in vec2 a_pos;
    out vec2 v_uv;
    void main() {
      v_uv = a_pos * 0.5 + 0.5;
      gl_Position = vec4(a_pos, 0.0, 1.0);
    }
  `;

  const simFragment = `#version 300 es
    precision highp float;
    in vec2 v_uv;
    out vec4 outColor;

    uniform highp sampler2D u_prev;
    uniform vec2 u_floorSize;
    uniform vec2 u_pointerPos;     // 現在のフロア座標
    uniform vec2 u_prevPointerPos; // 前フレームのフロア座標
    uniform float u_pointerSpeed;
    uniform int u_grabMode;        // 0: 通常, 1: つまみ中, 2: リリース
    uniform float u_grabHeight;    // つまみ上げている高さ
    uniform float u_grabRadius;
    uniform float u_damping;
    ${waveTexture}

    // 線分と点の最短距離（滑らかな航跡を作るため）
    float distToSegment(vec2 p, vec2 a, vec2 b) {
      vec2 pa = p - a, ba = b - a;
      float h = clamp(dot(pa, ba) / max(dot(ba, ba), 0.0001), 0.0, 1.0);
      return length(pa - ba * h);
    }

    void main() {
      ivec2 pixel = ivec2(gl_FragCoord.xy);
      ivec2 lastPixel = textureSize(u_prev, 0) - 1;
      vec2 data = readWave(u_prev, pixel, lastPixel);
      float h = data.r;
      float v = data.g;

      // 上下左右のサンプリング（境界は反射条件）
      ivec2 dx = ivec2(1, 0);
      ivec2 dy = ivec2(0, 1);
      float hL = readWave(u_prev, pixel - dx, lastPixel).x;
      float hR = readWave(u_prev, pixel + dx, lastPixel).x;
      float hD = readWave(u_prev, pixel - dy, lastPixel).x;
      float hU = readWave(u_prev, pixel + dy, lastPixel).x;

      // 2次元波動方程式（ラプラシアンによる加速度）
      float laplacian = (hL + hR + hD + hU) - 4.0 * h;
      v += laplacian * 0.18 - h * 0.0008;
      v *= u_damping;

      // 現在のピクセルのフロア空間上の座標
      vec2 floorPos = v_uv * u_floorSize;

      if (u_grabMode != 0) {
        vec2 offset = (floorPos - u_pointerPos) / u_grabRadius;
        float influence = exp(-dot(offset, offset));
        float targetH = u_grabHeight * influence;
        if (u_grabMode == 1) {
          h = mix(h, targetH, 0.66 * influence);
          v *= 1.0 - 0.93 * influence;
        } else {
          h = mix(h, targetH, 0.5 * influence);
          v -= targetH * 0.035;
        }
      } else if (u_pointerSpeed > 4.0) {
        // 【ホバー移動中】前フレームからの線分に沿って連続的に波紋を与える
        float dist = distToSegment(floorPos, u_prevPointerPos, u_pointerPos);
        float falloff = dist * dist / (70.0 * 70.0);
        float influence = exp(-falloff) - 0.25 * exp(-falloff * 0.25);
        v -= (u_pointerSpeed - 4.0) * influence * 0.0042;
      }

      // 発散防止のクランプ
      h = clamp(h + v, -150.0, 150.0);
      #ifdef PACKED_WAVE
        outColor = vec4(encodeWaveValue(h), encodeWaveValue(v));
      #else
        outColor = vec4(h, v, 0.0, 1.0);
      #endif
    }
  `;

  // 2. 描画用共通シェーダー
  const waterSurface = `
    uniform vec2 u_size;
    uniform float u_edgeLength;
    uniform float u_motion;
    uniform vec3 u_wind;
    uniform highp sampler2D u_waveTex;
    uniform vec2 u_waveTexelSize;
    uniform vec4 u_grab;      // xy: 交点の元の位置, zw: 引っ張り量
    uniform vec2 u_grabShape; // x: 影響半径, y: つまみの強さ
    ${waveTexture}

    const float waveWidthScale = 2.0;
    const float waveHeightScale = ${waveHeightScale.toFixed(1)};
    const vec2 screenToFloor = vec2(1.0, 1.0 / ${floorDepthScale});

    vec2 sampleWave(float distanceToFront, float spread, float frequency) {
      float front = distanceToFront / spread;
      if (abs(front) >= 1.0) return vec2(0.0);

      float falloff = 1.0 - front * front;
      float envelope = falloff * falloff;
      float envelopeSlope = -4.0 * front * falloff / spread;
      float phase = distanceToFront * frequency;
      float crest = cos(phase);
      float slope = crest * envelopeSlope - sin(phase) * frequency * envelope;
      return vec2(crest * envelope, slope);
    }

    vec2 sampleSurface(vec2 position, vec2 direction) {
      vec2 floorSize = u_size * screenToFloor;
      vec2 floorPosition = position * screenToFloor;
      vec2 floorDirection = direction * screenToFloor;

      // 風の計算（既存の軽量な解析波を維持）
      vec2 wind = vec2(0.0);
      if (u_wind.z >= 0.0) {
        vec2 windDirection = u_wind.xy;
        float span = dot(floorSize, abs(windDirection));
        float spread = span * 0.34 * waveWidthScale;
        float progress = u_wind.z / ${windDuration.toFixed(1)};
        float travel = progress * (1.25 - 0.25 * progress);
        float center = mix(-spread, span + spread, travel);
        float alongWind = dot(floorPosition - floorSize * 0.5, windDirection) + span * 0.5;
        wind = sampleWave(alongWind - center, spread, 19.0 / (span * waveWidthScale));
        wind.y *= dot(floorDirection, windDirection);
      }

      // GPGPUテクスチャから高さと勾配（Slope）を取得
      vec2 uv = clamp(floorPosition / floorSize, 0.0, 1.0);
      vec2 cellFloorSize = floorSize * u_waveTexelSize;
      #ifdef PACKED_WAVE
        vec2 samplePixel = uv / u_waveTexelSize - 0.5;
        ivec2 pixel = ivec2(floor(samplePixel));
        ivec2 lastPixel = textureSize(u_waveTex, 0) - 1;
        vec2 weight = fract(samplePixel);
        float h00 = readWave(u_waveTex, pixel, lastPixel).x;
        float h10 = readWave(u_waveTex, pixel + ivec2(1, 0), lastPixel).x;
        float h01 = readWave(u_waveTex, pixel + ivec2(0, 1), lastPixel).x;
        float h11 = readWave(u_waveTex, pixel + ivec2(1, 1), lastPixel).x;
        float hC = mix(mix(h00, h10, weight.x), mix(h01, h11, weight.x), weight.y);
        vec2 grad = vec2(mix(h10 - h00, h11 - h01, weight.y),
                         mix(h01 - h00, h11 - h10, weight.x)) / cellFloorSize;
      #else
        float hC = texture(u_waveTex, uv).r;
        float hR = texture(u_waveTex, uv + vec2(u_waveTexelSize.x, 0.0)).r;
        float hU = texture(u_waveTex, uv + vec2(0.0, u_waveTexelSize.y)).r;
        vec2 grad = vec2(hR - hC, hU - hC) / cellFloorSize;
      #endif
      float waveSlope = dot(grad, floorDirection);

      float windAmplitude = min(u_edgeLength * 0.14, 7.0);
      vec2 heightField = wind * windAmplitude + vec2(hC, waveSlope);

      float maxHeight = clamp(u_edgeLength * 0.43, 10.0, 20.0);
      float height = tanh(heightField.x / maxHeight);
      float motion = u_motion * waveHeightScale * ${heightProjection};
      return vec2(height * maxHeight, heightField.y * (1.0 - height * height)) * motion;
    }

    vec4 deformSurface(vec2 restPosition, vec2 direction) {
      vec2 surface = sampleSurface(restPosition, direction);
      vec2 position = restPosition - vec2(0.0, surface.x);
      vec2 tangent = direction - vec2(0.0, surface.y);

      if (u_grabShape.y > 0.0) {
        vec2 offset = (restPosition - u_grab.xy) * screenToFloor / u_grabShape.x;
        float weight = exp(-dot(offset, offset)) * u_grabShape.y;
        float weightSlope = -2.0 * weight * dot(offset, direction * screenToFloor) / u_grabShape.x;
        vec2 pulledPosition = restPosition + u_grab.zw;
        tangent = mix(tangent, direction, weight) + (pulledPosition - position) * weightSlope;
        position = mix(position, pulledPosition, weight);
      }

      return vec4(position, tangent);
    }

    vec4 toClipPosition(vec2 position) {
      vec2 clip = position / u_size * 2.0 - 1.0;
      return vec4(clip.x, -clip.y, 0.0, 1.0);
    }
  `;

  const lineVertex = `#version 300 es
    layout(location = 0) in vec2 a_strip;
    layout(location = 1) in vec4 a_edge;
    out float v_side;
    ${waterSurface}

    void main() {
      vec2 restPosition = mix(a_edge.xy, a_edge.zw, a_strip.x);
      vec2 direction = normalize(a_edge.zw - a_edge.xy);
      vec4 surface = deformSurface(restPosition, direction);

      vec2 tangent = surface.zw;
      vec2 position = surface.xy + normalize(vec2(-tangent.y, tangent.x)) * a_strip.y;
      gl_Position = toClipPosition(position);
      v_side = a_strip.y;
    }
  `;

  const lineFragment = `#version 300 es
    precision mediump float;
    in float v_side;
    out vec4 color;
    void main() {
      float coverage = 1.0 - smoothstep(0.06, 1.0, abs(v_side));
      color = vec4(vec3(0.55), coverage * 0.54);
    }
  `;

  const pointVertex = `#version 300 es
    layout(location = 0) in vec2 a_position;
    uniform float u_pointSize;
    ${waterSurface}

    void main() {
      gl_Position = toClipPosition(deformSurface(a_position, vec2(0.0)).xy);
      float focus = u_grabShape.y * (1.0 - smoothstep(0.0, 1.0, length(a_position - u_grab.xy)));
      gl_PointSize = u_pointSize * (1.0 + 0.45 * focus);
    }
  `;

  const pointFragment = `#version 300 es
    precision mediump float;
    uniform float u_radius;
    out vec4 color;
    void main() {
      float distanceToCenter = length(gl_PointCoord - vec2(0.5)) * (u_radius + 0.6) * 2.0;
      float coverage = 1.0 - smoothstep(u_radius - 0.4, u_radius + 0.6, distanceToCenter);
      color = vec4(vec3(0.59), coverage * 0.76);
    }
  `;

  function createProgram(gl, vertexSource, fragmentSource, uniformNames) {
    const program = gl.createProgram();
    for (const [type, source] of [[gl.VERTEX_SHADER, vertexSource], [gl.FRAGMENT_SHADER, fragmentSource]]) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      gl.attachShader(program, shader);
      gl.deleteShader(shader);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    const uniforms = Object.fromEntries(uniformNames.map((name) => [name, gl.getUniformLocation(program, name)]));
    return { program, uniforms };
  }

  function createAttribute(gl, location, size, divisor = 0) {
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(location, divisor);
    return buffer;
  }

  function createMesh(width, height) {
    const density = meshDensity(width);
    const columnWidth = width / density;
    const halfWidth = columnWidth / 2;
    const rowHeight = halfWidth * floorDepthScale;
    const rowLimit = Math.ceil(height / (2 * rowHeight)) + 1;
    const edges = [];
    const points = [];
    const strip = [];
    // 波が急峻になるため分割数を少し増やして曲線を滑らかに保つ
    const segments = Math.max(4, Math.min(24, Math.ceil(halfWidth / 1.5)));

    for (let segment = 0; segment <= segments; segment++) {
      const along = segment / segments;
      strip.push(along, -1, along, 1);
    }

    for (let row = -rowLimit; row <= rowLimit; row++) {
      const y = height / 2 + row * rowHeight;
      const nextY = height / 2 + (row + 1) * rowHeight;
      const staggered = row % 2 !== 0;
      const stagger = staggered ? halfWidth : 0;
      const nextStagger = halfWidth - stagger;

      for (let column = -1; column <= density; column++) {
        const x = column * columnWidth + stagger;
        points.push(x, y);
        if (row < rowLimit) {
          const leftColumn = staggered ? column : column - 1;
          const leftX = leftColumn * columnWidth + nextStagger;
          const rightX = (leftColumn + 1) * columnWidth + nextStagger;
          edges.push(x, y, leftX, nextY, x, y, rightX, nextY);
        }
      }
}
return {
edges: new Float32Array(edges),
points: new Float32Array(points),
strip: new Float32Array(strip),
edgeLength: Math.hypot(halfWidth, rowHeight),
pointRadius: Math.max(1, Math.min(2.45, halfWidth * 0.065)),
};
}
  function createFBO(gl, width, height, packed) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    const format = packed ? gl.RGBA8 : gl.RGBA16F;
    const type = packed ? gl.UNSIGNED_BYTE : gl.HALF_FLOAT;
    gl.texImage2D(gl.TEXTURE_2D, 0, format, width, height, 0, gl.RGBA, type, null);
    const filter = packed ? gl.NEAREST : gl.LINEAR;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const complete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    if (complete) {
      const zero = packed ? 128 / 255 : 0;
      if (packed) gl.disable(gl.DITHER);
      gl.clearColor(zero, 0, zero, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      if (packed) gl.enable(gl.DITHER);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (complete) return { tex, fbo };
    gl.deleteFramebuffer(fbo);
    gl.deleteTexture(tex);
    return null;
  }

  function createWaveBuffers(gl, width, height) {
    const supportsFloat = gl.getExtension('EXT_color_buffer_float')
      || gl.getExtension('EXT_color_buffer_half_float');
    for (const packed of supportsFloat ? [false, true] : [true]) {
      const buffers = [];
      for (let index = 0; index < 2; index++) {
        const buffer = createFBO(gl, width, height, packed);
        if (!buffer) break;
        buffers.push(buffer);
      }
      if (buffers.length === 2) return { buffers, packed };
      for (const { tex, fbo } of buffers) {
        gl.deleteFramebuffer(fbo);
        gl.deleteTexture(tex);
      }
    }
    throw new Error('波の計算に必要なフレームバッファーを作成できませんでした。');
  }

  function waveShader(source, packed) {
    return packed ? source.replace('#version 300 es', '#version 300 es\n#define PACKED_WAVE\n' + packedWave) : source;
  }

function createRenderer(gl) {
const simW = 256;
const simH = 256;
const { buffers: fbos, packed } = createWaveBuffers(gl, simW, simH);
const simUniforms = [
'u_prev', 'u_floorSize', 'u_pointerPos',
'u_prevPointerPos', 'u_pointerSpeed', 'u_grabMode', 'u_grabHeight', 'u_grabRadius', 'u_damping'
];
const sim = createProgram(gl, simVertex, waveShader(simFragment, packed), simUniforms);
const surfaceUniforms = ['u_size', 'u_edgeLength', 'u_motion', 'u_wind', 'u_waveTex', 'u_waveTexelSize', 'u_grab', 'u_grabShape'];
const lines = createProgram(gl, waveShader(lineVertex, packed), lineFragment, surfaceUniforms);
const points = createProgram(gl, waveShader(pointVertex, packed), pointFragment, [...surfaceUniforms, 'u_pointSize', 'u_radius']);
// シミュレーション用フルスクリーンクワッド
const quadVAO = gl.createVertexArray();
gl.bindVertexArray(quadVAO);
const quadBuf = createAttribute(gl, 0, 2);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
const lineGeometry = gl.createVertexArray();
gl.bindVertexArray(lineGeometry);
const stripBuffer = createAttribute(gl, 0, 2);
const edgeBuffer = createAttribute(gl, 1, 4, 1);
const pointGeometry = gl.createVertexArray();
gl.bindVertexArray(pointGeometry);
const pointBuffer = createAttribute(gl, 0, 2);
let ping = 0;
let width = 1, height = 1, floorWidth = 1, floorHeight = 1;
let stripCount, edgeCount, pointCount;
const startedAt = performance.now() / 1000;
let lastWindCycle = -1, windX = 0, windY = 0;
function resize() {
width = Math.max(1, canvas.clientWidth);
height = Math.max(1, canvas.clientHeight);
floorWidth = width;
floorHeight = height / floorDepthScale;
const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
canvas.width = Math.max(1, Math.round(width * pixelRatio));
canvas.height = Math.max(1, Math.round(height * pixelRatio));
const mesh = createMesh(width, height);
stripCount = mesh.strip.length / 2;
edgeCount = mesh.edges.length / 4;
pointCount = mesh.points.length / 2;
gl.bindBuffer(gl.ARRAY_BUFFER, stripBuffer);
gl.bufferData(gl.ARRAY_BUFFER, mesh.strip, gl.STATIC_DRAW);
gl.bindBuffer(gl.ARRAY_BUFFER, edgeBuffer);
gl.bufferData(gl.ARRAY_BUFFER, mesh.edges, gl.STATIC_DRAW);
gl.bindBuffer(gl.ARRAY_BUFFER, pointBuffer);
gl.bufferData(gl.ARRAY_BUFFER, mesh.points, gl.STATIC_DRAW);
for (const { program, uniforms } of [lines, points]) {
gl.useProgram(program);
gl.uniform2f(uniforms.u_size, width, height);
gl.uniform1f(uniforms.u_edgeLength, mesh.edgeLength);
gl.uniform2f(uniforms.u_waveTexelSize, 1 / simW, 1 / simH);
}
gl.uniform1f(points.uniforms.u_pointSize, (mesh.pointRadius + 0.6) * 2 * pixelRatio);
gl.uniform1f(points.uniforms.u_radius, mesh.pointRadius);
}
function stepSimulation(pointerState) {
const next = 1 - ping;
gl.bindFramebuffer(gl.FRAMEBUFFER, fbos[next].fbo);
gl.viewport(0, 0, simW, simH);
gl.disable(gl.BLEND);
if (packed) gl.disable(gl.DITHER);
gl.useProgram(sim.program);
gl.activeTexture(gl.TEXTURE0);
gl.bindTexture(gl.TEXTURE_2D, fbos[ping].tex);
gl.uniform1i(sim.uniforms.u_prev, 0);
gl.uniform2f(sim.uniforms.u_floorSize, floorWidth, floorHeight);
const grab = pointerState.grab;
const grabMode = grab?.id != null ? 1 : grab?.releasePending ? 2 : 0;
const sourceX = grabMode ? grab.x : pointerState.smoothX;
const sourceY = grabMode ? grab.y : pointerState.smoothY;
gl.uniform2f(sim.uniforms.u_pointerPos, sourceX, sourceY / floorDepthScale);
gl.uniform2f(sim.uniforms.u_prevPointerPos, pointerState.prevSmoothX, pointerState.prevSmoothY / floorDepthScale);
gl.uniform1f(sim.uniforms.u_pointerSpeed, pointerState.speed);
gl.uniform1i(sim.uniforms.u_grabMode, grabMode);
gl.uniform1f(sim.uniforms.u_grabHeight, grabMode ? grab.height : 0);
gl.uniform1f(sim.uniforms.u_grabRadius, grabMode ? Math.min(grab.radius * 0.75, 155) : 1);
gl.uniform1f(sim.uniforms.u_damping, 0.986); // 波の余韻の長さ（0.98〜0.992で調整可）
gl.bindVertexArray(quadVAO);
gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
if (packed) gl.enable(gl.DITHER);
gl.bindFramebuffer(gl.FRAMEBUFFER, null);
ping = next;
if (grabMode === 2) grab.releasePending = false;
}
function draw(time, pointerState, motion = 1) {
const elapsed = time - startedAt - windDelay;
const windCycle = Math.floor(elapsed / windInterval);
const windAge = elapsed - windCycle * windInterval;
const windActive = motion > 0 && elapsed >= 0 && windAge < windDuration;
if (windActive && windCycle !== lastWindCycle) {
const angle = Math.random() * 2 * Math.PI;
windX = Math.cos(angle);
windY = Math.sin(angle);
lastWindCycle = windCycle;
}
gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
gl.clearColor(215 / 255, 215 / 255, 215 / 255, 1);
gl.clear(gl.COLOR_BUFFER_BIT);
gl.enable(gl.BLEND);
gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
const activeWindAge = windActive ? windAge : -1;
const grab = pointerState.grab;
gl.activeTexture(gl.TEXTURE0);
gl.bindTexture(gl.TEXTURE_2D, fbos[ping].tex);
for (const prog of [lines, points]) {
gl.useProgram(prog.program);
gl.uniform1f(prog.uniforms.u_motion, motion);
gl.uniform3f(prog.uniforms.u_wind, windX, windY, activeWindAge);
gl.uniform1i(prog.uniforms.u_waveTex, 0);
gl.uniform4f(prog.uniforms.u_grab, grab?.x ?? 0, grab?.y ?? 0, grab?.offsetX ?? 0, grab?.offsetY ?? 0);
gl.uniform2f(prog.uniforms.u_grabShape, grab?.radius ?? 1, grab?.weight ?? 0);
}
gl.useProgram(lines.program);
gl.bindVertexArray(lineGeometry);
gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, stripCount, edgeCount);
gl.useProgram(points.program);
gl.bindVertexArray(pointGeometry);
gl.drawArrays(gl.POINTS, 0, pointCount);
}
return { resize, stepSimulation, draw };
}
  const pointerState = {
    targetX: 0, targetY: 0,
    smoothX: 0, smoothY: 0,
    prevSmoothX: 0, prevSmoothY: 0,
    speed: 0,
    hovering: false,
    grab: null,
  };

  function pointerCoords(event) {
    const rect = canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  function resetHover(x, y, hovering) {
    pointerState.targetX = pointerState.smoothX = pointerState.prevSmoothX = x;
    pointerState.targetY = pointerState.smoothY = pointerState.prevSmoothY = y;
    pointerState.hovering = hovering;
    pointerState.speed = 0;
  }

  function nearestJunction(x, y) {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const columnWidth = width / meshDensity(width);
    const rowHeight = columnWidth * 0.5 * floorDepthScale;
    const centerRow = Math.round((y - height / 2) / rowHeight);
    let nearest = { x: width / 2, y: height / 2 };
    let minDistance = Infinity;

    for (let row = centerRow - 1; row <= centerRow + 1; row++) {
      const rowY = height / 2 + row * rowHeight;
      if (rowY < 0 || rowY > height) continue;
      const stagger = row % 2 === 0 ? 0 : columnWidth / 2;
      const lastColumn = Math.floor((width - stagger) / columnWidth);
      const column = Math.max(0, Math.min(lastColumn, Math.round((x - stagger) / columnWidth)));
      const nodeX = column * columnWidth + stagger;
      const distance = Math.hypot(nodeX - x, rowY - y);
      if (distance < minDistance) {
        nearest = { x: nodeX, y: rowY };
        minDistance = distance;
      }
    }

    return nearest;
  }

  function pullJunction(grab, x, y) {
    const offsetX = x - grab.startX;
    const offsetY = y - grab.startY - 10;
    const distance = Math.hypot(offsetX, offsetY / floorDepthScale);
    const pull = Math.min(1, 225 / Math.max(1, distance));
    grab.offsetX = offsetX * pull;
    grab.offsetY = offsetY * pull;
    grab.radius = grab.baseRadius + distance * pull;
    grab.height = 0.55 * (-grab.offsetY + Math.abs(grab.offsetX) * 0.2) / (heightProjection * waveHeightScale);
  }

  canvas.addEventListener('pointerenter', (event) => {
    if (!renderer || !motionEnabled || !event.isPrimary || pointerState.grab?.id != null) return;
    const { x, y } = pointerCoords(event);
    resetHover(x, y, true);
  });

  canvas.addEventListener('pointerdown', (event) => {
    if (!renderer || !motionEnabled || !event.isPrimary || pointerState.grab?.id != null) return;
    if (event.pointerType === 'mouse' && event.button !== 0) return;
    const { x, y } = pointerCoords(event);
    canvas.setPointerCapture(event.pointerId);
    resetHover(x, y, false);
    const grab = {
      ...nearestJunction(x, y),
      id: event.pointerId, startX: x, startY: y,
      baseRadius: Math.max(65, Math.min(135, canvas.clientWidth / meshDensity(canvas.clientWidth) * 1.35)),
      weight: 1, releasePending: false,
    };
    pullJunction(grab, x, y);
    pointerState.grab = grab;
    canvas.style.cursor = 'grabbing';
  });

  canvas.addEventListener('pointermove', (event) => {
    if (!renderer || !motionEnabled || !event.isPrimary) return;
    const { x, y } = pointerCoords(event);
    const grab = pointerState.grab;
    if (grab?.id != null) {
      if (grab.id === event.pointerId) pullJunction(grab, x, y);
      return;
    }
    if (event.pointerType === 'touch') return;
    if (!pointerState.hovering) resetHover(x, y, true);
    pointerState.targetX = x;
    pointerState.targetY = y;
  });

  canvas.addEventListener('pointerleave', (event) => {
    if (event.isPrimary && pointerState.grab?.id == null) pointerState.hovering = false;
  });

  function releasePointer(event) {
    const grab = pointerState.grab;
    if (grab?.id !== event.pointerId) return;
    const { x, y } = pointerCoords(event);
    const released = event.type === 'pointerup';
    if (released) pullJunction(grab, x, y);
    grab.id = null;
    grab.releasePending = released;
    resetHover(x, y, released && event.pointerType !== 'touch'
      && x >= 0 && x <= canvas.clientWidth && y >= 0 && y <= canvas.clientHeight);
    canvas.style.cursor = '';
  }

  canvas.addEventListener('pointerup', releasePointer);
  canvas.addEventListener('pointercancel', releasePointer);
  canvas.addEventListener('lostpointercapture', releasePointer);

  function clearPointer() {
    const id = pointerState.grab?.id;
    pointerState.grab = null;
    pointerState.hovering = false;
    canvas.style.cursor = '';
    if (id != null && canvas.hasPointerCapture(id)) canvas.releasePointerCapture(id);
  }

  function stepPointer() {
    pointerState.speed = 0;
    if (!pointerState.hovering) return;
    pointerState.prevSmoothX = pointerState.smoothX;
    pointerState.prevSmoothY = pointerState.smoothY;
    pointerState.smoothX += (pointerState.targetX - pointerState.smoothX) * pointerSmoothing;
    pointerState.smoothY += (pointerState.targetY - pointerState.smoothY) * pointerSmoothing;
    const distance = Math.hypot(
      pointerState.smoothX - pointerState.prevSmoothX,
      (pointerState.smoothY - pointerState.prevSmoothY) / floorDepthScale,
    );
    pointerState.speed = Math.min(distance / simulationStep, 1000);
  }

  let renderer;
  let frame;
  let previousTime;
  let simulationTime = 0;

  function animate(timestamp) {
    const time = timestamp / 1000;
    const elapsed = Math.min(0.05, time - (previousTime ?? time));
    previousTime = time;
    simulationTime += elapsed;
    while (simulationTime >= simulationStep) {
      stepPointer();
      renderer.stepSimulation(pointerState);
      simulationTime -= simulationStep;
    }

    const grab = pointerState.grab;
    if (grab && grab.id == null) {
      grab.weight *= Math.exp(-8.5 * elapsed);
      if (grab.weight < 0.005 && !grab.releasePending) pointerState.grab = null;
    }

    renderer.draw(time, pointerState);
    frame = requestAnimationFrame(animate);
  }

  function updateMotion() {
    cancelAnimationFrame(frame);
    previousTime = undefined;
    simulationTime = 0;
    clearPointer();
    canvas.toggleAttribute('data-static', !renderer || !motionEnabled);
    motionToggle.hidden = !renderer || !reducedMotion.matches;
    motionToggle.textContent = motionEnabled ? 'ネットの動きを止める' : 'ネットの動きを有効にする';

    if (!renderer) return;
    if (motionEnabled) {
      frame = requestAnimationFrame(animate);
    } else {
      renderer.draw(performance.now() / 1000, pointerState, 0);
    }
  }

  reducedMotion.addEventListener('change', (event) => {
    motionEnabled = !event.matches;
    updateMotion();
  });

  motionToggle.addEventListener('click', () => {
    motionEnabled = !motionEnabled;
    updateMotion();
  });

  function start() {
    const gl = canvas.getContext('webgl2', { alpha: false, antialias: false, depth: false });
    if (!gl) {
      status.textContent = 'この表示には WebGL 2 に対応したブラウザが必要です。';
      updateMotion();
      return;
    }
    try {
      renderer = createRenderer(gl);
      renderer.resize();
      status.textContent = '';
      updateMotion();
    } catch (error) {
      console.error(error);
      renderer = null;
      updateMotion();
      status.textContent = 'ネットの描画を開始できませんでした。';
    }
  }

  window.addEventListener('resize', () => {
    clearPointer();
    renderer?.resize();
    if (renderer && !motionEnabled) {
      renderer.draw(performance.now() / 1000, pointerState, 0);
    }
  });

  canvas.addEventListener('webglcontextlost', (event) => {
    event.preventDefault();
    renderer = null;
    updateMotion();
    status.textContent = 'ネットの描画を復元しています。';
  });

  canvas.addEventListener('webglcontextrestored', start);
  start();
})();
