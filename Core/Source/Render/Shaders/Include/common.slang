#pragma once

static const uint32_t BEAM_PREPASS_SCALE_EXP = 2;
static const uint32_t INDIRECT_PREPASS_SCALE_EXP = 1;
static const uint32_t SENTINAL = 0xFFFFFFFF;

static const float PI = 3.14159f;

enum Axis {
  X,
  Y,
  Z,
};

typedef uint32_t Index;

struct DirectionalLight {
  float3 direction;
  float intensity;
  float3 color;
};

struct IndirectLightingRayDispatch {
  uint32_t3 tree_index;
};

struct DispatchCommand {
  uint32_t groups_x;
  uint32_t groups_y;
  uint32_t groups_z;
  uint32_t total_x;
  uint32_t total_y;
  uint32_t total_z;
};

struct RWDispatchCommand {
  Atomic<uint32_t> groups_x;
  Atomic<uint32_t> groups_y;
  Atomic<uint32_t> groups_z;
  Atomic<uint32_t> total_x;
  Atomic<uint32_t> total_y;
  Atomic<uint32_t> total_z;
};

struct Vertex {
  uint64_t position_data;
  uint16_t normal_data;
  uint16_t _p0;
  uint32_t uv_data;

  property float3 position {
    get { return f16tof32(uint32_t3(position_data >> uint32_t3(32, 16, 0)) & 0xFFFF); }
    set { position_data = SumComponents(uint64_t3(f32tof16(newValue)) << uint32_t3(32, 16, 0)); }
  };

  property float3 normal {
    get {
      const float theta_ratio = ((1 << 8) - 1) * (1.0f / (2.0f * PI));
      const float theta = (((normal_data >> 0) & 0xFF) / theta_ratio) - PI;

      const float phi_ratio = ((1 << 8) - 1) * (1.0f / PI);
      const float phi = (((normal_data >> 8) & 0xFF) / phi_ratio);

      return normalize(float3(cos(phi), sin(phi) * sin(theta), sin(phi) * cos(theta)));
    }
    set {
      normal_data = 0;

      const float theta_ratio = ((1 << 8) - 1) * (1.0f / (2.0f * PI));
      normal_data |= uint16_t((atan2(newValue.y, newValue.x) + PI) * theta_ratio) << 0;

      const float phi_ratio = ((1 << 8) - 1) * (1.0f / PI);
      normal_data |= uint16_t(acos(newValue.z) * phi_ratio) << 8;
    }
  };

  property float2 uv {
    get { return f16tof32((uv_data >> uint32_t2(0, 16)) & 0xFFFF); }
    set { uv_data = (uint32_t)SumComponents(asuint(f32tof16(newValue)) << uint32_t2(0, 16)); }
  };
}

interface IBuffer {
  T Load<T, let I : uint32_t>();
};

struct RWByteBuffer : IBuffer {
  RWByteAddressBuffer data;
  T Load<T, let I : uint32_t>() { return data.Load<T>(I); }
};

struct ByteBuffer : IBuffer {
  ByteAddressBuffer data;
  T Load<T, let I : uint32_t>() { return data.Load<T>(I); }
};

struct CountedBufferHeader {
  uint32_t max_count;
  uint32_t count;
  uint32_t _p0;
  uint32_t _p1;
};

struct RWCountedBuffer<T> {
  private RWByteAddressBuffer buffer;

  property uint32_t max_count {
    get { return buffer.Load<uint32_t>(0); }
  };

  bool Append(const T value) {
    uint32_t index;
    buffer.InterlockedAdd(4, 1, index);

    if (index >= this.max_count) {
      return false;
    }

    buffer.Store<T>(sizeof(CountedBufferHeader) + sizeof(T) * index, value);
    return true;
  }
};

struct CountedBuffer<T> {
  private ByteAddressBuffer buffer;

  property uint32_t max_count {
    get { return buffer.Load<uint32_t>(0); }
  }

  property uint32_t count {
    get { return buffer.Load<uint32_t>(4); }
  }

  __subscript(const uint32_t index)->T {
    get { return buffer.Load<T>(sizeof(CountedBufferHeader) + sizeof(T) * index); }
  }
};

uint32_t SumComponents(const uint32_t3 v) { return v.x + v.y + v.z; }
uint64_t SumComponents(const uint64_t2 v) { return v.x + v.y; }
uint64_t SumComponents(const uint64_t3 v) { return v.x + v.y + v.z; }
uint32_t SumComponents(const uint32_t4 v) { return v.x + v.y + v.z + v.w; }

T SumComponent<T>(const vector<T, 2> v)
    where T : IArithmetic { return v.x + v.y; }
T SumComponent<T>(const vector<T, 3> v)
    where T : IArithmetic { return v.x + v.y + v.z; }
T SumComponent<T>(const vector<T, 4> v)
    where T : IArithmetic { return v.x + v.y + v.z + v.w; }

uint32_t PackRGBA8(const float4 data) {
  static const float ratio = float((1 << 8) - 1);
  return SumComponents(uint32_t4(data * ratio) << uint32_t4(0, 8, 16, 24));
}

float4 UnpackRGBA8(const uint32_t data) {
  static const float ratio = 1.0f / float((1 << 8) - 1);
  return ((uint32_t4(data) >> uint32_t4(0, 8, 16, 24)) & uint32_t4(0xFF)) * ratio;
}

vector<float, N> FloatDivideBy2ToN<let N : int>(const vector<float, N> x, const uint32_t n) {
  return asfloat(asuint(x) - (n << 23));
}

vector<float, N> FloatMultiplyBy2ToN<let N : int>(const vector<float, N> x, const uint32_t n) {
  return asfloat(asuint(x) + (n << 23));
}

bool RayAABBIntersect(const float3 ray_dir, const float3 ray_origin, const float3 bounds_min,
                      const float3 bounds_max, inout float t_min, inout float t_max) {
  float t_min_y, t_max_y, t_min_z, t_max_z;
  const float x_inv_dir = 1 / ray_dir.x;
  if (x_inv_dir >= 0) {
    t_min = (bounds_min.x - ray_origin.x) * x_inv_dir;
    t_max = (bounds_max.x - ray_origin.x) * x_inv_dir;
  } else {
    t_min = (bounds_max.x - ray_origin.x) * x_inv_dir;
    t_max = (bounds_min.x - ray_origin.x) * x_inv_dir;
  }

  const float y_inv_dir = 1 / ray_dir.y;
  if (y_inv_dir >= 0) {
    t_min_y = (bounds_min.y - ray_origin.y) * y_inv_dir;
    t_max_y = (bounds_max.y - ray_origin.y) * y_inv_dir;
  } else {
    t_min_y = (bounds_max.y - ray_origin.y) * y_inv_dir;
    t_max_y = (bounds_min.y - ray_origin.y) * y_inv_dir;
  }

  if (t_min > t_max_y || t_min_y > t_max)
    return false;
  if (t_min_y > t_min)
    t_min = t_min_y;
  if (t_max_y < t_max)
    t_max = t_max_y;

  const float z_inv_dir = 1 / ray_dir.z;
  if (z_inv_dir >= 0) {
    t_min_z = (bounds_min.z - ray_origin.z) * z_inv_dir;
    t_max_z = (bounds_max.z - ray_origin.z) * z_inv_dir;
  } else {
    t_min_z = (bounds_max.z - ray_origin.z) * z_inv_dir;
    t_max_z = (bounds_min.z - ray_origin.z) * z_inv_dir;
  }

  if (t_min > t_max_z || t_min_z > t_max)
    return false;
  if (t_min_z > t_min)
    t_min = t_min_z;
  if (t_max_z < t_max)
    t_max = t_max_z;
  return t_max > 0.0f;
}

float3 GetBarycentric<let N : int>(const vector<float, N> positions[3], const vector<float, N> point) {
  const vector<float, N> v0 = positions[2] - positions[0];
  const vector<float, N> v1 = positions[1] - positions[0];
  const vector<float, N> v2 = point - positions[0];
  const float d00 = dot(v0, v0);
  const float d01 = dot(v0, v1);
  const float d11 = dot(v1, v1);
  const float d20 = dot(v2, v0);
  const float d21 = dot(v2, v1);
  const float denom = d00 * d11 - d01 * d01;

  float3 coords;
  coords.y = (d00 * d21 - d01 * d20) / denom;
  coords.z = (d11 * d20 - d01 * d21) / denom;
  coords.x = 1.0f - coords.z - coords.y;
  return coords;
}

float SampleBeamImage(const uint32_t2 main_image_location, const RWTexture2D<float> beam_image) {
  const float2 beam_image_location_float = FloatDivideBy2ToN(main_image_location, BEAM_PREPASS_SCALE_EXP);
  const int32_t2 beam_image_location = main_image_location >> BEAM_PREPASS_SCALE_EXP;
  const bool2 is_whole = (beam_image_location_float == float2(beam_image_location));

  float t = 0.0f;
  if (all(is_whole)) {
    t = beam_image.Load(beam_image_location);
    t = min(t, beam_image.Load(beam_image_location + int32_t2(0, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 0)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 1)));
  } else if (is_whole.x) {
    t = beam_image.Load(beam_image_location);
    t = min(t, beam_image.Load(beam_image_location + int32_t2(0, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 0)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(-1, 0)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(-1, 1)));
  } else if (is_whole.y) {
    t = beam_image.Load(beam_image_location);
    t = min(t, beam_image.Load(beam_image_location + int32_t2(0, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(0, -1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 0)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, -1)));
  } else {
    t = beam_image.Load(beam_image_location);
    t = min(t, beam_image.Load(beam_image_location + int32_t2(0, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 0)));
  }
  t *= 0.99f;
  return t;
}

float3 SampleEnvironmentColor(const float3 ray_dir, const DirectionalLight sun) {
  static const float3 GROUND_COLOR = float3(0.1f);
  static const float3 SKY_COLOR_HORIZON = float3(207, 236, 247) / 255.0f;
  static const float3 SKY_COLOR_ZENITH = float3(160, 217, 239) / 400.0f;

  const float sky_gradient_t = pow(smoothstep(0.0f, 0.4f, ray_dir.y), 0.35f);
  const float3 sky_gradient = lerp(SKY_COLOR_HORIZON, SKY_COLOR_ZENITH, sky_gradient_t);
  const float ground_to_sky_t = smoothstep(-0.01f, 0.0f, ray_dir.y);

  const float sun_focus = 300.0f / sun.intensity;
  const float sun_color = pow(max(0, dot(ray_dir, -sun.direction)), sun_focus);
  const float sun_mask = step(ground_to_sky_t, 1.0f);

  const float3 sky_color = lerp(GROUND_COLOR, sky_gradient, ground_to_sky_t) + sun_mask * sun_color * sun.intensity;
  return sky_color;
}
