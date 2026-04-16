#pragma once

static const uint32_t PTR_STACK_SIZE = 4;
static const uint32_t BEAM_PREPASS_SCALE = 4;

enum Axis {
  X,
  Y,
  Z,
};

struct DirectionalLight {
  float3 direction;
  float intesity;
  float3 color;
};

struct Vertex {
  float3 position;
  private uint32_t uv_data;

  property float2 uv {
    get { return float2(f16tof32(uv_data >> 16), f16tof32(uv_data & 0xFFFF)); }
  }
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

struct CountedBuffer<T> {
  private ByteAddressBuffer buffer;

  property uint32_t count {
    get { return buffer.Load<uint32_t>(0); }
  }

  __subscript(const uint32_t index)->T {
    get { return buffer.Load<T>(sizeof(uint32_t) + sizeof(DirectionalLight) * index); }
  }
};

uint32_t PackRGBA8(const float4 data) {
  const float ratio = 255.0f;

  uint32_t value = 0;

  value |= uint32_t(data.r * ratio) << 24;
  value |= uint32_t(data.g * ratio) << 16;
  value |= uint32_t(data.b * ratio) << 8;
  value |= uint32_t(data.a * ratio) << 0;

  return value;
}

float4 UnpackRGBA8(const uint32_t data) {
  const float ratio = 1.0f / 255.0f;

  float4 value;

  value.r = ((data >> 24) & 0xFF) * ratio;
  value.g = ((data >> 16) & 0xFF) * ratio;
  value.b = ((data >> 8) & 0xFF) * ratio;
  value.a = ((data >> 0) & 0xFF) * ratio;

  return value;
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
