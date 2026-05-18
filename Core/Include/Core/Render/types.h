#pragma once

#include "volk.h"

namespace Core {
const u32 GPU_ALIGNMENT = 16;

using Index = u32;
using Instance = VkAccelerationStructureInstanceKHR;

struct alignas(GPU_ALIGNMENT) AllocateInfo {
  u32 depth;
  u32 leaf;
  u32 mesh_id;
};

struct alignas(GPU_ALIGNMENT) Vertex {
  vec<4, f16> position;
  u16 normal;
  u16 _p0;
  vec<2, f16> uv;
};

struct alignas(GPU_ALIGNMENT) AABB {
  Vec3f32 min;
  f32 _p0;
  Vec3f32 max;
  f32 _p1;
};

struct alignas(GPU_ALIGNMENT) DirectionalLight {
  Vec3f32 direction;
  f32 intesity;
  Vec3f32 color;
};

struct alignas(GPU_ALIGNMENT) VoxelVolume {
  Vec3f32 min;
  u32 _p0;
  Vec3f32 max;
  u32 _p1;
};

struct alignas(GPU_ALIGNMENT) Raycast {
  Vec3f32 origin;
  f32 t_max = 1e10f;
  Vec3f32 dir;
  u32 _p1;
};

struct alignas(GPU_ALIGNMENT) RaycastResult {
  Vec4f32 hit_color;
  Vec3f32 hit_position;
  u32 iterations;
  Vec3f32 hit_normal;
  f32 t;
  Vec3u32 hit_tree_index;
  bool error;
  bool hit;
};

struct IndirectLightingRayDispatch {
  u32 frame_sample_count;
  u32 leaf_ptr;
};

struct Mesh {
  u64 id;
  u32 index_count;
  u32 vertex_count;
};

enum class SamplerFilter : u8 {
  Linear,
  Nearest,
};

enum class SamplerAddressMode : u8 {
  ClampEdge,
  ClampBorder,
  Repeat,
};

enum class DeviceResourceType : u8 {
  Buffer,
  RWBuffer,
  IndexBuffer,

  StorageImage,
  SampledImage,
  RWStorageImage,

  ColorAttachment,
  DepthAttachment,

  TransferSrc,
  TransferDst,

  CombinedImageSampler,
  Sampler,

  AccelerationStructure,
};

} // namespace Core
