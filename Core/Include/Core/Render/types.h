#pragma once

#include "volk.h"

namespace Core {
const u32 GPU_ALIGNMENT = 16;

struct Material {
  Vec4f32 albedo;
  float rough;
  float metallic;
  float emmisive;
  float reflect;
};

struct alignas(GPU_ALIGNMENT) ToneMapPushConstants {
  f32 delta_time;
};

struct alignas(GPU_ALIGNMENT) FrameLuminanceData {
  f32 acc_log_luminance;
  f32 total_weight;
};

struct alignas(GPU_ALIGNMENT) ChunkAllocationInfo {
  Vec3u32 chunk_index;
  u32 lod;
  u32 alloc_leaf;
  u32 color;
  i32 seed;
  u32 max_depth;
};

struct alignas(GPU_ALIGNMENT) VoxelVolume {
  Vec3u32 min_tree_index;
  u32 depth;
  Vec3u32 max_tree_index;
  u32 material_index;
};

struct alignas(GPU_ALIGNMENT) LightingPushConstants {
  f32 diffuse_alpha;
  f32 specular_alpha;
  u32 frame_number;
};

struct alignas(GPU_ALIGNMENT) Raycast {
  Vec3f32 origin;
  f32 t_max = 1e10f;
  Vec3f32 dir;
  u32 _p1;
};

struct alignas(GPU_ALIGNMENT) RaycastResult {
  Vec3f32 hit_position;
  uint32_t iterations;
  Vec3f32 hit_normal;
  float t;
  Vec3u32 hit_tree_index;
  uint32_t leaf_ptr;
  uint32_t hit_material_index;
  uint32_t hit_level;
  bool hit;
};

struct AABB {
  Vec3f32 min;
  Vec3f32 max;
};

struct Mesh {
  u32 index_count;
  u32 albedo_image_index;
};

struct IndirectLightingRayDispatch {
  Vec3u32 tree_index;
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
  IndirectDispatchBuffer,

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
