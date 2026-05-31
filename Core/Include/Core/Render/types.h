#pragma once

#include "volk.h"

namespace Core {
const u32 GPU_ALIGNMENT = 16;

using Index = u32;
using Instance = VkAccelerationStructureInstanceKHR;

struct alignas(GPU_ALIGNMENT) AllocateInfo {
  u32 depth;
  u32 leaf;
  u32 mesh_index;
  u32 albedo_index;
};

struct alignas(GPU_ALIGNMENT) Vertex {
  vec<4, f16> position;
  u16 normal;
  u16 _p0;
  vec<2, f16> uv;
};

struct alignas(GPU_ALIGNMENT) DirectionalLight {
  Vec3f32 direction;
  f32 intesity;
  Vec3f32 color;
};

struct LeafNode {
  u32 data;
};

static const u32 LEAF_PAGE_SIZE_EXP = 16;
static const u32 LEAF_PAGE_SIZE = 1 << LEAF_PAGE_SIZE_EXP;

static const u32 VOXEL_AABB_SIZE_EXP = 5;
static const f32 VOXEL_AABB_SIZE = 1 << VOXEL_AABB_SIZE_EXP;
static const f32 VOXEL_SIZE = (1 << (VOXEL_AABB_SIZE_EXP - 2));

struct LeafHeader {
  u32 allocated_leaf_count;
  u32 page_size_exp = LEAF_PAGE_SIZE_EXP;
};

struct AABB {
  Vec3f32 min;
  Vec3f32 max;
};

struct BranchNode {
  u64 child_mask;
  u32 child_ptr;
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
