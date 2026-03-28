#pragma once

namespace Core {
const u32 GPU_ALIGNMENT = 16;

using Index = u32;

struct alignas(GPU_ALIGNMENT) Vertex {
  Vec3f32 position;
  vec<2, f16> uv;
};

struct alignas(GPU_ALIGNMENT) AABB {
  Vec3f32 min;
  f32 _p0;
  Vec3f32 max;
  f32 _p1;
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
};

} // namespace Core
