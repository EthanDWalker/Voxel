#pragma once

#include "volk.h"

namespace Core {
enum class SamplerFilter : u8 {
  Linear = VK_FILTER_LINEAR,
  Nearest = VK_FILTER_NEAREST,
};

enum class SamplerAddressMode : u8 {
  ClampEdge,
  ClampBorder,
  Repeat,
};

struct VulkanSampler {
  VulkanSampler() = default;

  VulkanSampler(const VulkanSampler &) = delete;
  VulkanSampler &operator=(const VulkanSampler &) = delete;

  VulkanSampler(VulkanSampler &&) = default;
  VulkanSampler &operator=(VulkanSampler &&) = default;

  VkSampler obj;

  void Create(const SamplerFilter filter, const SamplerFilter mip_filter, const f32 lod_bias = 0.0f);

  ~VulkanSampler();
};
}; // namespace Core
