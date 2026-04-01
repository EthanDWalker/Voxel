#pragma once

#include "Core/Render/types.h"
#include "volk.h"

namespace Core {
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
