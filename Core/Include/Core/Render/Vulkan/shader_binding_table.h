#pragma once

#include "Core/Render/Vulkan/buffer.h"
#include "volk.h"
#include <span>

namespace Core {
struct VulkanShaderBindingTable {
  VulkanShaderBindingTable() = default;

  VulkanShaderBindingTable(const VulkanShaderBindingTable &) = delete;
  VulkanShaderBindingTable &operator=(const VulkanShaderBindingTable &) = delete;

  VulkanShaderBindingTable(VulkanShaderBindingTable &&) = default;
  VulkanShaderBindingTable &operator=(VulkanShaderBindingTable &&) = delete;

  VulkanBuffer ray_gen = "shader binding table ray gen";
  VulkanBuffer miss = "shader binding table miss";
  VulkanBuffer closest_hit = "shader binding table closest hit";

  VkStridedDeviceAddressRegionKHR ray_gen_entry{};
  VkStridedDeviceAddressRegionKHR miss_entry{};
  VkStridedDeviceAddressRegionKHR closest_hit_entry{};
  VkStridedDeviceAddressRegionKHR callable_entry{};

  void Destroy() {
    ray_gen.Destroy();
    miss.Destroy();
    closest_hit.Destroy();
  }
  ~VulkanShaderBindingTable() {}

  void Create(VkPipeline pipeline, std::span<VkRayTracingShaderGroupCreateInfoKHR> shader_groups);
};
} // namespace Core
