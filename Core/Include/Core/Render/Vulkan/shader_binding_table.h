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

  BaseVulkanBuffer ray_gen = "shader binding table ray gen";
  BaseVulkanBuffer miss = "shader binding table miss";
  BaseVulkanBuffer closest_hit = "shader binding table closest hit";

  VkStridedDeviceAddressRegionKHR ray_gen_entry{};
  VkStridedDeviceAddressRegionKHR miss_entry{};
  VkStridedDeviceAddressRegionKHR closest_hit_entry{};
  VkStridedDeviceAddressRegionKHR callable_entry{};

  void Destroy() {
    ray_gen.DestroyBase();
    miss.DestroyBase();
    closest_hit.DestroyBase();
  }
  ~VulkanShaderBindingTable() { Destroy(); }

  void Create(VkPipeline pipeline, std::span<VkRayTracingShaderGroupCreateInfoKHR> shader_groups);
};
} // namespace Core
