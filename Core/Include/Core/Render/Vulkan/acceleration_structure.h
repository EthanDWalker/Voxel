#pragma once

#include "Core/Render/Vulkan/buffer.h"
#include "volk.h"

namespace Core {
struct VulkanAccelerationStructure {
  VulkanAccelerationStructure() = default;

  VulkanAccelerationStructure(const VulkanAccelerationStructure &) = delete;
  VulkanAccelerationStructure &operator=(const VulkanAccelerationStructure &) = delete;

  VulkanAccelerationStructure(VulkanAccelerationStructure &&) = default;
  VulkanAccelerationStructure &operator=(VulkanAccelerationStructure &&) = delete;

  VkAccelerationStructureKHR obj;
  VulkanBuffer buffer = "acceleration structure buffer";

  ~VulkanAccelerationStructure();

  void CreateBase(const VkAccelerationStructureGeometryKHR &geometry,
                  const VkAccelerationStructureBuildRangeInfoKHR &offset,
                  const VkAccelerationStructureTypeKHR type);
  void CreateBottomLevel(const VulkanBuffer &vertex_buffer, const VulkanBuffer &index_buffer);
  void CreateTopLevel(const VulkanBuffer &instance_buffer, const u32 instance_count);

  void RecreateTopLevel(const VulkanBuffer &instance_buffer, const u32 instance_count);
};
} // namespace Core
