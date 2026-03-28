#pragma once

#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/Vulkan/descriptors.h"

namespace Core {
struct VulkanIndirectDrawCommand {
  VulkanIndirectDrawCommand() = default;

  VulkanIndirectDrawCommand(const VulkanIndirectDrawCommand &) = delete;
  VulkanIndirectDrawCommand &operator=(const VulkanIndirectDrawCommand &) = delete;

  VulkanIndirectDrawCommand(VulkanIndirectDrawCommand &&) = default;
  VulkanIndirectDrawCommand &operator=(VulkanIndirectDrawCommand &&) = default;

  VulkanBuffer draw_buffer;
  VulkanBuffer draw_count_buffer;

  VulkanDescriptor descriptor;

  u32 max_draw_count = 0;

  ~VulkanIndirectDrawCommand() {}

  void Create(const u32 max_draw_count);
};
} // namespace Core
