#pragma once

#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/Vulkan/descriptors.h"

namespace Core {
struct VulkanIndirectDrawCommand {
  VulkanIndirectDrawCommand() = delete;

  VulkanIndirectDrawCommand(const VulkanIndirectDrawCommand &) = delete;
  VulkanIndirectDrawCommand &operator=(const VulkanIndirectDrawCommand &) = delete;

  VulkanIndirectDrawCommand(VulkanIndirectDrawCommand &&) = default;
  VulkanIndirectDrawCommand &operator=(VulkanIndirectDrawCommand &&) = delete;

  VulkanBuffer draw_buffer = "indirect draw buffer";
  VulkanBuffer draw_count_buffer = "indirect draw count buffer";

  VulkanDescriptor descriptor;

  u32 max_draw_count = 0;

  ~VulkanIndirectDrawCommand() {}

  void Create(const u32 max_draw_count);
};
} // namespace Core
