#pragma once

#include <volk.h>

namespace Core {
struct VulkanCommandPool {
  VulkanCommandPool() = default;

  VulkanCommandPool(const VulkanCommandPool &) = delete;
  VulkanCommandPool &operator=(const VulkanCommandPool &) = delete;

  VulkanCommandPool(VulkanCommandPool &&) = default;
  VulkanCommandPool &operator=(VulkanCommandPool &&) = default;

  VkCommandPool obj = VK_NULL_HANDLE;

  ~VulkanCommandPool();

  void Create(u32 queue_index);

  void Reset();

  VkCommandBuffer AllocateCommandBuffer(VkCommandBufferLevel buffer_level);
};
} // namespace Core
