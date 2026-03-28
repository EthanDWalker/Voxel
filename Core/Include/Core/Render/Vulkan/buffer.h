#pragma once

#include "volk.h"

#include "vma/vk_mem_alloc.h"

namespace Core {
struct VulkanBuffer {
  VulkanBuffer() = default;

  VulkanBuffer(const VulkanBuffer &) = delete;
  VulkanBuffer &operator=(const VulkanBuffer &) = delete;

  VulkanBuffer(VulkanBuffer &&) = default;
  VulkanBuffer &operator=(VulkanBuffer &&) = default;

  VkBuffer obj = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  u64 size = 0;
  VkBufferUsageFlags usage = 0;
  void *address = nullptr;

  ~VulkanBuffer();
  void Destroy();

  void Create(const u64 size, const VkBufferUsageFlags usage, const bool host = false);
};
} // namespace Core
