#pragma once

#include "volk.h"

#include "vma/vk_mem_alloc.h"

namespace Core {
struct VulkanBuffer {
  VulkanBuffer() = delete;

  VulkanBuffer(const VulkanBuffer &) = delete;
  VulkanBuffer &operator=(const VulkanBuffer &) = delete;

  VulkanBuffer(VulkanBuffer &&) = default;
  VulkanBuffer &operator=(VulkanBuffer &&) = delete;

  VulkanBuffer(const char *const name) : name(name) {}
  const char *const name;
  VkBuffer obj = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  u64 size = 0;
  VkBufferUsageFlags usage = 0;
  VkDeviceAddress device_address;
  void *host_address;

  ~VulkanBuffer();
  void Destroy();

  void Create(const u64 size, const VkBufferUsageFlags usage, const bool host = false);
  void CreateAligned(const u64 size, const VkBufferUsageFlags usage, const u64 alignment, const bool host = false);
};
} // namespace Core
