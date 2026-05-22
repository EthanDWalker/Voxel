#pragma once

#include "volk.h"

#include "vma/vk_mem_alloc.h"

namespace Core {

struct BaseVulkanBuffer {
  BaseVulkanBuffer() = delete;

  BaseVulkanBuffer(const BaseVulkanBuffer &) = delete;
  BaseVulkanBuffer &operator=(const BaseVulkanBuffer &) = delete;

  BaseVulkanBuffer(BaseVulkanBuffer &&) = default;
  BaseVulkanBuffer &operator=(BaseVulkanBuffer &&) = delete;

  BaseVulkanBuffer(const char *const name) : name(name) {}
  const char *const name;
  VkBuffer obj = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  u64 size = 0;
  VkBufferUsageFlags usage = 0;
  VkPipelineStageFlags2 pipeline_stage_mask = VK_PIPELINE_STAGE_2_NONE;
  VkAccessFlags2 access_mask = VK_ACCESS_2_NONE;
  union {
    VkDeviceAddress device_address;
    void *host_address;
  };

  void DestroyBase();

  void CreateBase(const u64 size, const VkBufferUsageFlags usage, const bool host = false);
  void CreateAlignedBase(const u64 size, const VkBufferUsageFlags usage, const u64 alignment,
                         const bool host = false);
};
} // namespace Core
