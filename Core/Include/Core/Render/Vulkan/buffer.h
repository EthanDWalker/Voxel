#pragma once

#include "volk.h"

#include "vma/vk_mem_alloc.h"

namespace Core {

enum class BufferType : u8 {
  ByteBuffer,
  CountedBuffer,
  StructuredBuffer,
  StagingBuffer,
};

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

template <BufferType BufferType, typename ValueType = void> struct VulkanBuffer;

template <> struct VulkanBuffer<BufferType::ByteBuffer> : BaseVulkanBuffer {
  using BaseVulkanBuffer::BaseVulkanBuffer;

  void Create(const u64 size, const VkBufferUsageFlags usage, const u64 alignment = 0) {
    if (alignment == 0) {
      CreateBase(size, usage);
    } else {
      CreateAlignedBase(size, usage, alignment);
    }
  }

  void Destroy() { DestroyBase(); }
  ~VulkanBuffer<BufferType::ByteBuffer>() { Destroy(); }
};

template <typename ValueType> struct VulkanBuffer<BufferType::CountedBuffer, ValueType> : BaseVulkanBuffer {
  using BaseVulkanBuffer::BaseVulkanBuffer;

  void Create(const u64 max_count, const VkBufferUsageFlags usage) {
    static_assert(!std::is_same_v<ValueType, void>, "value type must be set");
    CreateBase(max_count * sizeof(ValueType) + sizeof(u32), usage);
  }

  void Destroy() { DestroyBase(); }
  ~VulkanBuffer<BufferType::CountedBuffer, ValueType>() { Destroy(); };
};

template <typename ValueType>
struct VulkanBuffer<BufferType::StructuredBuffer, ValueType> : BaseVulkanBuffer {
  using BaseVulkanBuffer::BaseVulkanBuffer;

  void Create(const u64 count, const VkBufferUsageFlags usage) {
    static_assert(!std::is_same_v<ValueType, void>, "value type must be set");
    CreateBase(count * sizeof(ValueType), usage);
  }

  void Destroy() { DestroyBase(); }
  ~VulkanBuffer<BufferType::StructuredBuffer, ValueType>() { Destroy(); };
};

template <> struct VulkanBuffer<BufferType::StagingBuffer> : BaseVulkanBuffer {
  using BaseVulkanBuffer::BaseVulkanBuffer;

  u32 current_byte_ptr = 0;

  void BuildAddStagingBinding(const u32 binding_size) { size += binding_size; }

  void Build() {
    CreateBase(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, /*host*/ true);
  }

  void IncrementMemory(const void *const binding_ptr, const u32 binding_size) {
    memcpy((char *)host_address + current_byte_ptr, binding_ptr, binding_size);
    current_byte_ptr += binding_size;
  }

  void Destroy() { DestroyBase(); }
  ~VulkanBuffer<BufferType::StagingBuffer>() { Destroy(); };
};
} // namespace Core
