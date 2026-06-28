#pragma once

#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/types.h"
#include "buffer.h"

namespace Core {

enum class BufferType : u8 {
  ByteBuffer,
  CountedBuffer,
  StructuredBuffer,
  StagingBuffer,
};

template <BufferType BufferType, typename ValueType = void> struct VulkanBuffer;

template <> struct VulkanBuffer<BufferType::ByteBuffer> : BaseVulkanBuffer {
  using BaseVulkanBuffer::BaseVulkanBuffer;

  void Create(const u64 size, const VkBufferUsageFlags usage, const bool host = false,
              const u64 alignment = 0) {
    if (alignment == 0) {
      CreateBase(size, usage, host);
    } else {
      CreateAlignedBase(size, usage, alignment, host);
    }
  }

  void Destroy() { DestroyBase(); }
  ~VulkanBuffer<BufferType::ByteBuffer>() { Destroy(); }
};

template <typename ValueType> struct VulkanBuffer<BufferType::CountedBuffer, ValueType> : BaseVulkanBuffer {
  using BaseVulkanBuffer::BaseVulkanBuffer;

  u32 max_count;
  u32 cpu_append_count;

  struct alignas(GPU_ALIGNMENT) Header {
    u32 max_count;
    u32 count;
  };

  void Create(const u32 max_count, const VkBufferUsageFlags usage) {
    static_assert(!std::is_same_v<ValueType, void>, "value type must be set");
    Assert((usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) != 0,
           "counted buffers must have TRANSFER_DST flag set");
    CreateBase(max_count * sizeof(ValueType) + sizeof(Header), usage);

    this->max_count = max_count;
    this->cpu_append_count = 0;

    VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
      VulkanSubPass<SubPassType::Transfer> transfer_pass;
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(*this);

      cmd.BindSubPass(transfer_pass);

      cmd.FillBuffer(*this, sizeof(u32), max_count, offsetof(Header, max_count));
    });
  }

  void Clear(VulkanCommandBuffer &cmd) { cmd.FillBuffer(*this, sizeof(u32), 0, offsetof(Header, count)); }

  // only call this if only appending on the CPU
  // staging buffer is assumed to only conatain one ValueType
  void Append(VulkanCommandBuffer &cmd, const VulkanBuffer<BufferType::StagingBuffer> &staging_buffer,
              const u32 count = 1) {
    cmd.UploadBufferToBuffer(staging_buffer, *this, sizeof(ValueType) * count, 0,
                             sizeof(ValueType) * cpu_append_count + sizeof(Header));
    cpu_append_count += count;
    cmd.FillBuffer(*this, sizeof(u32), cpu_append_count, offsetof(Header, count));
  };

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

  void Create(const u32 size, const VkBufferUsageFlags other_flags = 0) {
    CreateBase(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | other_flags,
               /*host*/ true);
    memset(host_address, 0, size);
  }

  void Destroy() { DestroyBase(); }
  ~VulkanBuffer<BufferType::StagingBuffer>() { Destroy(); };
};
} // namespace Core
