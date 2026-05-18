#pragma once

#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/Vulkan/swapchain.h"

namespace Core {

struct DeviceHashSetKeyValuePair {
  u32 key;
  // 1 bits in shadow
  // 1 bits shadow ray cast
  // 30 bits light hit
  u32 occlusion_data;
  // 32 bits sample count
  f32 lighting;
  u32 sample_count;
};

struct DeviceHashSetHeader {
  u32 size;
  u32 insertion_failures;
};

struct DeviceHashSetSwappedData {
  VulkanBuffer<BufferType::StructuredBuffer, DeviceHashSetKeyValuePair> set_buffer = "hash set buffer";
  VulkanBuffer<BufferType::StructuredBuffer, DeviceHashSetHeader> header_buffer = "hash set header buffer";

  VulkanBuffer<BufferType::StagingBuffer> header_staging_buffer = "hash set header staging buffer";

  VulkanDescriptor descriptor;
};

struct DeviceHashSet {
  static const u32 EMPTY_KEY = 0xFFFFFFFF; // max u32
  static const u32 SET_BINDING = 1;
  static const u32 BACK_SET_BINDING = 2;

  void Create(const u32 size, const VkShaderStageFlags stage_flags);
  void Recreate(const u32 size, const VkShaderStageFlags stage_flags);

  std::array<DeviceHashSetSwappedData, VulkanSwapchain::FRAME_OVERLAP> swapped_data = {};

  VulkanDescriptorLayout descriptor_layout;

  u32 size;
};
}; // namespace Core
