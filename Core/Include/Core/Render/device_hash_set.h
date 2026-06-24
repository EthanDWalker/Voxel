#pragma once

#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/Vulkan/other_buffer.h"
#include "Core/Render/Vulkan/swapchain.h"

namespace Core {

struct DeviceHashSetKey {
  u32 key;
};

struct DeviceHashSetHeader {
  u32 size;
};

struct DeviceHashSetLightingData {
  Vec3f32 acc_lighting;
  f32 sample_count;
};

struct DeviceHashSetSwappedData {
  VulkanBuffer<BufferType::StructuredBuffer, DeviceHashSetKey> key_buffer = "hash set key buffer";
  VulkanBuffer<BufferType::StructuredBuffer, DeviceHashSetLightingData> diffuse_data_buffer =
      "hash set diffuse data buffer";
  VulkanBuffer<BufferType::StructuredBuffer, DeviceHashSetLightingData> specular_data_buffer =
      "hash set specular data buffer";

  VulkanBuffer<BufferType::StructuredBuffer, DeviceHashSetHeader> header_buffer = "hash set header buffer";
  VulkanBuffer<BufferType::StagingBuffer> header_staging_buffer = "hash set header staging buffer";

  VulkanDescriptor descriptor;
};

struct DeviceHashSet {
  static const u32 EMPTY_KEY = 0xFFFFFFFF; // max u32
  static const u32 HEADER_BINDING = 0;

  static const u32 KEY_BINDING = 1;
  static const u32 BACK_KEY_BINDING = 2;

  static const u32 DIFFUSE_DATA_BINDING = 3;
  static const u32 BACK_DIFFUSE_DATA_BINDING = 4;

  static const u32 SPECULAR_DATA_BINDING = 5;
  static const u32 BACK_SPECULAR_DATA_BINDING = 6;

  void Create(const u32 size, const VkShaderStageFlags stage_flags);
  void Recreate(const u32 size, const VkShaderStageFlags stage_flags);

  std::array<DeviceHashSetSwappedData, VulkanSwapchain::FRAME_OVERLAP> swapped_data = {};

  VulkanDescriptorLayout descriptor_layout;

  u32 size;
};
}; // namespace Core
