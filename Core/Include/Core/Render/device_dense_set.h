#pragma once

#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/types.h"

namespace Core {

struct DeviceDenseSet {
  DeviceDenseSet() = delete;

  static const u32 HEADER_BINDING = 0;
  static const u32 VALUE_BINDING = 1;
  static const u32 KEY_BINDING = 2;
  static const bool HOST = false;
  static const u32 EMPTY_KEY = 0xFFFFFFFF;

  VulkanBuffer key_buffer = "device key buffer";
  VulkanBuffer value_buffer = "device value buffer";
  VulkanBuffer header_buffer = "device dense set header buffer";
  VulkanBuffer header_host_buffer = "device dense set header host buffer";
  VulkanDescriptor descriptor;
  u32 size;

  struct Value {
    u32 triangle_id;
  };

  struct alignas(GPU_ALIGNMENT) Header {
    u32 size;
    u32 value_count;
  };

  DeviceDenseSet(const u32 size, const VkShaderStageFlags stage_flags);
};
}; // namespace Core
