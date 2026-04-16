#include "Core/Render/Vulkan/context.h"

#include "Core/Render/Vulkan/util.h"

namespace Core {
VkDeviceAddress GetDeviceAddress(VkAccelerationStructureKHR as) {
  ZoneScoped;
  VkAccelerationStructureDeviceAddressInfoKHR device_address_info{};
  device_address_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
  device_address_info.accelerationStructure = as;
  return vkGetAccelerationStructureDeviceAddressKHR(VulkanContext::device, &device_address_info);
}

u32 AlignedSize(u32 value, u32 alignment) {
  ZoneScoped;
  return (value + alignment - 1) & ~(alignment - 1);
}

u64 AlignedSize(u64 value, u64 alignment) {
  ZoneScoped;
  return (value + alignment - 1) & ~(alignment - 1);
}

VkTransformMatrixKHR Mat4ToVkTransform(const Mat4f32 &m) {
  ZoneScoped;
  VkTransformMatrixKHR transform;

  for (u8 i = 0; i < 3; i++) {
    for (u8 j = 0; j < 3; j++) {
      transform.matrix[i][j] = m[i][j];
    }

    transform.matrix[i][3] = m[3][i];
    transform.matrix[i][3] = m[3][i];
    transform.matrix[i][3] = m[3][i];
  }

  return transform;
}

u32 CalculateMipLevels(const Vec3u32 image_extent) {
  ZoneScoped;
  return u32(std::floor(
             std::log2(std::max(std::max(image_extent.width, image_extent.height), image_extent.depth)))) +
         1;
}

u32 CalculateMipLevels(const Vec2u32 image_extent) {
  ZoneScoped;
  return u32(std::floor(std::log2(std::max(image_extent.width, image_extent.height)))) + 1;
}
} // namespace Core
