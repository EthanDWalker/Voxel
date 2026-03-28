#pragma once

#include "Core/Util/fail.h"
#include "vulkan/vk_enum_string_helper.h"

namespace Core {
#define VK_CHECK(x)                                                                                \
  do {                                                                                             \
    VkResult err = x;                                                                              \
    if (err < 0) {                                                                                 \
      Core::Assert(false, "Detected Vulkan error: {}", string_VkResult(err));                      \
    }                                                                                              \
  } while (0)

VkDeviceAddress GetDeviceAddress(VkAccelerationStructureKHR as);

u32 AlignedSize(u32 value, u32 alignment);

size_t AlignedSize(size_t value, size_t alignment);

VkTransformMatrixKHR Mat4ToVkTransform(const Mat4f32 &m);

u32 CalculateMipLevels(const Vec3u32 image_extent);

u32 CalculateMipLevels(const Vec2u32 image_extent);

} // namespace Core
