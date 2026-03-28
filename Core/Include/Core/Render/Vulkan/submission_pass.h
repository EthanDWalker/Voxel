#pragma once

#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/Vulkan/image.h"
#include "Core/Render/Vulkan/image_util.h"
#include "Core/Render/Vulkan/indirect_draw.h"
#include "Core/Render/Vulkan/info.h"
#include "Core/Render/types.h"
#include <vector>
#include <volk.h>

namespace Core {
enum class SubPassType : u8 {
  Transfer,
  Compute,
  Graphic,
};

template <SubPassType> struct VulkanSubPass {};

struct BaseVulkanSubPass {
  std::vector<VkBufferMemoryBarrier2> buffer_barriers;
  std::vector<VkImageMemoryBarrier2> image_barriers;

  template <typename T> void AddDependency(const T &object, const DeviceResourceType dependency_type) {
    static_assert(false, "Unsupported type");
  }

  VkBufferMemoryBarrier2 &NewBufferBarrier(const VulkanBuffer &buffer) {
    VkBufferMemoryBarrier2 &barrier = buffer_barriers.emplace_back();
    barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.buffer = buffer.obj;
    barrier.size = buffer.size;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    return barrier;
  }

  VkImageMemoryBarrier2 &NewImageBarrier(const BaseVulkanImage &image) {
    VkImageMemoryBarrier2 &barrier = image_barriers.emplace_back();
    barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.obj;
    barrier.subresourceRange = ImageSubresourceRange(IsDepthFormat(image.format) ? VK_IMAGE_ASPECT_DEPTH_BIT
                                                                                 : VK_IMAGE_ASPECT_COLOR_BIT);
    return barrier;
  }
};

template <> struct VulkanSubPass<SubPassType::Compute> : BaseVulkanSubPass {
  template <DeviceResourceType T> void AddDependency(const VulkanBuffer &object) {
    VkBufferMemoryBarrier2 &barrier = NewBufferBarrier(object);
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

    if constexpr (T == DeviceResourceType::RWBuffer) {
      barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    } else if constexpr (T == DeviceResourceType::Buffer) {
      barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    } else {
      static_assert(false, "Unsupported dependency type");
    }
  }

  template <DeviceResourceType T> void AddDependency(const BaseVulkanImage &object) {
    VkImageMemoryBarrier2 &barrier = NewImageBarrier(object);
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

    if constexpr (T == DeviceResourceType::SampledImage) {
      barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    } else if constexpr (T == DeviceResourceType::StorageImage) {
      barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    } else if constexpr (T == DeviceResourceType::RWStorageImage) {
      barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
      barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    } else {
      static_assert(false, "Unsupported dependency type");
    }
  }

  template <DeviceResourceType T> void AddDependency(const VulkanIndirectDrawCommand &object) {
    AddDependency<T>(object.draw_count_buffer);
    AddDependency<T>(object.draw_buffer);
  }
};

template <> struct VulkanSubPass<SubPassType::Graphic> : BaseVulkanSubPass {
  template <DeviceResourceType T> void AddDependency(const VulkanBuffer &object) {
    VkBufferMemoryBarrier2 &barrier = NewBufferBarrier(object);
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                           VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT;

    if constexpr (T == DeviceResourceType::Buffer) {
      barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    } else if constexpr (T == DeviceResourceType::RWBuffer) {
      barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    } else if constexpr (T == DeviceResourceType::IndexBuffer) {
      barrier.dstAccessMask = VK_ACCESS_2_INDEX_READ_BIT;
      barrier.dstStageMask = VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
    } else {
      static_assert(false, "Unsupported dependency type");
    }
  }

  template <DeviceResourceType T> void AddDependency(const BaseVulkanImage &object) {
    VkImageMemoryBarrier2 &barrier = NewImageBarrier(object);

    if constexpr (T == DeviceResourceType::SampledImage) {
      barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
      barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    } else if constexpr (T == DeviceResourceType::StorageImage) {
      barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
      barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    } else if constexpr (T == DeviceResourceType::ColorAttachment) {
      barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
      barrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    } else if constexpr (T == DeviceResourceType::DepthAttachment) {
      barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
      barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      barrier.dstStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    } else {
      static_assert(false, "Unsupported dependency type");
    }
  }

  template <DeviceResourceType T> void AddDependency(const VulkanIndirectDrawCommand &object) {
    VkBufferMemoryBarrier2 &barrier = NewBufferBarrier(object.draw_buffer);
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;

    if constexpr (T == DeviceResourceType::Buffer) {
      barrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    } else {
      static_assert(false, "Unsupported dependency type");
    }

    VkBufferMemoryBarrier2 &count_barrier = NewBufferBarrier(object.draw_count_buffer);
    count_barrier.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;

    if constexpr (T == DeviceResourceType::Buffer) {
      count_barrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    } else {
      static_assert(false, "Unsupported dependency type");
    }
  }
};

template <> struct VulkanSubPass<SubPassType::Transfer> : BaseVulkanSubPass {
  template <DeviceResourceType T> void AddDependency(const VulkanBuffer &object) {
    VkBufferMemoryBarrier2 &barrier = NewBufferBarrier(object);
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;

    if constexpr (T == DeviceResourceType::TransferSrc) {
      barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    } else if constexpr (T == DeviceResourceType::TransferDst) {
      barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    } else {
      static_assert(false, "Unsupported dependency type");
    }
  }

  template <DeviceResourceType T> void AddDependency(const BaseVulkanImage &object) {
    VkImageMemoryBarrier2 &barrier = NewImageBarrier(object);
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;

    if constexpr (T == DeviceResourceType::TransferSrc) {
      barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
      barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    } else if constexpr (T == DeviceResourceType::TransferDst) {
      barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    } else {
      static_assert(false, "Unsupported dependency type");
    }
  }

  template <DeviceResourceType T> void AddDependency(const VulkanIndirectDrawCommand &object) {
    AddDependency<T>(object.draw_count_buffer);
    AddDependency<T>(object.draw_buffer);
  }
};
} // namespace Core
