#pragma once

#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/Vulkan/pipeline.h"
#include "Core/Render/Vulkan/other_buffer.h"
#include <type_traits>

namespace Core {

template <PipelineType Type, typename DataType = void> struct IndirectDispatchBuffer {};

template <typename DispatchDataType> struct IndirectDispatchBuffer<PipelineType::Compute, DispatchDataType> {
  struct DispatchCommand {
    Vec3u32 dispatch_groups;
    Vec3u32 dispatch_counts;
  };

  VulkanBuffer<BufferType::CountedBuffer, DispatchDataType> dispatch_data = "indirect dispatch data buffer";
  VulkanBuffer<BufferType::StructuredBuffer, DispatchCommand> dispatch_cmd = "indirect dispatch cmd buffer";
  VulkanDescriptorLayout descriptor_layout;
  VulkanDescriptor descriptor;
  u64 size;

  void Create(const u64 size, const VkShaderStageFlags stage_flags) {
    this->size = size;

    static_assert(!std::is_void_v<DispatchDataType>, "must have dispatch data type");

    dispatch_data.Create(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    dispatch_cmd.Create(1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);

    DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&dispatch_data);
    DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&dispatch_cmd);
    DescriptorBuilder::BuildLayout(stage_flags, descriptor_layout);
    DescriptorBuilder::BuildSet(stage_flags, descriptor_layout, descriptor);
    DescriptorBuilder::Reset();
  }

  void Recreate(const u64 size, const VkShaderStageFlags stage_flags) {
    this->size = size;

    dispatch_data.Destroy();
    dispatch_data.Create(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    descriptor.Update<DeviceResourceType::Buffer>(0, &dispatch_data);
  }
};

} // namespace Core
