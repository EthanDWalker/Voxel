#include "Core/Render/Vulkan/shader_binding_table.h"
#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/util.h"
#include <cstring>
#include <span>
#include <vector>

namespace Core {
VkPhysicalDeviceRayTracingPipelinePropertiesKHR GetRaytracingPipelineProperties() {
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR properties{};
  properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;

  VkPhysicalDeviceProperties2 device_properties{};
  device_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  device_properties.pNext = &properties;

  vkGetPhysicalDeviceProperties2(VulkanContext::physical_device, &device_properties);

  return properties;
}

void VulkanShaderBindingTable::Create(VkPipeline pipeline,
                                      std::span<VkRayTracingShaderGroupCreateInfoKHR> shader_groups) {
  auto properties = GetRaytracingPipelineProperties();

  const u32 handle_size = properties.shaderGroupHandleSize;
  const u32 handle_size_aligned =
      AlignUp(properties.shaderGroupHandleSize, properties.shaderGroupHandleAlignment);
  const u32 group_count = static_cast<u32>(shader_groups.size());
  const u32 binding_table_size = group_count * handle_size_aligned;

  Assert(handle_size_aligned <= properties.maxShaderGroupStride, "properties dont support raytracing");
  Assert(handle_size_aligned % properties.shaderGroupHandleAlignment == 0,
         "properties dont support raytracing");

  std::vector<u8> shader_handle_storage(binding_table_size);
  VK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(VulkanContext::device, pipeline, 0, group_count,
                                                binding_table_size, shader_handle_storage.data()));

  const VkBufferUsageFlags buffer_usage_flags = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                                VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  ray_gen.CreateBase(handle_size_aligned, buffer_usage_flags);
  miss.CreateBase(handle_size_aligned, buffer_usage_flags);
  closest_hit.CreateBase(handle_size_aligned, buffer_usage_flags);

  BaseVulkanBuffer staging_buffer = "shader binding table staging buffer";
  staging_buffer.CreateBase(shader_handle_storage.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
  memcpy(staging_buffer.host_address, shader_handle_storage.data(), shader_handle_storage.size());

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    cmd.UploadBufferToBuffer(staging_buffer, ray_gen, handle_size_aligned);
    cmd.UploadBufferToBuffer(staging_buffer, miss, handle_size_aligned, handle_size_aligned);
    cmd.UploadBufferToBuffer(staging_buffer, closest_hit, handle_size_aligned, handle_size_aligned * 2);
  });

  ray_gen_entry.deviceAddress = ray_gen.device_address;
  ray_gen_entry.stride = handle_size_aligned;
  ray_gen_entry.size = handle_size_aligned;

  miss_entry.deviceAddress = miss.device_address;
  miss_entry.stride = handle_size_aligned;
  miss_entry.size = handle_size_aligned;

  closest_hit_entry.deviceAddress = closest_hit.device_address;
  closest_hit_entry.stride = handle_size_aligned;
  closest_hit_entry.size = handle_size_aligned;

  staging_buffer.DestroyBase();
}
} // namespace Core
