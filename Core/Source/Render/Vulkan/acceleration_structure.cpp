#include "Core/Render/Vulkan/acceleration_structure.h"
#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/util.h"
#include "Core/Render/types.h"

namespace Core {

VulkanAccelerationStructure::~VulkanAccelerationStructure() {
  vkDestroyAccelerationStructureKHR(VulkanContext::device, obj, nullptr);
}

void VulkanAccelerationStructure::RecreateTopLevel(const VulkanBuffer &instance_buffer,
                                                   const u32 instance_count) {
  ZoneScoped;
  vkDestroyAccelerationStructureKHR(VulkanContext::device, obj, nullptr);
  buffer.Destroy();
  CreateTopLevel(instance_buffer, instance_count);
}

void VulkanAccelerationStructure::CreateBase(const VkAccelerationStructureGeometryKHR &geometry,
                                             const VkAccelerationStructureBuildRangeInfoKHR &offset,
                                             const VkAccelerationStructureTypeKHR type) {
  ZoneScoped;
  std::vector<u32> max_primitive_count(1);
  max_primitive_count[0] = offset.primitiveCount;

  VkAccelerationStructureBuildGeometryInfoKHR build_info{};
  build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  build_info.type = type;
  build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
  build_info.geometryCount = 1;
  build_info.pGeometries = &geometry;

  VkAccelerationStructureBuildSizesInfoKHR size_info{};
  size_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

  vkGetAccelerationStructureBuildSizesKHR(VulkanContext::device,
                                          VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info,
                                          max_primitive_count.data(), &size_info);

  buffer.Create(size_info.accelerationStructureSize,
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

  VkAccelerationStructureCreateInfoKHR as_ci{};
  as_ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
  as_ci.type = type;
  as_ci.size = size_info.accelerationStructureSize;
  as_ci.buffer = buffer.obj;

  vkCreateAccelerationStructureKHR(VulkanContext::device, &as_ci, nullptr, &obj);

  VkPhysicalDeviceAccelerationStructurePropertiesKHR as_props{};
  as_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;

  VkPhysicalDeviceProperties2 props;
  props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props.pNext = &as_props;

  vkGetPhysicalDeviceProperties2(VulkanContext::physical_device, &props);

  VulkanBuffer scratch_buffer = "scratch buffer";
  scratch_buffer.CreateAligned(
      AlignUp(size_info.buildScratchSize, as_props.minAccelerationStructureScratchOffsetAlignment),
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      as_props.minAccelerationStructureScratchOffsetAlignment);

  build_info.dstAccelerationStructure = obj;
  build_info.scratchData.deviceAddress = (VkDeviceAddress)scratch_buffer.device_address;

  std::vector<const VkAccelerationStructureBuildRangeInfoKHR *> range_info = {
      &offset,
  };

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(cmd.obj, &dep);

    vkCmdBuildAccelerationStructuresKHR(cmd.obj, 1, &build_info, range_info.data());
  });
}

void VulkanAccelerationStructure::CreateBottomLevel(const VulkanBuffer &vertex_buffer,
                                                    const VulkanBuffer &index_buffer) {
  ZoneScoped;
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
  triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  triangles.vertexData.deviceAddress = vertex_buffer.device_address;
  triangles.vertexStride = sizeof(Vertex);
  triangles.indexType = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress = index_buffer.device_address;
  triangles.maxVertex = (vertex_buffer.size / sizeof(Vertex)) - 1;

  VkAccelerationStructureGeometryKHR geometry{};
  geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  geometry.flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR | VK_GEOMETRY_OPAQUE_BIT_KHR;
  geometry.geometry.triangles = triangles;

  VkAccelerationStructureBuildRangeInfoKHR offset{};
  offset.primitiveCount = (index_buffer.size / sizeof(Index)) / 3;

  CreateBase(geometry, offset, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);
}

void VulkanAccelerationStructure::CreateTopLevel(const VulkanBuffer &instance_buffer,
                                                 const u32 instance_count) {
  ZoneScoped;
  VkAccelerationStructureGeometryInstancesDataKHR instances{};
  instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  instances.data.deviceAddress = instance_buffer.device_address;

  VkAccelerationStructureGeometryKHR geometry{};
  geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  geometry.geometry.instances = instances;

  VkAccelerationStructureBuildRangeInfoKHR offset{};
  offset.primitiveCount = instance_count;

  CreateBase(geometry, offset, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR);
}
} // namespace Core
