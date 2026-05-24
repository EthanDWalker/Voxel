#pragma once

#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/Vulkan/other_buffer.h"
#include "Core/Render/types.h"
#include "volk.h"

namespace Core {

enum class AccelerationStructureType {
  TopLevel,
  AABB,
  Triangle,
};

struct BaseVulkanAccelerationStructure {
  BaseVulkanAccelerationStructure() = default;

  BaseVulkanAccelerationStructure(const BaseVulkanAccelerationStructure &) = delete;
  BaseVulkanAccelerationStructure &operator=(const BaseVulkanAccelerationStructure &) = delete;

  BaseVulkanAccelerationStructure(BaseVulkanAccelerationStructure &&) = default;
  BaseVulkanAccelerationStructure &operator=(BaseVulkanAccelerationStructure &&) = delete;

  VkAccelerationStructureKHR obj;
  BaseVulkanBuffer buffer = "acceleration structure buffer";

  ~BaseVulkanAccelerationStructure();

  void CreateBase(const VkAccelerationStructureGeometryKHR &geometry,
                  const VkAccelerationStructureBuildRangeInfoKHR &offset,
                  const VkAccelerationStructureTypeKHR type);
  void DestroyBase();
};

template <AccelerationStructureType Type> struct VulkanAccelerationStructure {};

template <>
struct VulkanAccelerationStructure<AccelerationStructureType::TopLevel> : BaseVulkanAccelerationStructure {
  using BaseVulkanAccelerationStructure::BaseVulkanAccelerationStructure;

  void Create(const VulkanBuffer<BufferType::CountedBuffer, Instance> &instance_buffer) {
    Assert((instance_buffer.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0,
           "must have VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT set for buffer {}", instance_buffer.name);

    VkAccelerationStructureGeometryInstancesDataKHR instances{};
    instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances.data.deviceAddress = instance_buffer.device_address + 16;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.instances = instances;

    VkAccelerationStructureBuildRangeInfoKHR offset{};
    offset.primitiveCount = instance_buffer.cpu_append_count;

    CreateBase(geometry, offset, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR);
  }

  void Recreate(const VulkanBuffer<BufferType::CountedBuffer, Instance> &instance_buffer) {
    DestroyBase();
    Create(instance_buffer);
  }
};

template <>
struct VulkanAccelerationStructure<AccelerationStructureType::AABB> : BaseVulkanAccelerationStructure {
  using BaseVulkanAccelerationStructure::BaseVulkanAccelerationStructure;

  void Create(const VulkanBuffer<BufferType::CountedBuffer, Mesh> &mesh_buffer) {

    Assert((mesh_buffer.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0,
           "must have VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT set for buffer {}", mesh_buffer.name);

    VkAccelerationStructureGeometryAabbsDataKHR aabbs{};
    aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    aabbs.data.deviceAddress = mesh_buffer.device_address;
    aabbs.stride = sizeof(Mesh);

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.aabbs = aabbs;

    VkAccelerationStructureBuildRangeInfoKHR offset{};
    offset.primitiveCount = mesh_buffer.cpu_append_count;
    offset.primitiveOffset = sizeof(VulkanBuffer<BufferType::CountedBuffer>::Header);

    CreateBase(geometry, offset, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);
  }

  void Recreate(const VulkanBuffer<BufferType::CountedBuffer, Mesh> &mesh_buffer) {
    DestroyBase();
    Create(mesh_buffer);
  }
};

template <>
struct VulkanAccelerationStructure<AccelerationStructureType::Triangle> : BaseVulkanAccelerationStructure {
  using BaseVulkanAccelerationStructure::BaseVulkanAccelerationStructure;

  void Create(const VulkanBuffer<BufferType::StructuredBuffer, Vertex> &vertex_buffer,
              const VulkanBuffer<BufferType::StructuredBuffer, Index> &index_buffer) {

    Assert((vertex_buffer.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0,
           "must have VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT set for buffer {}", vertex_buffer.name);
    Assert((index_buffer.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0,
           "must have VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT set for buffer {}", index_buffer.name);

    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R16G16B16_SFLOAT;
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

  void Recreate(const VulkanBuffer<BufferType::StructuredBuffer, Vertex> &vertex_buffer,
                const VulkanBuffer<BufferType::StructuredBuffer, Index> &index_buffer) {
    DestroyBase();
    Create(vertex_buffer, index_buffer);
  }
};

} // namespace Core
