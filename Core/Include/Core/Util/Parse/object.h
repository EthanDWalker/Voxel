#pragma once

#include "Core/Render/Vulkan/other_buffer.h"
#include "Core/Render/types.h"
#include <memory>

namespace Core {
struct MaterialData {
  std::string name;

  Vec2u32 albedo_extent;
  std::unique_ptr<VulkanBuffer<BufferType::StagingBuffer>> compressed_albedo_data_buffer =
      std::make_unique<VulkanBuffer<BufferType::StagingBuffer>>("compressed albedo staging buffer");
  bool initialized = false;
};

struct MeshData {
  std::string name;

  AABB aabb;

  std::unique_ptr<VulkanBuffer<BufferType::StagingBuffer>> vertex_host_buffer =
      std::make_unique<VulkanBuffer<BufferType::StagingBuffer>>("mesh vertex staging buffer");
  std::unique_ptr<VulkanBuffer<BufferType::StagingBuffer>> index_host_buffer =
      std::make_unique<VulkanBuffer<BufferType::StagingBuffer>>("mesh index staging buffer");

  u32 material_index;
  u32 vertex_count;
  u32 index_count;
};

struct InstanceData {
  Mat4f32 matrix;
  u32 mesh_index;
};

struct ObjectData {
  std::string name;

  std::vector<MeshData> mesh_data_arr;
  std::vector<MaterialData> material_data_arr;
  std::vector<InstanceData> instance_data_arr;
};

struct MeshFileHeader {
  AABB aabb;
  u32 vertex_count;
  u32 index_count;
  u32 material_index;
};

struct MaterialFileHeader {
  Vec2u32 extent;
  u32 image_data_size;
};

struct ObjectFolderHeader {
  u32 material_descriptor_count;
  u32 mesh_descriptor_count;
  u32 instance_data_count;
};

void WriteObjectFolder(const std::filesystem::path &folder_path, const ObjectData &object_data);
void ReadObjectFolder(const std::filesystem::path &folder_path, ObjectData &object_data);
} // namespace Core
