#pragma once

#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/types.h"
#include <filesystem>
#include <memory>

namespace Core {
struct MaterialData {
  Vec2u32 albedo_extent;
  u8 *albedo_data;

  void Free() { free(albedo_data); }
};

struct MeshData {
  AABB aabb;
  std::unique_ptr<VulkanBuffer> vertex_host_buffer =
      std::make_unique<VulkanBuffer>("mesh vertex host buffer");
  std::unique_ptr<VulkanBuffer> index_host_buffer = std::make_unique<VulkanBuffer>("mesh index host buffer");
  MaterialData material;
  u32 vertex_count;
  u32 index_count;

  void Free() { material.Free(); }
};

struct MeshFileData {
  MeshFileData() = default;

  MeshFileData(const MeshFileData &) = delete;
  MeshFileData &operator=(const MeshFileData &) = delete;

  MeshFileData(MeshFileData &&) = default;
  MeshFileData &operator=(MeshFileData &&) = default;

  std::vector<MeshData> mesh_data_arr;

  ~MeshFileData() {
    for (auto &mesh_data : mesh_data_arr) {
      mesh_data.Free();
    }
  }
};

void ParseGlbFile(const std::filesystem::path &file_path, MeshFileData &mesh_file_data);
} // namespace Core
