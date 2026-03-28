#pragma once

#include "Core/Render/types.h"
#include <filesystem>

namespace Core {
struct MaterialData {
  Vec2u32 albedo_extent;
  u8 *albedo_data;

  void Free() { free(albedo_data); }
};

struct MeshData {
  AABB aabb;
  std::vector<Vertex> vertex_arr;
  std::vector<Index> index_arr;
  MaterialData material;

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
