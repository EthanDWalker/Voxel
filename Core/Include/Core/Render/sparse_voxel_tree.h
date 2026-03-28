#pragma once

#include "Core/Util/Parse/gltf.h"
#include "Vulkan/buffer.h"
#include "Vulkan/descriptors.h"
#include "Vulkan/pipeline.h"
#include <memory>
#include <vector>

namespace Core {

struct SparseVoxelTree {
  struct Voxel {
    u32 data;
  };

  static const u32 PAGE_SIZE = 4000 * 8;
  static const u32 MAX_PAGES = 10000;
  static const u32 MAX_VOXLELIZE_DEPTH = 9;
  constexpr static const f32 MIN_BOUND = -2000.0f;
  constexpr static const f32 MAX_BOUND = 2000.0f;

  std::vector<std::unique_ptr<VulkanBuffer>> pages;
  VulkanDescriptor voxelize_descriptor;
  VulkanDescriptor tree_descriptor;
  VulkanBuffer tree_header_buffer;
  VulkanBuffer tree_header_host_buffer;
  VulkanSampler sampler;
  VulkanPipeline<PipelineType::Compute> voxelize_pipeline;

  SparseVoxelTree();

  void VoxelizeMesh(const MeshData &mesh_data);
};
} // namespace Core
