#pragma once

#include "Core/Util/Parse/gltf.h"
#include "Vulkan/buffer.h"
#include "Vulkan/descriptors.h"
#include "Vulkan/pipeline.h"
#include <memory>
#include <vector>

namespace Core {

struct alignas(GPU_ALIGNMENT) TreeHeader {
  u32 voxel_count;
  u32 allocated_page_count;
  u32 _page_size;
  f32 _min_bound;
  f32 _max_bound;
  u32 _max_voxelize_depth;
  u32 far_ptr_count;
  u32 allocated_far_ptr_page_count;
  u32 _far_ptr_page_size;
  u32 _far_ptr_page_size_exp;
  u32 _page_size_exp;
  u32 notifications;
};

struct SparseVoxelTree {
  struct Voxel {
    u32 data;
  };

  static const bool HOST = false;
  static const u32 PAGE_SIZE_EXP = 16;
  static const u32 PAGE_SIZE = Pow(2, PAGE_SIZE_EXP);
  static const u32 FAR_PTR_PAGE_SIZE_EXP = 16;
  static const u32 FAR_PTR_PAGE_SIZE = Pow(2, FAR_PTR_PAGE_SIZE_EXP);
  static const u32 MAX_PAGES = 10000;
  static const u32 MAX_VOXLELIZE_DEPTH = 10;
  constexpr static const f32 MIN_BOUND = -2000.0f;
  constexpr static const f32 MAX_BOUND = 2000.0f;

  std::vector<std::unique_ptr<VulkanBuffer>> pages;
  std::vector<std::unique_ptr<VulkanBuffer>> far_ptr_pages;
  VulkanDescriptor voxelize_descriptor;
  VulkanDescriptor tree_descriptor;
  VulkanBuffer tree_header_buffer;
  VulkanBuffer tree_header_host_buffer;
  VulkanPipeline<PipelineType::Compute> voxelize_pipeline;
  VulkanSampler sampler;
  const TreeHeader *header = nullptr;

  SparseVoxelTree();

  void VoxelizeMesh(const MeshData &mesh_data);
};
} // namespace Core
