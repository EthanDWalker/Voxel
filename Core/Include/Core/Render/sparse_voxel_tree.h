#pragma once

#include "Core/Util/Parse/gltf.h"
#include "Vulkan/buffer.h"
#include "Vulkan/descriptors.h"
#include "Vulkan/pipeline.h"
#include "types.h"
#include <memory>
#include <vector>

namespace Core {

struct SparseVoxelTree {

  static const bool HOST = false;

  static const u32 SENTINAL = 0xFFFFFFFF;

  static const u32 PAGE_SIZE_EXP = 17;
  static const u32 PAGE_SIZE = 1 << PAGE_SIZE_EXP;
  constexpr static const f32 MIN_BOUND = -2000.0f;
  constexpr static const f32 MAX_BOUND = 2000.0f;

  static const u32 MAX_PAGES = 10000;
  static const u32 MAX_VOXLELIZE_DEPTH = 5;

  struct alignas(GPU_ALIGNMENT) BranchNode {
    u64 child_mask;
    u32 child_ptr;
    u32 radiance;
  };

  struct LeafNode {
    u32 color;
    u32 radiance;
  };

  struct alignas(GPU_ALIGNMENT) TreeHeader {
    // const
    f32 _min_bound;
    f32 _max_bound;
    u32 _max_voxelize_depth;
    u32 _page_size_exp;

    // non-const
    u32 leaf_voxel_count;
    u32 level_page_offset[SparseVoxelTree::MAX_VOXLELIZE_DEPTH - 1];
    u32 level_voxel_count[SparseVoxelTree::MAX_VOXLELIZE_DEPTH - 1];
  };

  std::array<std::vector<std::unique_ptr<VulkanBuffer>>, MAX_VOXLELIZE_DEPTH - 1> pages{};
  std::vector<std::unique_ptr<VulkanBuffer>> leaf_pages;
  std::vector<std::unique_ptr<VulkanImage<ImageType::Planar>>> radiance_pages;
  VulkanDescriptor voxelize_descriptor;
  VulkanDescriptor tree_descriptor;
  VulkanBuffer tree_header_buffer;
  VulkanBuffer tree_header_host_buffer;
  VulkanBuffer empty_page_host_buffer;
  VulkanPipeline<PipelineType::Graphic> allocate_pipeline;
  VulkanPipeline<PipelineType::Graphic> allocate_child_mask_pipeline;
  VulkanSampler sampler;

  SparseVoxelTree();

  void VoxelizeMesh(const MeshData &mesh_data);
};

} // namespace Core
