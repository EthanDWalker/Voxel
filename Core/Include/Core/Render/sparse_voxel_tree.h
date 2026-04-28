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

  static const u32 MAX_PAGES = 100000;
  static const u32 MAX_VOXLELIZE_DEPTH = 5;

  static const u32 PAGE_SIZE_EXP = 17;
  static const u32 PAGE_SIZE = 1 << PAGE_SIZE_EXP;
  static const u32 RADIANCE_PAGE_SIZE_EXP = 9;
  static const u32 BOX_SIZE_EXP = 12;
  constexpr static const f32 MIN_BOUND = -1.0f * f32(1 << (BOX_SIZE_EXP - 1));
  constexpr static const f32 MAX_BOUND = f32(1 << (BOX_SIZE_EXP - 1));
  constexpr static const f32 VOXEL_SIZE = (1 << BOX_SIZE_EXP) / f32(1 << (MAX_VOXLELIZE_DEPTH << 1));

  struct BranchNode {
    u64 child_mask;
    u32 child_ptr;
  };

  struct LeafNode {
    u32 color;
    u32 normal;
  };

  struct alignas(GPU_ALIGNMENT) TreeHeader {
    // const
    f32 _min_bound = MIN_BOUND;
    u32 _box_size_exp = BOX_SIZE_EXP;
    u32 _max_voxelize_depth = MAX_VOXLELIZE_DEPTH;
    u32 _page_size_exp = PAGE_SIZE_EXP;

    // non-const
    u32 leaf_voxel_count;
    u32 level_page_offset[SparseVoxelTree::MAX_VOXLELIZE_DEPTH - 1];
    u32 level_voxel_count[SparseVoxelTree::MAX_VOXLELIZE_DEPTH - 1];
  };

  std::array<std::vector<std::unique_ptr<VulkanBuffer>>, MAX_VOXLELIZE_DEPTH - 1> pages{};
  std::vector<std::unique_ptr<VulkanBuffer>> leaf_pages;
  VulkanDescriptor voxelize_descriptor;
  VulkanDescriptor tree_descriptor;
  VulkanBuffer tree_header_buffer;
  VulkanBuffer tree_header_host_buffer;
  VulkanBuffer empty_page_host_buffer;
  VulkanPipeline<PipelineType::Graphic> allocate_pipeline;
  VulkanPipeline<PipelineType::Graphic> allocate_child_mask_pipeline;
  VulkanPipeline<PipelineType::Compute> mip_map_pipeline;
  VulkanSampler texture_sampler;

  SparseVoxelTree();

  void VoxelizeMesh(const MeshData &mesh_data);
};

} // namespace Core
