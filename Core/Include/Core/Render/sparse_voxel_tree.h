#pragma once

#include "Vulkan/descriptors.h"
#include "Vulkan/other_buffer.h"
#include "types.h"
#include <memory>
#include <vector>

namespace Core {

struct SparseVoxelTree {
  static const u32 SENTINAL = 0xFFFFFFFF;

  static const u32 MAX_PAGES = 100'000;
  static const u32 MAX_DEPTH = 5;
  // material index stored as 10 bit unsigned int
  static const u32 MAX_MATERIALS = Min(1024, 1 << 10);

  static const u32 PAGE_SIZE_EXP = 17;
  static const u32 PAGE_SIZE = 1 << PAGE_SIZE_EXP;
  static const u32 BOX_SIZE_EXP = 8;
  static const u32 VOXEL_GRID_SIZE = 1 << (MAX_DEPTH << 1);
  constexpr static const f32 MIN_BOUND = -1.0f * f32(1 << (BOX_SIZE_EXP - 1));
  constexpr static const f32 MAX_BOUND = f32(1 << (BOX_SIZE_EXP - 1));
  constexpr static const f32 VOXEL_SIZE = (1 << BOX_SIZE_EXP) / f32(1 << (MAX_DEPTH << 1));

  struct BranchNode {
    u64 child_mask;
    u32 child_ptr;
  };

  struct LeafNode {
    // 6 bit visibility mask
    // 10 uint material index
    u16 data;
  };

  struct alignas(GPU_ALIGNMENT) TreeHeader {
    // const
    f32 _min_bound = MIN_BOUND;
    u32 _box_size_exp = BOX_SIZE_EXP;
    u32 _max_depth = MAX_DEPTH;
    u32 _page_size_exp = PAGE_SIZE_EXP;

    // non-const
    u32 leaf_count;
    u32 branch_count;
    u32 allocated_leaf_count;
    u32 allocated_branch_count;
  };

  std::vector<std::unique_ptr<VulkanBuffer<BufferType::StructuredBuffer, BranchNode>>> branch_pages;
  std::vector<std::unique_ptr<VulkanBuffer<BufferType::StructuredBuffer, f32>>> branch_luminance_pages;

  std::vector<std::unique_ptr<VulkanBuffer<BufferType::StructuredBuffer, LeafNode>>> leaf_pages;
  VulkanDescriptorLayout descriptor_layout;
  VulkanDescriptor descriptor;
  VulkanBuffer<BufferType::StructuredBuffer, TreeHeader> tree_header_buffer = "tree header buffer";
  VulkanBuffer<BufferType::StagingBuffer> tree_header_host_buffer = "tree header host buffer";
  VulkanBuffer<BufferType::CountedBuffer, Material> material_buffer = "material buffer";
  std::vector<Material> material_arr;

  SparseVoxelTree();

  void ResizeBranch(const u32 count);
  void ResizeLeaf(const u32 count);
};

static constexpr Vec3u32 GetTreeIndex(const Vec3f32 p) {
  const Vec3f32 offset = (p - SparseVoxelTree::MIN_BOUND);
  // clang-format off
  return VecTypeCast<u32>(
    Vec3u32::To<Vec3f32>(
      (
        Vec3f32::To<Vec3u32>(offset) - (SparseVoxelTree::BOX_SIZE_EXP << 23)
      ) 
        + ((SparseVoxelTree::MAX_DEPTH << 1) << 23)
    )
  );
  // clang-format on
}

} // namespace Core
