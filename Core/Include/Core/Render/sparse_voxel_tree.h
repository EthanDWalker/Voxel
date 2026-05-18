#pragma once

#include "Vulkan/buffer.h"
#include "Vulkan/descriptors.h"
#include "types.h"
#include <memory>
#include <vector>

namespace Core {

struct SparseVoxelTree {
  static const bool HOST = false;

  static const u32 SENTINAL = 0xFFFFFFFF;

  static const u32 MAX_PAGES = 100'000;
  static const u32 MAX_DEPTH = 5;

  static const u32 PAGE_SIZE_EXP = 17;
  static const u32 PAGE_SIZE = 1 << PAGE_SIZE_EXP;
  static const u32 RADIANCE_PAGE_SIZE_EXP = 9;
  static const u32 BOX_SIZE_EXP = 12;
  constexpr static const f32 MIN_BOUND = -1.0f * f32(1 << (BOX_SIZE_EXP - 1));
  constexpr static const f32 MAX_BOUND = f32(1 << (BOX_SIZE_EXP - 1));
  constexpr static const f32 VOXEL_SIZE = (1 << BOX_SIZE_EXP) / f32(1 << (MAX_DEPTH << 1));

  struct BranchNode {
    u64 child_mask;
    u32 child_ptr;
  };

  // 5 bits r
  // 6 bits g
  // 6 bits b
  // 8 bit phi
  // 8 bit theta
  struct LeafNode {
    u32 data;
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
  };

  std::vector<std::unique_ptr<VulkanBuffer<BufferType::StructuredBuffer, BranchNode>>> branch_pages;
  std::vector<std::unique_ptr<VulkanBuffer<BufferType::StructuredBuffer, LeafNode>>> leaf_pages;
  VulkanDescriptorLayout descriptor_layout;
  VulkanDescriptor descriptor;
  VulkanBuffer<BufferType::StructuredBuffer, TreeHeader> tree_header_buffer = "tree header buffer";
  VulkanBuffer<BufferType::StagingBuffer> tree_header_host_buffer = "tree header host buffer";
  VulkanBuffer<BufferType::StagingBuffer> empty_page_host_buffer = "empty page buffer";

  SparseVoxelTree();

  void AllocateBranchPages(const u32 count);
  void AllocateLeafPages(const u32 count);
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
