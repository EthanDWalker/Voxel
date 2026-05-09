#define VOXEL_TREE_ATOMIC
#define VOXEL_TREE_SET_BINDING 0
#include "Include/voxel_tree.slang"

struct VoxelVolume {
  float3 min;
  uint32_t _p0;
  float3 max;
  uint32_t _p1;
};

void ClearIndexFromPtrStack(const uint32_t3 index, const uint32_t initial_level) {
  for (uint32_t level = initial_level; level > 0; level--) {
    const uint32_t ptr = ray_ptr_stack_mem[memory_index][level - initial_ptr_stack_depth];
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    const uint32_t local_index = GetLocalIndex(index, level);
    const uint64_t clear_mask = ~(uint64_t(1) << local_index);

    const uint64_t child_mask = voxel_tree[page_index][page_offset].child_mask.and(clear_mask);

    if ((child_mask & clear_mask) != 0)
      break;
  }
}

[[numthreads(1, 1, 1)]]
void CMain(const uint32_t3 dispatch_id: SV_DispatchThreadID, const uniform VoxelVolume volume) {
  const uint32_t3 index = GetTreeIndex(volume.min + dispatch_id * voxel_size);

  uint32_t level = 1;
  uint32_t ptr = GetLocalIndex(index, 0);
  BranchNode node;
  bool is_empty;
  uint32_t local_index;
  TraverseTreeFromLevelPtr(ptr, level, node, is_empty, local_index, index);

  if (!is_empty) {
    ClearIndexFromPtrStack(index, level);
  }
}
