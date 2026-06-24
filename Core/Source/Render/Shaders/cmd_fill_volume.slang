#define VOXEL_TREE_SET_BINDING 0
#define VOXEL_TREE_READ_WRITE
#define VOXEL_TREE_ATOMIC
#include "Include/voxel_tree.slang"

static const uint32_t SENTINAL = 0xFFFFFFFF;

[shader("compute")]
[numthreads(1, 1, 1)]
void CMain(const uint32_t3 dispatch_id: SV_DispatchThreadID, const uniform VoxelVolume volume) {
  const uint32_t3 index = volume.min_tree_index + dispatch_id;
  if (any(index >= grid_n)) {
    return;
  }

  uint32_t level = 1;
  uint32_t local_index = GetLocalIndex(index, 0);
  uint32_t ptr = local_index;

  for (;; level++) {
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    local_index = GetLocalIndex(index, level);

    ptr = voxel_tree[page_index][page_offset].child_ptr_data.compareExchange(0, SENTINAL);

    if (ptr == 0) {
      if (level == voxel_tree_header.max_depth - 1) {
        voxel_tree[page_index][page_offset].child_ptr_data =
            voxel_tree_header.AllocateLeafVoxels(64) | IS_LEAF_BIT;
      } else {
        voxel_tree[page_index][page_offset].child_ptr_data = voxel_tree_header.AllocateBranchVoxels(64);
      }
      return;
    } else if (ptr == SENTINAL) {
      return;
    }

    const uint64_t child_bit = uint64_t(1) << local_index;
    voxel_tree[page_index][page_offset].child_mask.or(child_bit);

    const ChildPtrData child_ptr_data =
        (ChildPtrData)voxel_tree[page_index][page_offset].child_ptr_data.load();

    ptr = child_ptr_data.child_ptr + local_index;

    if (child_ptr_data.is_leaf) {
      break;
    }
  }

  const uint32_t leaf_page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t leaf_page_offset = ptr - (leaf_page_index << voxel_tree_header.page_size_exp);

  LeafNode node;
  node.material_index = (uint16_t)volume.material_index;
  voxel_tree_leaf[leaf_page_index][leaf_page_offset] = node;
}
