#define VOXEL_TREE_ATOMIC
#define VOXEL_TREE_SET_BINDING 0
#include "Include/voxel_tree.slang"

struct VoxelVolume {
  float3 min;
  uint32_t _p0;
  float3 max;
  uint32_t _p1;
};

static const int32_t3 offset_masks[26] = {
  // top
  int32_t3(0, 1, 0),
  int32_t3(1, 1, 0),
  int32_t3(-1, 1, 0),
  int32_t3(0, 1, 1),
  int32_t3(0, 1, -1),

  int32_t3(1, 1, -1),
  int32_t3(1, 1, 1),
  int32_t3(-1, 1, -1),
  int32_t3(-1, 1, 1),

  // middle (no (0, 0, 0))
  int32_t3(1, 0, 0),
  int32_t3(-1, 0, 0),
  int32_t3(0, 0, 1),
  int32_t3(0, 0, -1),

  int32_t3(1, 0, -1),
  int32_t3(1, 0, 1),
  int32_t3(-1, 0, -1),
  int32_t3(-1, 0, 1),

  // bottom
  int32_t3(0, -1, 0),
  int32_t3(1, -1, 0),
  int32_t3(-1, -1, 0),
  int32_t3(0, -1, 1),
  int32_t3(0, -1, -1),

  int32_t3(1, -1, -1),
  int32_t3(1, -1, 1),
  int32_t3(-1, -1, -1),
  int32_t3(-1, -1, 1),
};

void ClearIndexFromPtrStack(const uint32_t3 index, const uint32_t initial_level) {
  for (uint32_t level = initial_level; level >= initial_ptr_stack_depth; level--) {
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

void UpdateLeafNormalFromClearedVolume(const uint32_t3 tree_index, const uint32_t3 min_tree_index,
                                       const uint32_t3 max_tree_index) {
  uint32_t local_index = GetLocalIndex(tree_index, 0);
  uint32_t ptr = local_index;
  uint32_t level = 1;
  bool is_empty = false;
  BranchNode node;

  TraverseTreeFromLevelPtr(ptr, level, node, is_empty, local_index, tree_index);

  if (is_empty) {
    return;
  }

  ptr = node.child_ptr_data.child_ptr + local_index;

  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);
  LeafNode leaf_node = UnpackLeafNode(voxel_tree_leaf[page_index][page_offset].data.load());
  leaf_node.normal = 0.0f;

  for (int32_t x = -1; x <= 1; x++) {
    for (int32_t y = -1; y <= 1; y++) {
      for (int32_t z = -1; z <= 1; z++) {
        const int32_t3 offset = int32_t3(x, y, z);
        if (all(offset == 0))
          continue;

        const int32_t3 current_index = int32_t3(tree_index) + offset;

        if (any(current_index < 0) || any(current_index >= grid_n) ||
            all(current_index >= min_tree_index && current_index < max_tree_index)) {
          is_empty = true;
        } else {
          level = 1;
          ptr = GetLocalIndex(current_index, 0);

          TraverseTreeFromLevelPtr(ptr, level, node, is_empty, local_index, current_index);
        }

        if (is_empty) {
          leaf_node.normal += normalize(offset);
        }
      }
    }
  }

  leaf_node.normal = all(leaf_node.normal == 0.0f) ? float3(0.0f, 1.0f, 0.0f) : normalize(leaf_node.normal);

  voxel_tree_leaf[page_index][page_offset].data.store(PackLeafNode(leaf_node));
}

[[numthreads(1, 1, 1)]]
void CMain(const uint32_t3 dispatch_id: SV_DispatchThreadID, const uniform VoxelVolume volume) {
  const float3 position = volume.min + voxel_size * dispatch_id;
  if (any(position >= (voxel_tree_header.min_bound + box_size))) {
    return;
  }

  const uint32_t3 index_min = GetTreeIndex(volume.min);
  const uint32_t3 index_max = GetTreeIndex(volume.max);
  const uint32_t3 index = index_min + dispatch_id;

  uint32_t level = 1;
  uint32_t ptr = GetLocalIndex(index, 0);
  BranchNode node;
  bool is_empty;
  uint32_t local_index;
  TraverseTreeFromLevelPtr(ptr, level, node, is_empty, local_index, index);

  const bool3 min_edge = index == index_min;
  const bool3 max_edge = index == (index_max - 1);

  if (!is_empty) {
    ClearIndexFromPtrStack(index, level);

    if (any(min_edge || max_edge)) {
      for (uint32_t i = 0; i < 26; i++) {
        const int32_t3 offseted_index = int32_t3(index) + offset_masks[i];
        if (any(offseted_index < 0) || any(offseted_index >= grid_n)) {
          continue;
        }
        UpdateLeafNormalFromClearedVolume(offseted_index, index_min, index_max);
      }
    }
  }
}
