#define VOXEL_TREE_ATOMIC
#define VOXEL_TREE_SET_BINDING 0
#include "Include/color.slang"
#include "Include/voxel_tree.slang"

void ClearIndexFromPtrStack(const uint32_t3 index, const uint32_t initial_level) {
  for (uint32_t level = initial_level; level >= initial_ptr_stack_depth; level--) {
    const uint32_t ptr = ptr_stack_mem[memory_index][level - initial_ptr_stack_depth];
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    const uint32_t local_index = GetLocalIndex(index, level);
    const uint64_t clear_mask = ~(uint64_t(1) << local_index);

    const uint64_t child_mask = voxel_tree[page_index][page_offset].child_mask.and(clear_mask);

    if ((child_mask & clear_mask) != 0)
      break;
  }
}

void AppendLuminanceFromPtrStack(const float luminance) {
  if (luminance == 0.0f) {
    return;
  }

  for (uint32_t i = 0; i < PTR_STACK_SIZE; i++) {
    const uint32_t ptr = ptr_stack_mem[memory_index][i];
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    voxel_tree[page_index][page_offset].luminance.add(luminance);
  }
}

Optional<uint32_t> CheckFilled(const uint32_t3 index) {
  uint32_t ptr = GetLocalIndex(index, 0);

  for (uint32_t i = 1;; i++) {
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    const uint32_t local_index = GetLocalIndex(index, i);

    const uint64_t child_mask = voxel_tree[page_index][page_offset].child_mask.load();

    if ((child_mask & (uint64_t(1) << local_index)) == 0) {
      return none;
    }

    ptr_stack_mem[memory_index][i - 1] = ptr;

    const ChildPtrData child_ptr_data =
        ChildPtrData(voxel_tree[page_index][page_offset].child_ptr_data.load());

    ptr = child_ptr_data.child_ptr + local_index;
    if (child_ptr_data.is_leaf) {
      break;
    }
  }

  return ptr;
}

void UpdateVisibility(const uint32_t visibility_index, const uint32_t other_visiblity_index,
                      const uint32_t3 index) {
  const Optional<uint32_t> leaf_ptr_opt = CheckFilled(index + visibility_direction_arr[visibility_index]);
  if (leaf_ptr_opt.hasValue) {
    const uint32_t leaf_page_index = leaf_ptr_opt.value >> voxel_tree_header.page_size_exp;
    const uint32_t leaf_page_offset =
        leaf_ptr_opt.value - (leaf_page_index << voxel_tree_header.page_size_exp);

    const LeafNode old_node = voxel_tree_leaf[leaf_page_index][leaf_page_offset];

    voxel_tree_leaf[leaf_page_index][leaf_page_offset].visibility_mask |= 1 << other_visiblity_index;

    if (old_node.visibility_mask == 0) {
      AppendLuminanceFromPtrStack(material_arr[old_node.material_index].emissive);
    }
  }
}

[[numthreads(4, 4, 4)]]
void CMain(const uint32_t3 dispatch_id: SV_DispatchThreadID, const uniform VoxelVolume volume) {
  const uint32_t3 index = volume.min_tree_index + dispatch_id;
  if (any(index >= grid_n) || !volume.Contains(index)) {
    return;
  }

  uint32_t level = 1;
  uint32_t ptr = GetLocalIndex(index, 0);
  BranchNode node;
  bool is_empty;
  uint32_t local_index;
  TraverseTreeFromLevelPtr(ptr, level, node, is_empty, local_index, index);

  if (!is_empty) {
    ClearIndexFromPtrStack(index, level);

    ptr = node.child_ptr_data.child_ptr + local_index;
    const uint32_t leaf_page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t leaf_page_offset = ptr - (leaf_page_index << voxel_tree_header.page_size_exp);
    const LeafNode leaf_node = voxel_tree_leaf[leaf_page_index][leaf_page_offset];
    voxel_tree_leaf[leaf_page_index][leaf_page_offset].data = 0;

    if (leaf_node.visibility_mask != 0) {
      AppendLuminanceFromPtrStack(-material_arr[leaf_node.material_index].emissive);
    }

    const bool3 min_edge = index == volume.min_tree_index;
    const bool3 max_edge = index == volume.max_tree_index;

    if (any(min_edge || max_edge)) {
      if (min_edge.x) {
        UpdateVisibility(0, 1, index);
      }
      if (max_edge.x) {
        UpdateVisibility(1, 0, index);
      }

      if (min_edge.y) {
        UpdateVisibility(2, 3, index);
      }
      if (max_edge.y) {
        UpdateVisibility(3, 2, index);
      }

      if (min_edge.z) {
        UpdateVisibility(4, 5, index);
      }
      if (max_edge.z) {
        UpdateVisibility(5, 4, index);
      }
    }
  }
}
