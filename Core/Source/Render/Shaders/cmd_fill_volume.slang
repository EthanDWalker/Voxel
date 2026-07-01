#define VOXEL_TREE_SET_BINDING 0
#define VOXEL_TREE_READ_WRITE
#define VOXEL_TREE_ATOMIC
#include "Include/color.slang"
#include "Include/voxel_tree.slang"

struct PushConstants {
  VoxelVolume volume;
  uint32_t depth;
};

void FillNode(const uint32_t3 index, const LeafNode node) {
  uint32_t ptr = GetLocalIndex(index, 0);

  const Material material = material_arr[node.material_index];

  for (uint32_t i = 1;; i++) {
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    const uint32_t local_index = GetLocalIndex(index, i);

    const ChildPtrData child_ptr_data =
        ChildPtrData(voxel_tree[page_index][page_offset].child_ptr_data.load());
    if (node.visibility_mask != 0 && material.emissive != 0.0f) {
      voxel_tree[page_index][page_offset].luminance.add(material.emissive * countbits(node.visibility_mask));
    }

    ptr = child_ptr_data.child_ptr + local_index;
    if (child_ptr_data.is_leaf) {
      break;
    }
  }

  const uint32_t leaf_page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t leaf_page_offset = ptr - (leaf_page_index << voxel_tree_header.page_size_exp);

  voxel_tree_leaf[leaf_page_index][leaf_page_offset] = node;
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

void UpdateVisibility(const uint32_t visibility_index, const uint32_t other_visibility_index,
                      const uint32_t3 index, inout uint32_t visibility_mask) {
  const Optional<uint32_t> leaf_ptr_opt = CheckFilled(index + visibility_direction_arr[visibility_index]);
  if (leaf_ptr_opt.hasValue) {
    const uint32_t leaf_page_index = leaf_ptr_opt.value >> voxel_tree_header.page_size_exp;
    const uint32_t leaf_page_offset =
        leaf_ptr_opt.value - (leaf_page_index << voxel_tree_header.page_size_exp);

    const LeafNode old_node = voxel_tree_leaf[leaf_page_index][leaf_page_offset];

    voxel_tree_leaf[leaf_page_index][leaf_page_offset].visibility_mask &= ~(1 << other_visibility_index);

    if (voxel_tree_leaf[leaf_page_index][leaf_page_offset].visibility_mask != old_node.visibility_mask) {
      AppendLuminanceFromPtrStack(-material_arr[old_node.material_index].emissive);
    }

  } else {
    visibility_mask |= 1 << visibility_index;
  }
}

[shader("compute")]
[numthreads(4, 4, 4)]
void CMain(const uint32_t3 dispatch_id: SV_DispatchThreadID, const uniform PushConstants info) {
  const uint32_t3 index = info.volume.min_tree_index + dispatch_id;
  if (any(index >= grid_n) || !info.volume.Contains(index)) {
    return;
  }

  if (info.depth < voxel_tree_header.max_depth) {
    if (info.depth == voxel_tree_header.max_depth - 1) {
      AllocateNode(index, info.depth, VoxelTreeNodeType::Leaf);
    } else {
      AllocateNode(index, info.depth, VoxelTreeNodeType::Branch);
    }
  } else {
    LeafNode node;
    node.material_index = info.volume.material_index;

    uint32_t visibility = 0;

    uint32_t3 last_index = 0xFFFFFFFF;

    const bool3 min_edge = (index == info.volume.min_tree_index);
    const bool3 max_edge = (index == info.volume.max_tree_index);
    if (any(min_edge || max_edge)) {
      if (min_edge.x) {
        UpdateVisibility(0, 1, index, visibility);
      }
      if (max_edge.x) {
        UpdateVisibility(1, 0, index, visibility);
      }

      if (min_edge.y) {
        UpdateVisibility(2, 3, index, visibility);
      }
      if (max_edge.y) {
        UpdateVisibility(3, 2, index, visibility);
      }

      if (min_edge.z) {
        UpdateVisibility(4, 5, index, visibility);
      }
      if (max_edge.z) {
        UpdateVisibility(5, 4, index, visibility);
      }
    }

    node.visibility_mask = visibility;

    FillNode(index, node);
  }
}
