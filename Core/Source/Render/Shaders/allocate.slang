#include "Include/common.slang"
#define VOXEL_TREE_ATOMIC
#define VOXEL_TREE_READ_WRITE
#define VOXEL_TREE_SET_BINDING 1
#include "Include/voxel_tree.slang"

[[vk::binding(4, 0)]]
Texture2D<float4> albedo_image_arr[];

[[vk::binding(5, 0)]]
SamplerState albedo_image_sampler;

void Allocate(const uint32_t3 index, const uint32_t depth, const bool leaf) {
  uint32_t ptr = GetLocalIndex(index, 0);

  for (uint32_t i = 1; i < depth; i++) {
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    const uint32_t local_index = GetLocalIndex(index, i);

    const ChildPtrData child_ptr_data =
        ChildPtrData(voxel_tree[page_index][page_offset].child_ptr_data.load());

    if (child_ptr_data.is_leaf) {
      break;
    } else {
      ptr = child_ptr_data.child_ptr + local_index;
    }
  }

  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

  if (voxel_tree[page_index][page_offset].child_ptr_data.compareExchange(SENTINAL, 0) == SENTINAL) {
    if (leaf) {
      voxel_tree[page_index][page_offset].child_ptr_data =
          voxel_tree_header.AllocateLeafVoxels(64) | IS_LEAF_BIT;
    } else {
      voxel_tree[page_index][page_offset].child_ptr_data = voxel_tree_header.AllocateBranchVoxels(64);
    }
  }
}

[shader("fragment")]
void FMain(const VoxelizeInfo info, const uniform AllocateInfo alloc_info) {
  const uint32_t3 tree_index = GetTreeIndex(info.world_position);
  Allocate(tree_index, alloc_info.depth, alloc_info.leaf == 1);
}
