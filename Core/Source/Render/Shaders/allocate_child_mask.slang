#include "Include/common.slang"
#define VOXEL_TREE_SET_BINDING 1
#define VOXEL_TREE_ATOMIC
#define VOXEL_TREE_READ_WRITE
#include "Include/voxel_tree.slang"

[[vk::binding(4, 0)]]
Texture2D<float4> albedo_image_arr[];

[[vk::binding(5, 0)]]
SamplerState albedo_image_sampler;

[shader("fragment")]
void FMain(const VoxelizeInfo info, const uniform AllocateInfo alloc_info) {
  const float4 color = albedo_image_arr[alloc_info.mesh_id].SampleLevel(albedo_image_sampler, info.uv, 0.0f);
  if (color.a == 0.0f)
    return;

  const uint32_t3 tree_index = GetTreeIndex(info.world_position);
  uint32_t ptr = GetLocalIndex(tree_index, 0);

  for (uint32_t i = 1;; i++) {
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    const uint32_t local_index = GetLocalIndex(tree_index, i);

    const ChildPtrData child_ptr_data =
        ChildPtrData(voxel_tree[page_index][page_offset].child_ptr_data.load());

    if (child_ptr_data.data == SENTINAL)
      break;

    voxel_tree[page_index][page_offset].child_mask.or(uint64_t(1) << local_index);

    ptr = child_ptr_data.child_ptr + local_index;

    if (child_ptr_data.is_leaf)
      break;
  }

  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

  LeafNode leaf_node;
  leaf_node.normal = info.normal;
  leaf_node.color = color;

  voxel_tree_leaf[page_index][page_offset].data = PackLeafNode(leaf_node);
}
