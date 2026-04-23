#include "Include/common.slang"
#define VOXEL_TREE_SET_BINDING 1
#define VOXEL_TREE_ATOMIC
#define VOXEL_TREE_READ_WRITE
#include "Include/voxel_tree.slang"

[[vk::binding(1, 0)]]
Texture2D<float4> albedo_texture;
[[vk::binding(2, 0)]]
SamplerState sampler;

[shader("fragment")]
void FMain(const VoxelizeInfo info) {
  const uint32_t3 tree_index = GetTreeIndex(info.world_position);
  uint32_t ptr = GetLocalIndex(tree_index, 0);
  uint32_t local_index;

  for (uint32_t i = 1; i < voxel_tree_header.max_depth; i++) {
    const uint32_t level_page_offset = voxel_tree_header.GetLevelPageOffset(i - 1);
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    local_index = GetLocalIndex(tree_index, i);

    voxel_tree[page_index + level_page_offset][page_offset].child_mask.or(uint64_t(1) << local_index);

    ptr = voxel_tree[page_index + level_page_offset][page_offset].child_ptr.load() + local_index;
  }

  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);
  const float4 color = albedo_texture.SampleLevel(sampler, info.uv, 0.0f);
  voxel_tree_leaf[page_index][page_offset].color.max(PackRGBA8(color));
}
