#include "Include/voxelize_terrain.slang"

static const float4 SNOW_COLOR = float4(0.96f, 0.86f, 0.79f, 1.0f);
static const float4 GRASS_COLOR = float4(0.08f, 0.43f, 0.08f, 1.0f);
static const float4 STONE_COLOR = float4(136, 140, 141, 255) / 255.0f;
static const float4 DIRT_COLOR = float4(137, 81, 41, 255) / 255.0f;

float4 GetColor(const TerrainVoxelizeInfo info) {
  const float slant = 1.0f - info.normal.y;

  if (info.world_position.y > 200.0f && slant < 0.2f) {
    return SNOW_COLOR;
  } else if (slant < 0.5f) {
    return GRASS_COLOR;
  } else if (slant > 0.7f) {
    return STONE_COLOR;
  } else {
    return DIRT_COLOR;
  }
}

void AllocateLeaf(const uint32_t3 tree_index, const TerrainVoxelizeInfo info) {
  uint32_t ptr = GetLocalIndex(tree_index, 0);

  for (uint32_t i = 1;; i++) {
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    const uint32_t local_index = GetLocalIndex(tree_index, i);

    const ChildPtrData child_ptr_data =
        ChildPtrData(voxel_tree[page_index][page_offset].child_ptr_data.load());

    if (child_ptr_data.data == SENTINAL)
      return;

    const uint64_t ret = voxel_tree[page_index][page_offset].child_mask.or(uint64_t(1) << local_index);

    ptr = child_ptr_data.child_ptr + local_index;

    if (child_ptr_data.is_leaf)
      break;
  }

  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

  LeafNode leaf_node;
  leaf_node.normal = info.normal;
  leaf_node.color = GetColor(info);

  voxel_tree_leaf[page_index][page_offset].data.max(PackLeafNode(leaf_node));
}

[shader("fragment")]
void FMain(const TerrainVoxelizeInfo info, const uniform TerrainAllocateInfo alloc_info) {
  const uint32_t3 tree_index = GetTreeIndex(info.world_position);

  AllocateLeaf(tree_index, info);
}
