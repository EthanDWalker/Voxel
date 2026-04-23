#include "Include/common.slang"
#define VOXEL_TREE_ATOMIC
#define VOXEL_TREE_READ_WRITE
#define VOXEL_TREE_SET_BINDING 1
#include "Include/voxel_tree.slang"

void Allocate(const uint32_t3 index, const uint32_t depth) {
  uint32_t ptr = GetLocalIndex(index, 0);

  for (uint32_t i = 1; i < depth; i++) {
    const uint32_t page_level_offset = voxel_tree_header.GetLevelPageOffset(i - 1);
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    const uint32_t local_index = GetLocalIndex(index, i);

    const uint32_t child_ptr = voxel_tree[page_index + page_level_offset][page_offset].child_ptr.load();

    ptr = child_ptr + local_index;
  }

  const uint32_t page_level_offset = voxel_tree_header.GetLevelPageOffset(depth - 1);
  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;

  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

  if (voxel_tree[page_index + page_level_offset][page_offset].child_ptr.compareExchange(SENTINAL, 0) ==
      SENTINAL) {
    voxel_tree[page_index + page_level_offset][page_offset].child_radiance_ptr =
        voxel_tree_header.AllocateRadianceBrick();
    if (depth == voxel_tree_header.max_depth - 1) {
      voxel_tree[page_index + page_level_offset][page_offset].child_ptr =
          voxel_tree_header.AllocateLeafVoxels(64);
    } else {
      voxel_tree[page_index + page_level_offset][page_offset].child_ptr =
          voxel_tree_header.AllocateVoxels(depth, 64);
    }
  }
}

[shader("fragment")]
void FMain(const VoxelizeInfo info, const uniform uint32_t depth) {
  Allocate(GetTreeIndex(info.world_position), depth);
}
