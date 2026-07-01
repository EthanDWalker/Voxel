#define VOXEL_TREE_ATOMIC
#define VOXEL_TREE_READ_WRITE
#define VOXEL_TREE_SET_BINDING 0
#include "Include/noise.slang"
#include "Include/voxel_tree.slang"

struct ChunkAllocationInfo {
  uint32_t3 chunk_index;
  uint32_t lod;
  uint32_t alloc_leaf;
  uint32_t color;
  int32_t seed;
  uint32_t max_depth;
};

void AllocateLeaf(const uint32_t3 tree_index, const LeafNode leaf_node) {
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
  voxel_tree_leaf[page_index][page_offset] = leaf_node;
}

[numthreads(4, 4, 4)]
void CMain(const int32_t3 dispatch_id: SV_DispatchThreadID, const uniform ChunkAllocationInfo info) {
  const uint32_t max_depth_shift = (voxel_tree_header.max_depth - info.max_depth) << 1;
  const float max_depth_voxel_size = FloatMultiplyBy2ToN(voxel_size, max_depth_shift).x;
  const uint32_t3 index =
      (info.chunk_index << ((voxel_tree_header.max_depth - 1) << 1)) + (dispatch_id << max_depth_shift);
  const float3 world_position = GetWorldPositionCenter(index, info.max_depth - 1);

  if (any(index == 0 || index == grid_n - 1)) {
    return;
  }

  VoxelVolume volume;
  volume.min_tree_index = 50;
  volume.max_tree_index = grid_n - 51;

  if (volume.Contains(index)) {
    return;
  }

  if (info.color == true) {
    LeafNode node = {};
    node.material_index = 1;
    uint32_t visibility_mask = 0;
    for (uint32_t i = 0; i < VISIBILITY_DIRECTION_COUNT; i++) {
      const uint32_t3 offseted_index = visibility_direction_arr[i] + index;
      if (volume.Contains(offseted_index) || any(offseted_index == 0 || offseted_index == grid_n - 1)) {
        visibility_mask |= 1 << i;
      }
    }
    node.visibility_mask = visibility_mask;

    AllocateLeaf(index, node);
  } else {
    if (info.alloc_leaf == true) {
      AllocateNode(index, info.lod, VoxelTreeNodeType::Leaf);
    } else if (WaveIsFirstLane()) {
      AllocateNode(index, info.lod, VoxelTreeNodeType::Branch);
    }
  }
}
