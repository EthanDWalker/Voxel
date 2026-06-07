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
  uint32_t seed;
};

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

[numthreads(4, 4, 4)]
void CMain(const uint32_t3 dispatch_id: SV_DispatchThreadID, const uniform ChunkAllocationInfo info) {
  const uint32_t3 index = (info.chunk_index << ((voxel_tree_header.max_depth - 1) << 1)) + dispatch_id;
  const float3 world_position = GetWorldPositionCenter(index, voxel_tree_header.max_depth - 1);

  const float noise_height = SampleNoise(info.seed, world_position.xz);

  if (world_position.y > noise_height) {
    return;
  }

  if (info.color == true) {
  } else {
    Allocate(index, info.lod, info.alloc_leaf == true);
  }
}
