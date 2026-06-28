#define VOXEL_TREE_SET_BINDING 0
#define VOXEL_TREE_READ_WRITE
#define VOXEL_TREE_ATOMIC
#include "Include/color.slang"
#include "Include/voxel_tree.slang"

[[vk::binding(1, 1)]]
RWStructuredBuffer<uint16_t> emissive_voxel_arr;

static const uint32_t SENTINAL = 0xFFFFFFFF;

struct PushConstants {
  VoxelVolume volume;
  uint32_t depth;
};

void Allocate(const uint32_t3 index, const uint32_t depth, const bool leaf) {}

[shader("compute")]
[numthreads(4, 4, 4)]
void CMain(const uint32_t3 dispatch_id: SV_DispatchThreadID, const uniform PushConstants info) {
  const uint32_t3 index = info.volume.min_tree_index + dispatch_id;
  if (any(index >= grid_n) || !info.volume.Contains(index)) {
    return;
  }

  uint32_t ptr = GetLocalIndex(index, 0);

  for (uint32_t i = 1; i < info.depth; i++) {
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

  const uint32_t local_index = GetLocalIndex(index, info.depth);

  const uint64_t child_mask = uint64_t(1) << local_index;
  const uint64_t ret = voxel_tree[page_index][page_offset].child_mask.or(child_mask);

  if (ret == 0) {
    if (info.depth == voxel_tree_header.max_depth - 1) {
      voxel_tree[page_index][page_offset].child_ptr_data =
          voxel_tree_header.AllocateLeafVoxels(64) | IS_LEAF_BIT;
    } else {
      voxel_tree[page_index][page_offset].child_ptr_data = voxel_tree_header.AllocateBranchVoxels(64);
    }
  }

  if (info.depth == voxel_tree_header.max_depth - 1) {
    const bool3 min_edge = index == info.volume.min_tree_index;
    const bool3 max_edge = index == info.volume.max_tree_index - 1;

    const uint32_t3 extent = info.volume.max_tree_index - info.volume.min_tree_index;

    uint32_t emissive_index;
    if (any(min_edge || max_edge) && (ret & child_mask) == 0) {
      if (min_edge.x) {
        emissive_voxel_arr[dispatch_id.y + dispatch_id.z * extent.y] = 1;
      } else if (max_edge.x) {
        emissive_voxel_arr[extent.y * extent.z + dispatch_id.y + dispatch_id.z * extent.y] = 1;
      }

      uint32_t offset = extent.y * extent.z * 2;
      if (min_edge.y) {
        emissive_voxel_arr[offset + dispatch_id.x + dispatch_id.z * extent.x] = 1;
      } else if (max_edge.y) {
        emissive_voxel_arr[offset + extent.x * extent.z + dispatch_id.x + dispatch_id.z * extent.x] = 1;
      }

      offset += extent.x * extent.z * 2;
      if (min_edge.z) {
        emissive_voxel_arr[offset + dispatch_id.x + dispatch_id.y * extent.x] = 1;
      } else if (max_edge.z) {
        emissive_voxel_arr[offset + extent.x * extent.y + dispatch_id.x + dispatch_id.y * extent.x] = 1;
      }
    }
  }
}
