#include "Include/common.slang"
#include "Include/voxel.slang"
#include "Include/voxelize.slang"

[[vk::binding(0, 0)]]
RWStructuredBuffer<AtomicBranchNode> mesh_node_brick_arr[];

[[vk::binding(1, 0)]]
RWCountedBuffer<AABB> voxel_aabb_arr_arr[];

[[vk::binding(0, 1)]]
RWStructuredBuffer<AtomicPackedLeafNode> leaf_node_page_arr[];

[[vk::binding(1, 1)]]
RWStructuredBuffer<AtomicLeafHeader> leaf_header;

[shader("fragment")]
void FMain(const FragmentInfo info, const uniform AllocateInfo alloc_info) {
  const AABB mesh_aabb = mesh_aabb_arr[alloc_info.mesh_index];
  const uint32_t3 voxel_aabb_index =
      uint32_t3((info.model_position - mesh_aabb_arr[alloc_info.mesh_index].min_bound) / VOXEL_AABB_SIZE);
  const uint32_t3 voxel_aabb_max_index = uint32_t3(
      (mesh_aabb_arr[alloc_info.mesh_index].max_bound - mesh_aabb_arr[alloc_info.mesh_index].min_bound) /
      VOXEL_AABB_SIZE);

  AABB voxel_aabb;
  voxel_aabb.min_bound = mesh_aabb.min_bound + voxel_aabb_index * VOXEL_AABB_SIZE;
  voxel_aabb.max_bound = voxel_aabb.min_bound + VOXEL_AABB_SIZE;

  const uint32_t arr_index = voxel_aabb_index.x + (voxel_aabb_index.y * voxel_aabb_max_index.x) +
                             (voxel_aabb_index.z * voxel_aabb_max_index.x * voxel_aabb_max_index.y);

  const uint32_t3 local_index = uint32_t3((info.model_position - voxel_aabb.min_bound) / VOXEL_SIZE);
  const uint32_t local_index_index = SumComponents(local_index << uint32_t3(0, 2, 4));

  const uint64_t ret =
      mesh_node_brick_arr[alloc_info.mesh_index][arr_index].child_mask.or(uint64_t(1) << local_index_index);

  if (ret == 0) {
    mesh_node_brick_arr[alloc_info.mesh_index][arr_index].child_ptr =
        leaf_header[0].allocated_leaf_count.add(64);
    voxel_aabb_arr_arr[alloc_info.mesh_index].Append(voxel_aabb);
  }
}
