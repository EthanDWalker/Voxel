#include "Include/common.slang"
#include "Include/voxel.slang"
#include "Include/voxelize.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<BranchNode> mesh_node_brick_arr[];

[[vk::binding(1, 0)]]
RWCountedBuffer<AABB> voxel_aabb_arr_arr[];

[shader("compute")]
[numthreads(64, 1, 1)]
void CMain(const uint32_t voxel_aabb_index: SV_DispatchThreadID, const uniform AllocateInfo alloc_info) {
  if (voxel_aabb_index >= voxel_aabb_arr_arr[alloc_info.mesh_index].count)
    return;

  const AABB voxel_aabb = voxel_aabb_arr_arr[alloc_info.mesh_index][voxel_aabb_index];

  const uint32_t3 voxel_aabb_brick_index =
      uint32_t3((voxel_aabb.min_bound - mesh_aabb_arr[alloc_info.mesh_index].min_bound) / VOXEL_AABB_SIZE);
  const uint32_t3 voxel_aabb_max_brick_index = uint32_t3(
      (mesh_aabb_arr[alloc_info.mesh_index].max_bound - mesh_aabb_arr[alloc_info.mesh_index].min_bound) /
      VOXEL_AABB_SIZE);

  const uint32_t voxel_aabb_arr_index =
      voxel_aabb_brick_index.x + (voxel_aabb_brick_index.y * voxel_aabb_max_brick_index.x) +
      (voxel_aabb_brick_index.z * voxel_aabb_max_brick_index.x * voxel_aabb_max_brick_index.y);

  const uint64_t child_mask = mesh_node_brick_arr[alloc_info.mesh_index][voxel_aabb_arr_index].child_mask;

  AABB aabb;
  aabb.min_bound = 1e6f;
  aabb.max_bound = -1e6f;

  for (uint32_t i = 0; i < 64; i++) {
    const bool empty = ((child_mask & (uint64_t(1) << i)) == 0);
    if (empty) {
      continue;
    }

    const uint32_t3 local_index = ((i >> uint32_t3(0, 2, 4)) & uint32_t3(0x3));
    aabb.min_bound = min(aabb.min_bound, voxel_aabb.min_bound + VOXEL_SIZE * local_index);
    aabb.max_bound = max(aabb.max_bound, voxel_aabb.min_bound + VOXEL_SIZE * (local_index + 1));
  }

  voxel_aabb_arr_arr[alloc_info.mesh_index].Store(voxel_aabb_index, aabb);
}
