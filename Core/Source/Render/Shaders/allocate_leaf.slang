#include "Include/common.slang"
#include "Include/voxel.slang"
#include "Include/voxelize.slang"

[[vk::binding(0, 0)]]
RWStructuredBuffer<BranchNode> mesh_node_brick_arr[];

[[vk::binding(0, 1)]]
RWStructuredBuffer<AtomicPackedLeafNode> leaf_node_page_arr[];

[[vk::binding(5, 0)]]
Texture2D<float4> albedo_image_arr[];

[[vk::binding(6, 0)]]
SamplerState albedo_image_sampler;

[[vk::binding(1, 1)]]
StructuredBuffer<LeafHeader> leaf_header;

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

  const uint32_t child_ptr =
      mesh_node_brick_arr[alloc_info.mesh_index][arr_index].child_ptr + local_index_index;
  const uint32_t page_index = child_ptr >> leaf_header[0].page_size_exp;
  const uint32_t page_offset = child_ptr - (page_index << leaf_header[0].page_size_exp);

  LeafNode leaf_node;
  leaf_node.normal = info.normal;
  leaf_node.color = albedo_image_arr[alloc_info.albedo_index].Sample(albedo_image_sampler, info.uv);

  leaf_node_page_arr[page_index][page_offset].data.max(PackLeafNode(leaf_node));
}
