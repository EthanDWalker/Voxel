#include "Include/common.slang"

struct BranchNode {
  Atomic<uint64_t> child_mask;
};

[[vk::binding(0, 0)]]
RWStructuredBuffer<BranchNode> mesh_node_brick_arr[];

[[vk::binding(1, 0)]]
RWCountedBuffer<AABB> voxel_aabb_arr_arr[];

[[vk::binding(2, 0)]]
RWStructuredBuffer<Vertex> vertex_arr_arr[];

[[vk::binding(3, 0)]]
RWStructuredBuffer<Index> index_arr_arr[];

[[vk::binding(4, 0)]]
RWStructuredBuffer<AABB> mesh_aabb_arr;

struct AllocateInfo {
  uint32_t depth;
  uint32_t leaf;
  uint32_t mesh_index;
  uint32_t albedo_index;
};

struct TriangleInfo {
  uint32_t triangle_id;
};

[shader("vertex")]
TriangleInfo VMain(const uint32_t triangle_id: SV_VertexID, const uniform AllocateInfo alloc_info) {
  TriangleInfo info;
  info.triangle_id = triangle_id;
  return info;
}

struct FragmentInfo {
  float4 position : SV_Position;
  float3 model_position : varying;
};

[shader("geometry")]
[maxvertexcount(3)]
void GMain(point TriangleInfo triangle_info[1], inout TriangleStream<FragmentInfo> output_stream,
           const uniform AllocateInfo alloc_info) {
  const uint32_t mesh_id = alloc_info.mesh_index;

  const Vertex input[3] = {
    vertex_arr_arr[mesh_id][index_arr_arr[mesh_id][triangle_info[0].triangle_id * 3 + 0]],
    vertex_arr_arr[mesh_id][index_arr_arr[mesh_id][triangle_info[0].triangle_id * 3 + 1]],
    vertex_arr_arr[mesh_id][index_arr_arr[mesh_id][triangle_info[0].triangle_id * 3 + 2]],
  };

  const float3 normal = cross(input[1].position - input[0].position, input[2].position - input[0].position);
  const float3 abs_normal = abs(normal);

  const Axis projection_axis = (abs_normal.x > abs_normal.y && abs_normal.x > abs_normal.z) ? Axis::X
                               : (abs_normal.y > abs_normal.z)                              ? Axis::Y
                                                                                            : Axis::Z;

  const float3 aabb_size = mesh_aabb_arr[mesh_id].max_bound - mesh_aabb_arr[mesh_id].min_bound;
  const float3 aabb_center = aabb_size / 2.0f;

  for (uint32_t i = 0; i < 3; i++) {
    FragmentInfo frag_info;
    frag_info.model_position = input[i].position;

    const float3 pos = (input[i].position / aabb_size);
    frag_info.position = float4((projection_axis == Axis::X)   ? pos.yz
                                : (projection_axis == Axis::Y) ? pos.xz
                                                               : pos.xy,
                                1.0f, 1.0f);
    output_stream.Append(frag_info);
  }
  output_stream.RestartStrip();
}

static const float VOXEL_AABB_SIZE = 0.5f;

[shader("fragment")]
void FMain(const FragmentInfo info, const uniform AllocateInfo alloc_info) {
  const AABB mesh_aabb = mesh_aabb_arr[alloc_info.mesh_index];
  const uint32_t3 voxel_aabb_index =
      uint32_t3((info.model_position - mesh_aabb_arr[alloc_info.mesh_index].min_bound) / VOXEL_AABB_SIZE);
  uint32_t3 voxel_aabb_max_index = uint32_t3(
      ceil((mesh_aabb_arr[alloc_info.mesh_index].max_bound - mesh_aabb_arr[alloc_info.mesh_index].min_bound) /
           VOXEL_AABB_SIZE));
  voxel_aabb_max_index = select(voxel_aabb_max_index == 0, 1, voxel_aabb_max_index);

  AABB voxel_aabb;
  voxel_aabb.min_bound = mesh_aabb.min_bound + voxel_aabb_index * VOXEL_AABB_SIZE;
  voxel_aabb.max_bound = voxel_aabb.min_bound + VOXEL_AABB_SIZE;

  const uint32_t arr_index = voxel_aabb_index.x + (voxel_aabb_index.y * voxel_aabb_max_index.x) +
                             (voxel_aabb_index.z * voxel_aabb_max_index.x * voxel_aabb_max_index.y);

  const uint32_t3 local_index =
      uint32_t3((info.model_position - voxel_aabb.min_bound) / (VOXEL_AABB_SIZE / 4.0f));
  const uint32_t local_index_index = SumComponents(local_index << uint32_t3(0, 2, 4));

  const uint64_t ret =
      mesh_node_brick_arr[alloc_info.mesh_index][arr_index].child_mask.or(uint64_t(1) << local_index_index);

  if (ret == 0) {
    voxel_aabb_arr_arr[alloc_info.mesh_index].Append(voxel_aabb);
  }
}
