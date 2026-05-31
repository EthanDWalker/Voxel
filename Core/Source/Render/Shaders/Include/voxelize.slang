#include "common.slang"

[[vk::binding(2, 0)]]
StructuredBuffer<Vertex> vertex_arr_arr[];

[[vk::binding(3, 0)]]
StructuredBuffer<Index> index_arr_arr[];

[[vk::binding(4, 0)]]
StructuredBuffer<AABB> mesh_aabb_arr;

struct AllocateInfo {
  uint32_t depth;
  uint32_t leaf;
  uint32_t mesh_index;
  uint32_t albedo_index;
};

static uint32_t CHILD_PTR_SENTINAL = 0xFFFFFFFF;

struct TriangleInfo {
  uint32_t triangle_id : flat;
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
  float3 normal : varying;
  float2 uv : varying;
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

  const float3 triangle_centriod = (input[0].position + input[1].position + input[2].position) / 3.0f;

  for (uint32_t i = 0; i < 3; i++) {
    FragmentInfo frag_info;
    frag_info.model_position = input[i].position;

    const float3 pos = (input[i].position - triangle_centriod) / aabb_size;
    frag_info.position = float4((projection_axis == Axis::X)   ? pos.yz
                                : (projection_axis == Axis::Y) ? pos.xz
                                                               : pos.xy,
                                1.0f, 1.0f);
    frag_info.normal = input[i].normal;
    frag_info.uv = input[i].uv;
    output_stream.Append(frag_info);
  }
  output_stream.RestartStrip();
}
