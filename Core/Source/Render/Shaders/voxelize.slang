#include "Include/common.slang"
#define VOXEL_TREE_SET_BINDING 1
#include "Include/voxel_tree.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<uint32_t> mesh_triangle_offsets;

[[vk::binding(1, 0)]]
RWStructuredBuffer<Vertex> vertex_arr[];

[[vk::binding(2, 0)]]
StructuredBuffer<uint32_t> index_arr[];

[[vk::binding(3, 0)]]
StructuredBuffer<uint32_t> triangle_voxelized_depth_arr[];

[[vk::binding(4, 0)]]
Texture2D<float4> albedo_image_arr[];

[[vk::binding(5, 0)]]
SamplerState albedo_image_sampler;

struct VertexInfo {
  uint32_t vertex_id;
  uint32_t triangle_id;
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

[shader("geometry")]
[maxvertexcount(3)]
void GMain(point TriangleInfo triangle_info[1], inout TriangleStream<VoxelizeInfo> output_stream,
           const uniform AllocateInfo alloc_info) {
  const Vertex input[3] = {
    vertex_arr[alloc_info.mesh_id][index_arr[alloc_info.mesh_id][triangle_info[0].triangle_id * 3 + 0]],
    vertex_arr[alloc_info.mesh_id][index_arr[alloc_info.mesh_id][triangle_info[0].triangle_id * 3 + 1]],
    vertex_arr[alloc_info.mesh_id][index_arr[alloc_info.mesh_id][triangle_info[0].triangle_id * 3 + 2]],
  };

  const float3 normal =
      normalize(cross(input[1].position - input[0].position, input[2].position - input[0].position));
  const float3 abs_normal = abs(normal);

  const Axis projection_axis = (abs_normal.x > abs_normal.y && abs_normal.x > abs_normal.z) ? Axis::X
                               : (abs_normal.y > abs_normal.z)                              ? Axis::Y
                                                                                            : Axis::Z;

  for (uint32_t i = 0; i < 3; i++) {
    VoxelizeInfo voxelize_info;
    const float3 pos = (input[i].position / (box_size / 2.0f));
    voxelize_info.position = float4((projection_axis == Axis::X)   ? pos.yz
                                    : (projection_axis == Axis::Y) ? pos.xz
                                                                   : pos.xy,
                                    1.0f, 1.0f);
    voxelize_info.uv = float16_t2(input[i].uv);
    voxelize_info.world_position = input[i].position;
    voxelize_info.normal = normal;
    output_stream.Append(voxelize_info);
  }
  output_stream.RestartStrip();
}
