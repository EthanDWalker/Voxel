#include "Include/common.slang"
#define VOXEL_TREE_SET_BINDING 1
#include "Include/voxel_tree.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<Vertex> vertex_arr;

[shader("vertex")]
Vertex VMain(const uint32_t vertex_id: SV_VertexID) {
  //
  return vertex_arr[vertex_id];
}

[shader("geometry")]
[maxvertexcount(3)]
void GMain(triangle Vertex input[3], inout TriangleStream<VoxelizeInfo> output_stream) {
  const float3 abs_normal =
      abs(cross(input[1].position - input[0].position, input[2].position - input[0].position));

  const Axis projection_axis = (abs_normal.x > abs_normal.y && abs_normal.x > abs_normal.z) ? Axis::X
                               : (abs_normal.y > abs_normal.z)                              ? Axis::Y
                                                                                            : Axis::Z;

  for (uint32_t i = 0; i < 3; i++) {
    VoxelizeInfo voxelize_info;
    const float3 pos = (input[i].position / voxel_tree_header.max_bound);
    voxelize_info.position = float4((projection_axis == Axis::X)   ? pos.yz
                                    : (projection_axis == Axis::Y) ? pos.xz
                                                                   : pos.xy,
                                    1.0f, 1.0f);
    voxelize_info.uv = float16_t2(input[i].uv.yx);
    voxelize_info.world_position = input[i].position;
    output_stream.Append(voxelize_info);
  }
  output_stream.RestartStrip();
}
