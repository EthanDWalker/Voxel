#include "Include/common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<Vertex> vertex_arr;
[[vk::binding(1, 0)]]
Texture2D<float4> albedo_texture;
[[vk::binding(2, 0)]]
SamplerState sampler;

[shader("vertex")]
Vertex VMain(const uint32_t vertex_id: SV_VertexID) {
  //
  return vertex_arr[vertex_id];
}

#define VOXEL_TREE_SET_BINDING 1
#define VOXEL_TREE_ATOMIC
#define VOXEL_TREE_READ_WRITE
#include "Include/voxel_tree.slang"

struct GeometryOut {
  float4 position : SV_Position;
  float3 world_position;
  float16_t2 uv;
};

[shader("geometry")]
[maxvertexcount(3)]
void GMain(triangle Vertex input[3], inout TriangleStream<GeometryOut> output_stream) {
  const float3 abs_normal =
      abs(cross(input[1].position - input[0].position, input[2].position - input[0].position));

  const Axis projection_axis = (abs_normal.x > abs_normal.y && abs_normal.x > abs_normal.z) ? Axis::X
                               : (abs_normal.y > abs_normal.z)                              ? Axis::Y
                                                                                            : Axis::Z;

  for (uint32_t i = 0; i < 3; i++) {
    GeometryOut geometry_out;
    const float3 pos = input[i].position / voxel_tree_header.max_bound;
    geometry_out.position = float4((projection_axis == Axis::X)   ? pos.yz
                                   : (projection_axis == Axis::Y) ? pos.xz
                                                                  : pos.xy,
                                   1.0f, 1.0f);
    geometry_out.uv = float16_t2(input[i].uv.yx);
    geometry_out.world_position = input[i].position;
    output_stream.Append(geometry_out);
  }
  output_stream.RestartStrip();
}

[shader("fragment")]
void FMain(const GeometryOut input) {
  const uint32_t3 tree_index = GetTreeIndex(input.world_position);
  uint32_t ptr = GetLocalIndex(tree_index, 0);
  uint32_t local_index;

  for (uint32_t i = 1; i < voxel_tree_header.max_depth; i++) {
    const uint32_t level_page_offset = voxel_tree_header.GetLevelPageOffset(i - 1);
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    local_index = GetLocalIndex(tree_index, i);

    voxel_tree[page_index + level_page_offset][page_offset].child_mask.or(uint64_t(1) << local_index);

    ptr = voxel_tree[page_index + level_page_offset][page_offset].child_ptr.load() + local_index;
  }

  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);
  const float4 color = albedo_texture.SampleLevel(sampler, input.uv, 0.0f);
  voxel_tree_leaf[page_index][page_offset].color = PackRGBA8(color);
}
