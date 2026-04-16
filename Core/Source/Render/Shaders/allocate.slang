#include "Include/common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<Vertex> vertex_arr;

[shader("vertex")]
Vertex VMain(const uint32_t vertex_id: SV_VertexID) {
  //
  return vertex_arr[vertex_id];
}

#define VOXEL_TREE_ATOMIC
#define VOXEL_TREE_READ_WRITE
#define VOXEL_TREE_SET_BINDING 1
#include "Include/voxel_tree.slang"

void Allocate(const uint32_t3 index, const uint32_t depth) {
  uint32_t ptr = GetLocalIndex(index, 0);

  for (uint32_t i = 1; i < depth; i++) {
    const uint32_t page_level_offset = voxel_tree_header.GetLevelPageOffset(i - 1);
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    const uint32_t local_index = GetLocalIndex(index, i);

    const uint32_t child_ptr = voxel_tree[page_index + page_level_offset][page_offset].child_ptr.load();

    ptr = child_ptr + local_index;
  }

  const uint32_t page_level_offset = voxel_tree_header.GetLevelPageOffset(depth - 1);
  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;

  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

  if (voxel_tree[page_index + page_level_offset][page_offset].child_ptr.compareExchange(SENTINAL, 0) ==
      SENTINAL) {
    if (depth == voxel_tree_header.max_depth - 1) {
      voxel_tree[page_index + page_level_offset][page_offset].child_ptr =
          voxel_tree_header.AllocateLeafVoxels(64);
    } else {
      voxel_tree[page_index + page_level_offset][page_offset].child_ptr =
          voxel_tree_header.AllocateVoxels(depth, 64);
    }
  }
}

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
void FMain(const GeometryOut input, const uniform uint32_t depth) {
  Allocate(GetTreeIndex(input.world_position), depth);
}
