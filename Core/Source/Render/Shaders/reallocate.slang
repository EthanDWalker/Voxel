#include "Include/common.slang"
#define VOXEL_TREE_ATOMIC
#define VOXEL_TREE_SET_BINDING 1
#include "Include/voxel_tree.slang"
#define DENSE_SET_SET_BINDING 2
#include "Include/dense_set.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<uint32_t> mesh_triangle_offsets;

[[vk::binding(1, 0)]]
RWStructuredBuffer<Vertex> vertex_arr[];

[[vk::binding(2, 0)]]
StructuredBuffer<uint32_t> index_arr[];

[[vk::binding(3, 0)]]
RWStructuredBuffer<uint32_t> triangle_voxelized_depth_arr[];

[shader("vertex")]
uint32_t VMain(const uint32_t dense_set_value_index: SV_VertexID) { return dense_set_value_index; }

struct ReallocateInfo {
  float4 position : SV_Position;
  float3 world_position;
  float16_t2 uv;
  float3 normal;
};

[shader("geometry")]
[maxvertexcount(3)]
void GMain(point uint32_t dense_set_value_index[1], inout TriangleStream<ReallocateInfo> output_stream) {
  if (dense_set_value_index[0] >= dense_set_header.value_count)
    return;

  const uint32_t global_triangle_id = value_arr[dense_set_value_index[0]];

  uint32_t mesh_id = 0;
  while (true) {
    if (global_triangle_id < mesh_triangle_offsets[mesh_id + 1] || mesh_triangle_offsets[mesh_id + 1] == 0) {
      break;
    } else {
      mesh_id++;
    }
  }

  const uint32_t local_triangle_id = global_triangle_id - mesh_triangle_offsets[mesh_id];

  triangle_voxelized_depth_arr[mesh_id][local_triangle_id] += 1;
  return;

  const Vertex input[3] = {
    vertex_arr[mesh_id][index_arr[mesh_id][local_triangle_id * 3 + 0]],
    vertex_arr[mesh_id][index_arr[mesh_id][local_triangle_id * 3 + 1]],
    vertex_arr[mesh_id][index_arr[mesh_id][local_triangle_id * 3 + 2]],
  };

  const float3 normal =
      normalize(cross(input[1].position - input[0].position, input[2].position - input[0].position));
  const float3 abs_normal = abs(normal);

  const Axis projection_axis = (abs_normal.x > abs_normal.y && abs_normal.x > abs_normal.z) ? Axis::X
                               : (abs_normal.y > abs_normal.z)                              ? Axis::Y
                                                                                            : Axis::Z;

  for (uint32_t i = 0; i < 3; i++) {
    ReallocateInfo reallocate_info;
    const float3 pos = (input[i].position / (box_size / 2.0f));
    reallocate_info.position = float4((projection_axis == Axis::X)   ? pos.yz
                                      : (projection_axis == Axis::Y) ? pos.xz
                                                                     : pos.xy,
                                      1.0f, 1.0f);
    reallocate_info.uv = float16_t2(input[i].uv);
    reallocate_info.world_position = input[i].position;
    reallocate_info.normal = normal;
    output_stream.Append(reallocate_info);
  }
  output_stream.RestartStrip();
}

[shader("fragment")]
void FMain(const ReallocateInfo info) {
  const uint32_t3 tree_index = GetTreeIndex(info.world_position);
  uint32_t ptr = GetLocalIndex(tree_index, 0);

  for (uint32_t i = 1;; i++) {
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    const uint32_t local_index = GetLocalIndex(tree_index, i);

    const ChildPtrData child_ptr_data =
        ChildPtrData(voxel_tree[page_index][page_offset].child_ptr_data.load());

    if (child_ptr_data.data == SENTINAL)
      return;

    ptr = child_ptr_data.child_ptr + local_index;

    if (child_ptr_data.is_leaf) {
      break;
    }
  }

  const uint32_t leaf_page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t leaf_page_offset = ptr - (leaf_page_index << voxel_tree_header.page_size_exp);

  LeafNode leaf_node = UnpackLeafNode(voxel_tree_leaf[leaf_page_index][leaf_page_offset].data.load());
  leaf_node.color = 1.0f;
  voxel_tree_leaf[leaf_page_index][leaf_page_offset].data = PackLeafNode(leaf_node);
}
