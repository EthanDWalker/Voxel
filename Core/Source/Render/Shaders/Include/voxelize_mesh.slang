#pragma once

#include "common.slang"
#define VOXEL_TREE_READ_WRITE
#define VOXEL_TREE_ATOMIC
#define VOXEL_TREE_SET_BINDING 1
#include "voxel_tree.slang"

struct VoxelizeInfo {
  float4 position : SV_Position;
  float3 world_position : varying;
  float16_t2 uv : varying;
  float3 normal : varying;
};

struct MeshAllocateInfo {
  float4x4 instance_matrix;
  uint32_t depth;
  uint32_t leaf;
  uint32_t mesh_index;
  uint32_t albedo_image_index;
};

[[vk::binding(0, 0)]]
StructuredBuffer<Vertex> vertex_arr_arr[];

[[vk::binding(1, 0)]]
StructuredBuffer<uint32_t> index_arr_arr[];

struct VertexInfo {
  uint32_t vertex_id;
  uint32_t triangle_id;
};

struct TriangleInfo {
  uint32_t triangle_id;
};

[shader("vertex")]
TriangleInfo VMain(const uint32_t triangle_id: SV_VertexID, const uniform MeshAllocateInfo alloc_info) {
  TriangleInfo info;
  info.triangle_id = triangle_id;

  return info;
}

float3x3 GetNormalMatrix(const float4x4 m) {
  const float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
  const float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
  const float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];

  const float b01 = a22 * a11 - a12 * a21;
  const float b11 = -a22 * a10 + a12 * a20;
  const float b21 = a21 * a10 - a11 * a20;

  const float det = a00 * b01 + a01 * b11 + a02 * b21;

  // clang-format off
  return transpose(
    float3x3(
      b01, (-a22 * a01 + a02 * a21), (a12 * a01 - a02 * a11), 
      b11, (a22 * a00 - a02 * a20) , (-a12 * a00 + a02 * a10), 
      b21, (-a21 * a00 + a01 * a20), (a11 * a00 - a01 * a10)
    ) / det
  );
  // clang-format on
}

[shader("geometry")]
[maxvertexcount(3)]
void GMain(point TriangleInfo triangle_info[1], inout TriangleStream<VoxelizeInfo> output_stream,
           const uniform MeshAllocateInfo alloc_info) {
  const uint32_t mesh_id = alloc_info.mesh_index;

  Vertex input[3];

  for (uint32_t i = 0; i < 3; i++) {
    input[i] = vertex_arr_arr[mesh_id][index_arr_arr[mesh_id][triangle_info[0].triangle_id * 3 + i]];
  }

  const float4x4 instance_matrix = alloc_info.instance_matrix;
  const float3x3 normal_matrix = GetNormalMatrix(instance_matrix);

  const float3 normal =
      normalize(cross(input[1].position - input[0].position, input[2].position - input[0].position));
  const float3 abs_normal = abs(normal);

  const Axis projection_axis = (abs_normal.x > abs_normal.y && abs_normal.x > abs_normal.z) ? Axis::X
                               : (abs_normal.y > abs_normal.z)                              ? Axis::Y
                                                                                            : Axis::Z;

  for (uint32_t i = 0; i < 3; i++) {
    VoxelizeInfo voxelize_info;
    voxelize_info.uv = float16_t2(input[i].uv);
    voxelize_info.world_position = mul(instance_matrix, float4(input[i].position, 1.0f)).xyz;
    voxelize_info.normal = input[i].normal;

    const float3 pos = (voxelize_info.world_position / (box_size / 2.0f));
    voxelize_info.position = float4((projection_axis == Axis::X)   ? pos.yz
                                    : (projection_axis == Axis::Y) ? pos.xz
                                                                   : pos.xy,
                                    1.0f, 1.0f);
    output_stream.Append(voxelize_info);
  }
  output_stream.RestartStrip();
}
