#pragma once

#include "noise.slang"
#define VOXEL_TREE_READ_WRITE
#define VOXEL_TREE_ATOMIC
#define VOXEL_TREE_SET_BINDING 0
#include "voxel_tree.slang"

struct TerrainAllocateInfo {
  uint32_t2 extent;
  float2 center;
  float density;
  int32_t seed;
  uint32_t depth;
  uint32_t leaf;
};

struct GeometryInfo {
  uint32_t triangle_index;
};

[shader("vertex")]
GeometryInfo VMain(const uint32_t triangle_index: SV_VertexID) {
  GeometryInfo info;
  info.triangle_index = triangle_index;
  return info;
}

Vertex GetVertexInfo(const uint32_t2 grid_index, const TerrainAllocateInfo info) {
  const uint32_t2 non_dense_extent = uint32_t2(info.extent * info.density);

  const int32_t seed = info.seed;
  const float frequency = 500.0f;

  const uint32_t vertex_arr_index = grid_index.x + grid_index.y * non_dense_extent.x;

  const float position_step = 1.0f / info.density;

  const float2 position = (grid_index * position_step) + info.center;

  const float h = SampleNoise(seed, position / frequency);

  const float delta = 1.0f;
  const float h_x = (SampleNoise(seed, (position + float2(delta, 0)) / frequency) - h) / delta;
  const float h_y = (SampleNoise(seed, (position + float2(0, delta)) / frequency) - h) / delta;

  Vertex vertex;
  vertex.position = ((float3(grid_index.x, h, grid_index.y) -
                      (float3(non_dense_extent.x, 0.0f, non_dense_extent.y) * 0.5f)) /
                     float3(info.density, 1.0f, info.density)) +
                    float3(info.center.x, 0.0f, info.center.y);
  vertex.normal = normalize(float3(-h_y, 1.0f, -h_x));

  return vertex;
}

struct TerrainVoxelizeInfo {
  float4 position : SV_Position;
  float3 world_position : varying;
  float3 normal : varying;
};

void OutputProjectedTriangle(const Vertex vertex_arr[3],
                             inout TriangleStream<TerrainVoxelizeInfo> output_stream) {

  const float3 normal = normalize(cross(vertex_arr[1].position - vertex_arr[0].position,
                                        vertex_arr[2].position - vertex_arr[0].position));
  const float3 abs_normal = abs(normal);

  const Axis projection_axis = (abs_normal.x > abs_normal.y && abs_normal.x > abs_normal.z) ? Axis::X
                               : (abs_normal.y > abs_normal.z)                              ? Axis::Y
                                                                                            : Axis::Z;

  for (uint32_t i = 0; i < 3; i++) {
    TerrainVoxelizeInfo voxelize_info;
    voxelize_info.world_position = vertex_arr[i].position;
    voxelize_info.normal = vertex_arr[i].normal;

    const float3 pos = (voxelize_info.world_position / (box_size / 2.0f));
    voxelize_info.position = float4((projection_axis == Axis::X)   ? pos.yz
                                    : (projection_axis == Axis::Y) ? pos.xz
                                                                   : pos.xy,
                                    1.0f, 1.0f);
    output_stream.Append(voxelize_info);
  }
  output_stream.RestartStrip();
}

[shader("geometry")]
[maxvertexcount(6)]
void GMain(point GeometryInfo geometry_info[1], inout TriangleStream<TerrainVoxelizeInfo> output_stream,
           const uniform TerrainAllocateInfo info) {
  const uint32_t triangle_index = geometry_info[0].triangle_index;

  const uint32_t2 grid_index = uint32_t2(triangle_index % uint32_t(info.extent.x * info.density),
                                         triangle_index / uint32_t(info.extent.x * info.density));

  const Vertex triangle_0[3] = {
    GetVertexInfo(grid_index + uint32_t2(0, 0), info),
    GetVertexInfo(grid_index + uint32_t2(1, 1), info),
    GetVertexInfo(grid_index + uint32_t2(0, 1), info),
  };

  OutputProjectedTriangle(triangle_0, output_stream);

  const Vertex triangle_1[3] = {
    GetVertexInfo(grid_index + uint32_t2(0, 0), info),
    GetVertexInfo(grid_index + uint32_t2(1, 0), info),
    GetVertexInfo(grid_index + uint32_t2(1, 1), info),
  };

  OutputProjectedTriangle(triangle_1, output_stream);
}
