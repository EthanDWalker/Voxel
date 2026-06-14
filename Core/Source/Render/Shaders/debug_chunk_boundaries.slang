#define VOXEL_TREE_SET_BINDING 0
#include "Include/camera.slang"
#include "Include/voxel_tree.slang"

[[vk::binding(0, 1)]]
ConstantBuffer<Camera> camera;

[[vk::binding(1, 2)]]
RWTexture2D<float> depth_image;

[shader("vertex")]
uint32_t VMain(const uint32_t vertex_id: SV_VertexID) { return vertex_id; }

struct FragmentInfo {
  float4 position : SV_Position;
};

[shader("geometry")]
[maxvertexcount(24)]
void GMain(const point uint32_t vertex_id[1], inout LineStream<FragmentInfo> output_stream) {
  const float4x4 camera_view_proj = mul(camera.projection, camera.view);

  const uint32_t local_index = vertex_id[0];
  const uint32_t3 local_offset = GetLocalOffset(local_index);

  if (voxel_tree[0][local_index].child_mask == 0)
    return;

  const float3 min_bound = voxel_tree_header.min_bound + (box_size / 4.0f) * local_offset;
  const float3 max_bound = min_bound + (box_size / 4.0f);

  const float3 corners[8] = {
    float3(min_bound.x, min_bound.y, min_bound.z), float3(max_bound.x, min_bound.y, min_bound.z),
    float3(max_bound.x, max_bound.y, min_bound.z), float3(min_bound.x, max_bound.y, min_bound.z),

    float3(min_bound.x, min_bound.y, max_bound.z), float3(max_bound.x, min_bound.y, max_bound.z),
    float3(max_bound.x, max_bound.y, max_bound.z), float3(min_bound.x, max_bound.y, max_bound.z),
  };

  const uint32_t2 edges[12] = {
    uint32_t2(0, 1), uint32_t2(1, 2), uint32_t2(2, 3), uint32_t2(3, 0), uint32_t2(4, 5), uint32_t2(5, 6),
    uint32_t2(6, 7), uint32_t2(7, 4), uint32_t2(0, 4), uint32_t2(1, 5), uint32_t2(2, 6), uint32_t2(3, 7),
  };

  for (uint i = 0; i < 12; i++) {
    {
      FragmentInfo fragment_info;
      fragment_info.position = mul(camera_view_proj, float4(corners[edges[i].x], 1.0f));
      output_stream.Append(fragment_info);
    }

    {
      FragmentInfo fragment_info;
      fragment_info.position = mul(camera_view_proj, float4(corners[edges[i].y], 1.0f));
      output_stream.Append(fragment_info);
    }

    output_stream.RestartStrip();
  }
}

[shader("fragment")]
float4 FMain(const FragmentInfo info) : SV_Target { return 1.0f; }
