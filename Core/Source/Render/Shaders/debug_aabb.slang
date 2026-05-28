#include "Include/camera.slang"
#include "Include/common.slang"

[[vk::binding(1, 0)]]
CountedBuffer<AABB> aabb_arr;

[[vk::binding(0, 1)]]
ConstantBuffer<Camera> camera;

struct FragmentInfo {
  float4 position : SV_Position;
  float4 color : varying;
};

[shader("vertex")]
uint32_t VMain(const uint32_t aabb_index: SV_VertexID) { return aabb_index; }

[shader("geometry")]
[maxvertexcount(24)]
void GMain(point uint32_t point_info[1], inout LineStream<FragmentInfo> output_stream) {
  if (point_info[0] >= aabb_arr.count) {
    return;
  }

  const float4x4 camera_view_proj = mul(camera.projection, camera.view);

  const float3 min_bound = aabb_arr[point_info[0]].min_bound;
  const float3 max_bound = aabb_arr[point_info[0]].max_bound;

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
      fragment_info.color = 1.0f;
      output_stream.Append(fragment_info);
    }

    {
      FragmentInfo fragment_info;
      fragment_info.position = mul(camera_view_proj, float4(corners[edges[i].y], 1.0f));
      fragment_info.color = 1.0f;
      output_stream.Append(fragment_info);
    }

    output_stream.RestartStrip();
  }
}

[shader("fragment")]
float4 FMain(const FragmentInfo fragment_info) : SV_Target { return fragment_info.color; }
