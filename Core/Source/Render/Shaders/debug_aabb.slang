#include "Include/camera.slang"
#include "Include/common.slang"

<<<<<<< Updated upstream
[[vk::binding(0, 0)]]
CountedBuffer<Mesh> mesh_arr;

[[vk::binding(1, 0)]]
CountedBuffer<Instance> instance_arr;

=======
>>>>>>> Stashed changes
[[vk::binding(0, 1)]]
ConstantBuffer<Camera> camera;

struct FragmentInfo {
  float4 position : SV_Position;
  float4 color : varying;
};

[shader("vertex")]
uint32_t VMain(const uint32_t instance_index: SV_VertexID) { return instance_index; }

[shader("geometry")]
[maxvertexcount(24)]
void GMain(point uint32_t point_info[1], inout LineStream<FragmentInfo> output_stream) {
  const float4x4 camera_view_proj = mul(camera.projection, camera.view);

  const float4x4 instance_matrix = instance_arr[point_info[0]].matrix;
  const uint32_t mesh_index = instance_arr[point_info[0]].mesh_index;

  const float3 min_bound = mul(instance_matrix, float4(mesh_arr[mesh_index].min_bound, 1.0f)).xyz;
  const float3 max_bound = mul(instance_matrix, float4(mesh_arr[mesh_index].max_bound, 1.0f)).xyz;

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
