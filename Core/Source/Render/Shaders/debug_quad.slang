#include "Include/common.slang"
#include "Include/camera.slang"

[[vk::binding(0, 0)]]
CountedBuffer<AxisAlignedQuad> emissive_quad_arr;

[[vk::binding(0, 1)]]
ConstantBuffer<Camera> camera;

[[vk::binding(1, 2)]]
RWTexture2D<float> depth_image;

[shader("vertex")]
uint32_t VMain(const uint32_t vertex_id: SV_VertexID) { return vertex_id; }

struct FragmentInfo {
  float4 position : SV_Position;
  float3 color : varying;
};

[shader("geometry")]
[maxvertexcount(10)]
void GMain(const point uint32_t quad_index[1], inout LineStream<FragmentInfo> output_stream) {
  const float4x4 camera_view_proj = mul(camera.projection, camera.view);

  const AxisAlignedQuad quad = emissive_quad_arr[quad_index[0]];
  const float3 diag = quad.point_2 - quad.point_1;
  float3 edge_arr[2];
  uint32_t edge_index = 0;
  for (uint32_t i = 0; i < 3; i++) {
    if (diag[i] != 0.0f) {
      edge_arr[edge_index] = 0.0f;
      edge_arr[edge_index][i] = diag[i];
      edge_index++;
    }
  }

  const float3 corners[4] = {
    quad.point_1,
    quad.point_1 + edge_arr[0],
    quad.point_1 + edge_arr[0] + edge_arr[1],
    quad.point_1 + edge_arr[1],
  };

  const uint32_t2 edges[5] = {
    uint32_t2(0, 1),
    uint32_t2(1, 2),
    uint32_t2(2, 3),
    uint32_t2(3, 0),
    uint32_t2(1, 3),
  };

  for (uint i = 0; i < 5; i++) {
    {
      FragmentInfo fragment_info;
      fragment_info.position = mul(camera_view_proj, float4(corners[edges[i][0]], 1.0f));
      fragment_info.color = quad.normal;
      output_stream.Append(fragment_info);
    }

    {
      FragmentInfo fragment_info;
      fragment_info.position = mul(camera_view_proj, float4(corners[edges[i][1]], 1.0f));
      fragment_info.color = quad.normal;
      output_stream.Append(fragment_info);
    }

    output_stream.RestartStrip();
  }
}

[shader("fragment")]
float4 FMain(const FragmentInfo info) : SV_Target { return float4(info.color, 1.0f); }
