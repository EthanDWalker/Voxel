#include "Include/camera.slang"
#define VOXEL_TREE_SET_BINDING 2
#include "Include/voxel_tree.slang"

[[vk::binding(1, 0)]]
RWTexture2D<float> beam_image;

[[vk::binding(0, 1)]]
ConstantBuffer<Camera> camera;

[numthreads(8, 8, 1)]
void CMain(const uint32_t2 dispatch_id: SV_DispatchThreadID) {
  uint32_t2 image_size;
  beam_image.GetDimensions(image_size.x, image_size.y);

  if (all(image_size <= dispatch_id))
    return;

  const float4x4 inv_view_proj =
      mul(LookAtInverse(camera.position, camera.position + camera.front, camera.up),
          PerspectiveReverseZInverse(camera.z_near, camera.z_near + 1.0f, camera.fov_y, camera.aspect_ratio));

  const float2 ndc = float2(float(dispatch_id.x) / float(image_size.x) * 2.0f - 1.0f,
                            float(dispatch_id.y) / float(image_size.y) * -2.0f + 1.0f);

  const float3 ndc_min = float3(ndc, 0.0f);
  const float3 ndc_max = float3(ndc, 1.0f);

  const float4 world_min = mul(inv_view_proj, float4(ndc_min, 1.0f));
  const float4 world_max = mul(inv_view_proj, float4(ndc_max, 1.0f));

  const float3 ray_dir = normalize((world_max.xyz / world_max.w) - (world_min.xyz / world_min.w));
  const float3 ray_origin = camera.position;

  RaycastResult res;
  Raycast(ray_origin, ray_dir, res);
  beam_image.Store(dispatch_id, res.t);
}
