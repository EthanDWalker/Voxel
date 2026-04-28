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
          PerspectiveReverseZInverse(1.0f, 2.0f, camera.fov_y, camera.aspect_ratio));

  const float3 ray_origin = camera.position;

  const float2 center_ndc = float2(float(dispatch_id.x + 0.5) / image_size.x * 2.0 - 1.0,
                                   float(dispatch_id.y + 0.5) / image_size.y * -2.0 + 1.0);
  const float4 center_world = mul(inv_view_proj, float4(center_ndc, 1.0, 1.0));
  const float3 ray_dir = normalize(center_world.xyz / center_world.w - ray_origin);

  const float2 pixel_corner_offsets[4] = {
    float2(0.0f, 0.0f),
    float2(1.0f, 0.0f),
    float2(0.0f, 1.0f),
    float2(1.0f, 1.0f),
  };

  float max_r = 0.0f;
  for (uint32_t i = 0; i < 4; i++) {
    const float2 corner_ndc =
        float2(float(dispatch_id.x + pixel_corner_offsets[i].x) / float(image_size.x) * 2.0f - 1.0f,
               float(dispatch_id.y + pixel_corner_offsets[i].y) / float(image_size.y) * -2.0f + 1.0f);

    const float4 world_pixel_corner = mul(inv_view_proj, float4(corner_ndc, 1.0f, 1.0f));
    max_r = max(max_r, length(world_pixel_corner.xyz / world_pixel_corner.w - center_world.xyz / center_world.w));
  }

  RaycastResult res;
  Raycast(ray_origin, ray_dir, res, voxel_radius / max_r);
  beam_image.Store(dispatch_id, res.t);
}
