#include "Include/common.slang"
#define VOXEL_TREE_SET_BINDING 2
#include "Include/camera.slang"
#include "Include/voxel_tree.slang"

[[vk::binding(0, 0)]]
RWTexture2D<float4> image;
[[vk::binding(1, 0)]]
RWTexture2D<float> beam_image;

[[vk::binding(0, 1)]]
ConstantBuffer<Camera> camera;

[[vk::binding(0, 3)]]
CountedBuffer<DirectionalLight> directional_light_arr;

[shader("compute")]
[numthreads(8, 8, 1)]
void CMain(const uint32_t2 dispatch_id: SV_DispatchThreadID) {
  uint32_t2 image_size;
  image.GetDimensions(image_size.x, image_size.y);

  if (all(image_size <= dispatch_id))
    return;

  const float2 beam_image_location_float = dispatch_id / float(BEAM_PREPASS_SCALE);
  const int32_t2 beam_image_location = dispatch_id / BEAM_PREPASS_SCALE;
  const bool2 is_whole = (beam_image_location_float == beam_image_location);

  float t;
  if (all(is_whole)) {
    t = beam_image.Load(beam_image_location);
    t = min(t, beam_image.Load(beam_image_location + int32_t2(0, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 0)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 1)));
  } else if (is_whole.x) {
    t = beam_image.Load(beam_image_location);
    t = min(t, beam_image.Load(beam_image_location + int32_t2(0, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 0)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(-1, 0)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(-1, 1)));
  } else if (is_whole.y) {
    t = beam_image.Load(beam_image_location);
    t = min(t, beam_image.Load(beam_image_location + int32_t2(0, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(0, -1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 0)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, -1)));
  } else {
    t = beam_image.Load(beam_image_location);
    t = min(t, beam_image.Load(beam_image_location + int32_t2(0, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 0)));
  }

  const float4x4 inv_view_proj =
      mul(LookAtInverse(camera.position, camera.position + camera.front, camera.up),
          PerspectiveReverseZInverse(camera.z_near, camera.z_near + 1.0f, camera.fov_y, camera.aspect_ratio));

  const float2 ndc = float2(float(dispatch_id.x + 0.5f) / float(image_size.x) * 2.0f - 1.0f,
                            float(dispatch_id.y + 0.5f) / float(image_size.y) * -2.0f + 1.0f);

  const float3 ndc_min = float3(ndc, 0.0f);
  const float3 ndc_max = float3(ndc, 1.0f);

  const float4 world_min = mul(inv_view_proj, float4(ndc_min, 1.0f));
  const float4 world_max = mul(inv_view_proj, float4(ndc_max, 1.0f));

  const float3 ray_dir = normalize((world_max.xyz / world_max.w) - (world_min.xyz / world_min.w));
  const float3 ray_origin = camera.position;

  RaycastResult res;
  Raycast(ray_origin + t * ray_dir, ray_dir, res);

  if (res.hit) {
    const float3 L = -directional_light_arr[0].direction;
    const float light_impact = max(dot(L, res.hit_normal), 0.3f);

    const float3 acc = res.hit_color.rgb * light_impact;

    image.Store(dispatch_id, float4(acc, 1.0f));
  } else {
    image.Store(dispatch_id, float4(float3(0.0f), 1.0f));
  }
}
