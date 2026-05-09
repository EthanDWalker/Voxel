#include "Include/common.slang"
#define VOXEL_TREE_READ_WRITE
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
void CMain(const uint32_t3 dispatch_id: SV_DispatchThreadID) {
  uint32_t2 image_size;
  image.GetDimensions(image_size.x, image_size.y);

  if (all(image_size <= dispatch_id.xy))
    return;

  const float2 beam_image_location_float = dispatch_id.xy / float(PREPASS_SCALE);
  const int32_t2 beam_image_location = dispatch_id.xy / PREPASS_SCALE;
  const bool2 is_whole = (beam_image_location_float == float2(beam_image_location));

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
  t -= 1.0f;

  const float4x4 inv_view_proj =
      mul(LookAtInverse(camera.position, camera.position + camera.front, camera.up),
          PerspectiveReverseZInverse(1.0f, 2.0f, camera.fov_y, camera.aspect_ratio));

  const float3 ray_origin = camera.position;

  const float2 center_ndc = float2(float(dispatch_id.x + 0.5) / image_size.x * 2.0 - 1.0,
                                   float(dispatch_id.y + 0.5) / image_size.y * -2.0 + 1.0);
  const float4 center_world = mul(inv_view_proj, float4(center_ndc, 1.0, 1.0));
  const float3 ray_dir = normalize(center_world.xyz / center_world.w - ray_origin);

  RaycastResult res;
  Raycast(ray_origin + t * ray_dir, ray_dir, res);

  if (res.hit) {
    const float3 L = -directional_light_arr[0].direction;
    const float light_impact = max(dot(L, res.hit_normal), 0.3f);

    const float3 acc = res.hit_color.rgb * light_impact;

    image.Store(dispatch_id.xy, float4(float3(acc), 1.0f));
  } else {
    image.Store(dispatch_id.xy, float4(0.0f));
  }
}
