#include "Include/common.slang"
#define VOXEL_TREE_SET_BINDING 2
#include "Include/camera.slang"
#include "Include/voxel_tree.slang"
#define HASH_SET_SET_BINDING 4
#include "Include/hash_set.slang"

[[vk::binding(0, 0)]]
RWTexture2D<float4> image;
[[vk::binding(1, 0)]]
RWTexture2D<float> beam_image;

[[vk::binding(0, 1)]]
ConstantBuffer<Camera> camera;
[[vk::binding(1, 1)]]
ConstantBuffer<Camera> last_camera;

[[vk::binding(0, 3)]]
CountedBuffer<DirectionalLight> directional_light_arr;

float3 GetPixelRayDir(const uint32_t2 pixel_index, const uint32_t2 image_size, const float4x4 inv_view_proj) {
  const float2 pixel_center_ndc = float2(float(pixel_index.x + 0.5f) / image_size.x * 2.0f - 1.0f,
                                         float(pixel_index.y + 0.5f) / image_size.y * -2.0f + 1.0f);
  const float4 pixel_center_world = mul(inv_view_proj, float4(pixel_center_ndc, 1.0f, 1.0f));
  return normalize(pixel_center_world.xyz / pixel_center_world.w - camera.position);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void CMain(const uint32_t2 dispatch_id: SV_DispatchThreadID) {
  uint32_t2 image_size;
  image.GetDimensions(image_size.x, image_size.y);

  if (all(image_size <= dispatch_id))
    return;

  const float4x4 inv_view_proj =
      mul(LookAtInverse(camera.position, camera.position + camera.front, camera.up),
          PerspectiveReverseZInverse(1.0f, 2.0f, camera.fov_y, camera.aspect_ratio));

  const float3 ray_origin = camera.position;
  const float3 ray_dir = GetPixelRayDir(dispatch_id, image_size, inv_view_proj);

  const float t = SampleBeamImage(dispatch_id, beam_image);

  RaycastResult res;
  Raycast(ray_origin + t * ray_dir, ray_dir, res);

  if (res.hit) {
    const float3 L = -directional_light_arr[0].direction;
    const float light_impact = max(dot(L, res.hit_normal), 0.0f);

    bool in_set;
    const IndirectLightValue shadow_value = hash_set.GetId(res.leaf_ptr, in_set);
    float shadow_impact;
    if (in_set) {
      shadow_impact = 1.0f - float(shadow_value.in_shadow);
    } else {
      RaycastResult shadow_res;
      const float3 shadow_ray_origin =
          StepTreeRaySingle(GetWorldPositionCenter(res.hit_tree_index), res.hit_normal, res.hit_level);
      Raycast(shadow_ray_origin, L, shadow_res);
      shadow_impact = 1.0f - float(shadow_res.hit);
    }
    shadow_impact = max(shadow_impact, 0.0f);

    RaycastResult indirect_light_res;
    const float3 indirect_light_dir = GetPixelRayDir(dispatch_id >> INDIRECT_PREPASS_SCALE_EXP,
                                                     image_size >> INDIRECT_PREPASS_SCALE_EXP, inv_view_proj);
    const float indirect_t = SampleBeamImage(
        (dispatch_id >> INDIRECT_PREPASS_SCALE_EXP) << INDIRECT_PREPASS_SCALE_EXP, beam_image);
    Raycast(ray_origin + indirect_t * indirect_light_dir, indirect_light_dir, indirect_light_res);

    const IndirectLightValue indirect_value = hash_set.GetId(indirect_light_res.leaf_ptr, in_set);

    float3 acc = 0.0f;
    acc += res.hit_color.rgb * light_impact * shadow_impact;
    acc += res.hit_color.rgb * indirect_value.indirect_light * directional_light_arr[0].intesity;

    image.Store(dispatch_id,
                lerp(float4(indirect_value.indirect_light * directional_light_arr[0].intesity, 1.0f),
                     image.Load(dispatch_id), 0.5f));
  } else {
    const float3 ground_color = float3(0.1f);
    const float3 sky_color_horizon = float3(0.8f);
    const float3 sky_color_zenith = float3(0.5f, 0.65f, 0.92f);

    const float sky_gradient_t = pow(smoothstep(0.0f, 0.4f, ray_dir.y), 0.35f);
    const float3 sky_gradient = lerp(sky_color_horizon, sky_color_zenith, sky_gradient_t);
    const float ground_to_sky_t = smoothstep(-0.01f, 0.0f, ray_dir.y);

    const float sun_focus = 100.0f;
    const float sun = pow(max(0, dot(ray_dir, -directional_light_arr[0].direction)), sun_focus) *
                      directional_light_arr[0].intesity;
    const float sun_mask = step(ground_to_sky_t, 1.0f);

    const float3 sky_color = lerp(ground_color, sky_gradient, ground_to_sky_t) + sun_mask * sun;
    image.Store(dispatch_id, float4(sky_color, 1.0f));
  }
}
