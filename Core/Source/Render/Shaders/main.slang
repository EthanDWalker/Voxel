#include "Include/common.slang"
#define VOXEL_TREE_SET_BINDING 2
#include "Include/camera.slang"
#include "Include/color.slang"
#include "Include/hash_set.slang"
#include "Include/voxel_tree.slang"

[[vk::binding(0, 0)]]
RWTexture2D<float4> image;
[[vk::binding(1, 0)]]
RWTexture2D<float> beam_image;

[[vk::binding(0, 1)]]
ConstantBuffer<Camera> camera;

[[vk::binding(0, 3)]]
CountedBuffer<DirectionalLight> directional_light_arr;

[[vk::binding(HASH_SET_HEADER_BINDING, 4)]]
StructuredBuffer<HashSetHeader> hash_set_header;

[[vk::binding(HASH_SET_KEY_BINDING, 4)]]
StructuredBuffer<HashSetKey> hash_set_key_arr;

[[vk::binding(HASH_SET_LIGHTING_DATA_BINDING, 4)]]
StructuredBuffer<LightingData> lighting_data_arr;

[[vk::binding(HASH_SET_OCCLUSION_DATA_BINDING, 4)]]
StructuredBuffer<OcclusionData> occlusion_data_arr;

struct FrameLuminanceData {
  Atomic<float> acc_log_luminance;
  Atomic<uint32_t> sample_count;
};

[[vk::binding(0, 5)]]
RWStructuredBuffer<FrameLuminanceData> frame_luminance_data;

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

  float3 acc = 0.0f;
  if (res.hit) {
    const Optional<uint32_t> hash_set_index = HashSetFind(res.leaf_ptr, hash_set_key_arr, hash_set_header);

    {
      const float3 L = -directional_light_arr[0].direction;
      const float light_impact = max(dot(L, res.hit_normal), 0.0f) * directional_light_arr[0].intensity;
      float shadow_impact = 1.0f;

      const OcclusionData occlusion_data = occlusion_data_arr[hash_set_index.value];
      if (hash_set_index.hasValue && !occlusion_data.null) {
        shadow_impact = float(occlusion_data.in_shadow);
      } else {
        RaycastResult shadow_res;
        const float3 shadow_ray_origin = StepTreeRaySingle(
            GetWorldPositionCenter(res.hit_tree_index, res.hit_level), res.hit_normal, res.hit_level);
        Raycast(shadow_ray_origin, L, shadow_res);
        shadow_impact = float(shadow_res.hit);
      }

      shadow_impact = max(1.0f - shadow_impact, 0.0f);

      acc += res.hit_color.rgb * shadow_impact * light_impact;
    }

    {

      LightingData lighting_data;

      if (hash_set_index.hasValue) {
        lighting_data = lighting_data_arr[hash_set_index.value];
      } else {
        RaycastResult indirect_light_res;
        const float3 indirect_light_dir =
            GetPixelRayDir(dispatch_id >> INDIRECT_PREPASS_SCALE_EXP,
                           image_size >> INDIRECT_PREPASS_SCALE_EXP, inv_view_proj);
        const float indirect_t = SampleBeamImage(
            (dispatch_id >> INDIRECT_PREPASS_SCALE_EXP) << INDIRECT_PREPASS_SCALE_EXP, beam_image);
        Raycast(ray_origin + indirect_t * indirect_light_dir, indirect_light_dir, indirect_light_res);

        const Optional<uint32_t> closest_hash_set_index =
            HashSetFind(indirect_light_res.leaf_ptr, hash_set_key_arr, hash_set_header);

        if (closest_hash_set_index.hasValue) {
          lighting_data = lighting_data_arr[closest_hash_set_index.value];
        }
      }

      acc += res.hit_color.rgb * (lighting_data.acc_lighting / max(lighting_data.sample_count, 1.0f)) *
             directional_light_arr[0].intensity;
    }

  } else {
    acc = SampleEnvironmentColor(ray_dir, directional_light_arr[0]);
  }

  image.Store(dispatch_id, float4(float3(acc), 1.0f));

  static groupshared float shared_acc_log_luminance[64];

  const float luminance = GetLuminance(acc);

  shared_acc_log_luminance[memory_index] =
      saturate((log2(luminance) - MIN_LOG_LUMINANCE) * INV_LOG_LUMINANCE_RANGE);

  GroupMemoryBarrierWithGroupSync();

  for (uint32_t stride = 32; stride > 0; stride >>= 1) {
    if (memory_index < stride)
      shared_acc_log_luminance[memory_index] += shared_acc_log_luminance[memory_index + stride];

    GroupMemoryBarrierWithGroupSync();
  }

  if (memory_index == 0) {
    frame_luminance_data[0].sample_count += 64;
    frame_luminance_data[0].acc_log_luminance += shared_acc_log_luminance[0];
  }
}
