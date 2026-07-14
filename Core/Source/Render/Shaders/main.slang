#include "Include/common.slang"
#define VOXEL_TREE_SET_BINDING 2
#include "Include/camera.slang"
#include "Include/color.slang"
#include "Include/hash_set.slang"
#include "Include/noise.slang"
#include "Include/pbr.slang"
#include "Include/voxel_tree.slang"

[[vk::binding(0, 0)]]
RWTexture2D<float4> image;
[[vk::binding(1, 0)]]
RWTexture2D<float> beam_image;

[[vk::binding(0, 1)]]
ConstantBuffer<Camera> camera;

[[vk::binding(HASH_SET_HEADER_BINDING, 3)]]
StructuredBuffer<HashSetHeader> hash_set_header;

[[vk::binding(HASH_SET_KEY_BINDING, 3)]]
StructuredBuffer<HashSetKey> hash_set_key_arr;

[[vk::binding(HASH_SET_LAST_KEY_BINDING, 3)]]
StructuredBuffer<HashSetKey> last_hash_set_key_arr;

[[vk::binding(HASH_SET_DIFFUSE_DATA_BINDING, 3)]]
StructuredBuffer<LightingData> diffuse_data_arr;

[[vk::binding(HASH_SET_LAST_DIFFUSE_DATA_BINDING, 3)]]
RWStructuredBuffer<LightingData> last_diffuse_data_arr;

[[vk::binding(HASH_SET_SPECULAR_DATA_BINDING, 3)]]
StructuredBuffer<LightingData> specular_data_arr;

[[vk::binding(HASH_SET_LAST_SPECULAR_DATA_BINDING, 3)]]
RWStructuredBuffer<LightingData> last_specular_data_arr;

struct FrameLuminanceData {
  Atomic<float> acc_log_luminance;
  Atomic<float> total_weight;
};

[[vk::binding(0, 4)]]
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

  const float hit_distance =
      length(GetWorldPositionCenter(res.hit_tree_index, res.hit_level) - camera.position);

  float3 acc = 0.0f;
  if (res.hit) {
    const Optional<uint32_t> hash_set_index =
        HashSetFind(PackTreeIndex(res.hit_tree_index), hash_set_key_arr, hash_set_header);

    const Material hit_material = material_arr[res.hit_node.material_index];
    acc += hit_material.albedo.rgb * hit_material.emissive;

    {
      LightingData diffuse_data;
      LightingData specular_data;

      if (hash_set_index.hasValue) {
        diffuse_data = diffuse_data_arr[hash_set_index.value];
        specular_data = specular_data_arr[hash_set_index.value];
      } else {
        diffuse_data = {};
        specular_data = {};
      }

      acc += hit_material.albedo.rgb * (1.0f - hit_material.metallic) *
             (diffuse_data.acc_lighting / max(diffuse_data.sample_count, 1.0f));
      // acc += hit_material.albedo.rgb * (specular_data.acc_lighting /
      // max(specular_data.sample_count, 1.0f));
    }
  } else {
    acc = SampleEnvironmentColor(ray_dir);
  }

  image.Store(dispatch_id, float4(float3(acc), 1.0f));

  static groupshared float shared_acc_log_luminance[64];
  static groupshared float shared_total_weight[64];

  shared_total_weight[memory_index] = WeightLuminanceSample(dispatch_id, image_size);

  const float log_luminance = log2(GetLuminance(acc)) * shared_total_weight[memory_index];

  shared_acc_log_luminance[memory_index] =
      saturate((log_luminance - MIN_LOG_LUMINANCE) * INV_LOG_LUMINANCE_RANGE);

  GroupMemoryBarrierWithGroupSync();

  for (uint32_t stride = 32; stride > 0; stride >>= 1) {
    if (memory_index < stride) {
      shared_acc_log_luminance[memory_index] += shared_acc_log_luminance[memory_index + stride];
      shared_total_weight[memory_index] += shared_total_weight[memory_index + stride];
    }

    GroupMemoryBarrierWithGroupSync();
  }

  if (memory_index == 0) {
    frame_luminance_data[0].total_weight += shared_total_weight[0];
    frame_luminance_data[0].acc_log_luminance += shared_acc_log_luminance[0];
  }
}
