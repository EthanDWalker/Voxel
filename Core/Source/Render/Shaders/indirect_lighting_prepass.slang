#include "Include/common.slang"
#define VOXEL_TREE_SET_BINDING 2
#include "Include/camera.slang"
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
RWStructuredBuffer<HashSetHeader> hash_set_header;

[[vk::binding(HASH_SET_KEY_BINDING, 4)]]
RWStructuredBuffer<AtomicHashSetKey> hash_set_key_arr;

[[vk::binding(HASH_SET_LAST_KEY_BINDING, 4)]]
StructuredBuffer<HashSetKey> last_hash_set_key_arr;

[[vk::binding(HASH_SET_LIGHTING_DATA_BINDING, 4)]]
RWStructuredBuffer<LightingData> lighting_data_arr;

[[vk::binding(HASH_SET_LAST_LIGHTING_DATA_BINDING, 4)]]
RWStructuredBuffer<LightingData> last_lighting_data_arr;

[[vk::binding(HASH_SET_OCCLUSION_DATA_BINDING, 4)]]
RWStructuredBuffer<AtomicOcclusionData> occlusion_data_arr;

[[vk::binding(0, 5)]]
RWCountedBuffer<IndirectLightingRayDispatch> ray_dispatch_arr;

[[vk::binding(1, 5)]]
RWStructuredBuffer<RWDispatchCommand> ray_dispatch_cmd;

static const float ALPHA = 0.005f;

float3 GetPixelRayDir(const uint32_t2 pixel_index, const uint32_t2 image_size, const Camera camera) {
  const float4x4 inv_view_proj =
      mul(LookAtInverse(camera.position, camera.position + camera.front, camera.up),
          PerspectiveReverseZInverse(1.0f, 2.0f, camera.fov_y, camera.aspect_ratio));

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
  image_size = image_size >> INDIRECT_PREPASS_SCALE_EXP;

  if (all(image_size <= dispatch_id))
    return;

  const float4x4 inv_view_proj =
      mul(LookAtInverse(camera.position, camera.position + camera.front, camera.up),
          PerspectiveReverseZInverse(1.0f, 2.0f, camera.fov_y, camera.aspect_ratio));

  const float3 ray_dir = GetPixelRayDir(dispatch_id, image_size, camera);

  const float t = SampleBeamImage(dispatch_id << INDIRECT_PREPASS_SCALE_EXP, beam_image);

  RaycastResult res;
  Raycast(camera.position + t * ray_dir, ray_dir, res);

  if (res.hit) {

    const Optional<uint32_t> last_hash_set_index =
        HashSetFind(res.leaf_ptr, last_hash_set_key_arr, hash_set_header);

    const LightingData last_lighting_data =
        last_hash_set_index.hasValue ? last_lighting_data_arr[last_hash_set_index.value] : LightingData();

    const Optional<uint32_t> hash_set_index =
        HashSetFindOrInitialize(res.leaf_ptr, hash_set_key_arr, hash_set_header);
    if (!hash_set_index.hasValue) {
      return;
    }

    const bool claimed_shadow_ray = occlusion_data_arr[hash_set_index.value].data.compareExchange(0, 1) == 0;

    if (claimed_shadow_ray) {
      const float3 L = -directional_light_arr[0].direction;

      const float3 ray_origin = StepTreeRaySingle(GetWorldPositionCenter(res.hit_tree_index, res.hit_level),
                                                  res.hit_normal, res.hit_level);

      RaycastResult shadow_res;
      Raycast(ray_origin, L, shadow_res);
      if (shadow_res.hit) {
        occlusion_data_arr[hash_set_index.value].data.or(1 << 1);
      }

      if (last_hash_set_index.hasValue) {

        lighting_data_arr[hash_set_index.value].acc_lighting =
            last_lighting_data.acc_lighting * (1.0f - ALPHA);

        lighting_data_arr[hash_set_index.value].sample_count =
            last_lighting_data.sample_count * (1.0f - ALPHA);

      } else {
        const LightingData lighting_data = lighting_data_arr[hash_set_index.value];

        if (lighting_data.sample_count != 0.0f) {
          lighting_data_arr[hash_set_index.value].acc_lighting = lighting_data.acc_lighting * (1.0f - ALPHA);

          lighting_data_arr[hash_set_index.value].sample_count = lighting_data.sample_count * (1.0f - ALPHA);
        }
      }
    }

    uint32_t frame_sample_count;
    if (last_lighting_data.sample_count < 50) {
      frame_sample_count = 5;
    } else if (last_lighting_data.sample_count < 100) {
      frame_sample_count = 3;
    } else if (last_lighting_data.sample_count < 500) {
      frame_sample_count = 2;
    } else {
      frame_sample_count = 1;
    }

    IndirectLightingRayDispatch dispatch;
    dispatch.tree_index = res.hit_tree_index;

    for (uint32_t i = 0; i < frame_sample_count; i++) {
      if (ray_dispatch_arr.Append(dispatch)) {
        const uint32_t last_index = ray_dispatch_cmd[0].total_x.increment();

        if ((last_index % 64 == 63) && last_index != 0) {
          ray_dispatch_cmd[0].groups_x.increment();
        }
      }
    }
  }
}
