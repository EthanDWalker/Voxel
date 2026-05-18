#include "Include/common.slang"
#define VOXEL_TREE_SET_BINDING 2
#include "Include/camera.slang"
#include "Include/voxel_tree.slang"
#define HASH_SET_ATOMIC
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

float3 SampleFibonacciSphere(const uint32_t sample_index, const uint32_t sample_count) {
  const float PHI = PI * (sqrt(5.0f) - 1.0f);

  const float y = 1.0f - (sample_index / float(sample_count - 1.0f)) * 2.0f;
  const float radius = sqrt(1.0f - y * y);
  const float theta = PHI * sample_index;
  const float x = cos(theta) * radius;
  const float z = sin(theta) * radius;

  return float3(x, y, z);
}

float FloatHash(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return (float(x) * (1.0f / 4294967295.0f));
}

float2 Random2(const uint seed) { return float2(FloatHash(seed), FloatHash(seed * 1664525U + 1013904223U)); }

float3 TransformFromNormal(const float3 normal, const float3 local_dir) {
  const float3 up = abs(normal.z) < 0.999f ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);

  const float3 tangent = normalize(cross(up, normal));
  const float3 bitangent = cross(normal, tangent);

  return tangent * local_dir.x + bitangent * local_dir.y + normal * local_dir.z;
}

float3 CosineWeightedHemisphere(const float3 normal, const float2 rand) {
  const float2 offset = rand * 2.0f - 1.0f;

  float theta;
  float r;
  if (all(offset == 0.0f)) {
    r = 0.0f;
    theta = 0.0f;
  } else if (abs(offset.x) > abs(offset.y)) {
    r = offset.x;
    theta = (PI / 4.0f) * (offset.y / offset.x);
  } else {
    r = offset.y;
    theta = (PI / 2.0f) - (PI / 4.0f) * (offset.x / offset.y);
  }

  const float2 disk_point = r * float2(cos(theta), sin(theta));
  const float z = sqrt(max(0.0f, 1.0f + dot(disk_point, -disk_point)));

  const float3 local_dir = float3(disk_point, z);

  return lerp(TransformFromNormal(normal, local_dir), normal, 0.0);
}

float CastRay(float3 origin, float3 direction, uint32_t seed, float weight, const uint32_t max_bounces) {
  for (uint32_t i = 0;; i++) {
    RaycastResult res;
    Raycast(origin, direction, res);

    if (!res.hit) {
      return weight;
    } else if (i == max_bounces) {
      return 0.0f;
    }

    origin = StepTreeRaySingle(GetWorldPositionCenter(res.hit_tree_index), res.hit_normal, res.hit_level);
    seed = Hash(seed ^ (res.leaf_ptr) ^ (i * 4312u));
    direction = CosineWeightedHemisphere(res.hit_normal, Random2(seed));
    weight = dot(res.hit_normal, direction) / PI;
  }
}

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
void CMain(const uint32_t2 dispatch_id: SV_DispatchThreadID, const uniform uint32_t frame_number) {
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

  const float3 ray_origin =
      StepTreeRaySingle(GetWorldPositionCenter(res.hit_tree_index), res.hit_normal, res.hit_level);

  if (res.hit) {
    bool last_in_set;
    const IndirectLightValue value = last_hash_set.GetId(res.leaf_ptr, last_in_set);

    if (hash_set.ClaimShadowRay(res.leaf_ptr)) {
      const float3 L = -directional_light_arr[0].direction;

      RaycastResult shadow_res;
      Raycast(ray_origin, L, shadow_res);
      if (shadow_res.hit) {
        hash_set.SetInShadow(res.leaf_ptr);
      }

      if (last_in_set) {
        hash_set.StoreIndirectLight(res.leaf_ptr, value.lighting, value.sample_count);
      }
    }

    uint32_t frame_sample_count = 1;

    float indirect_light = 0.0f;
    for (uint32_t i = 0; i < frame_sample_count; i++) {
      const uint32_t seed = Hash(res.leaf_ptr ^ (frame_number * 9781u) ^
                                 (dispatch_id.x * dispatch_id.y * 32197u) ^ (i * 421351u));
      const float3 indirect_lighting_dir = CosineWeightedHemisphere(res.hit_normal, Random2(seed));
      const float weight = dot(res.hit_normal, indirect_lighting_dir) / PI;

      const uint32_t max_bounces = 1;
      indirect_light += CastRay(ray_origin, indirect_lighting_dir, seed, weight, max_bounces);
    }

    hash_set.StoreIndirectLight(res.leaf_ptr, indirect_light, frame_sample_count);
  }
}
