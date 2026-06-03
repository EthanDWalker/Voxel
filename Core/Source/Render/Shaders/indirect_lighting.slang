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

[[vk::binding(0, 5)]]
CountedBuffer<IndirectLightingRayDispatch> ray_dispatch_arr;

[[vk::binding(1, 5)]]
StructuredBuffer<DispatchCommand> ray_dispatch_cmd;

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

float3 CastRay(float3 origin, float3 direction, uint32_t seed, float weight, const uint32_t max_bounces) {
  float3 color = 1.0f;
  for (uint32_t i = 0;; i++) {
    if (weight < 0.10f) {
      return 0.0f;
    }
    RaycastResult res;
    Raycast(origin, direction, res);
    if (res.hit) {
      if (i == max_bounces) {
        return 0.0f;
      }
      color *= res.hit_color.rgb;
    } else {
      return color * SampleEnvironmentColor(direction, directional_light_arr[0]) * weight;
    }

    origin = StepTreeRaySingle(GetWorldPositionCenter(res.hit_tree_index, res.hit_level), res.hit_normal,
                               res.hit_level);
    seed = Hash(seed ^ (res.leaf_ptr) ^ (i * 4312u));
    direction = CosineWeightedHemisphere(res.hit_normal, Random2(seed));
    weight = dot(res.hit_normal, direction) / PI;
  }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void CMain(const uint32_t dispatch_id: SV_DispatchThreadID, const uniform uint32_t frame_number) {
  if (dispatch_id >= ray_dispatch_arr.count || dispatch_id >= ray_dispatch_cmd[0].total_x)
    return;

  const IndirectLightingRayDispatch dispatch = ray_dispatch_arr[dispatch_id];

  uint32_t level;
  uint32_t leaf_ptr;
  const LeafNode leaf_node = GetLeafNodeAtFilledIndex(dispatch.tree_index, leaf_ptr, level);

  const float3 normal = leaf_node.normal;

  const float3 ray_origin =
      StepTreeRaySingle(GetWorldPositionCenter(dispatch.tree_index, level), leaf_node.normal, level);

  const uint32_t seed = Hash(leaf_ptr ^ (frame_number * 9781u) ^ (dispatch_id.x * dispatch_id.x * 32197u));
  const float3 indirect_lighting_dir = CosineWeightedHemisphere(normal, Random2(seed));
  const float weight = dot(normal, indirect_lighting_dir) / PI;

  const uint32_t max_bounces = 0;
  const float3 indirect_light = CastRay(ray_origin, indirect_lighting_dir, seed, weight, max_bounces);

  bool in_set;
  const uint32_t hash_set_index = hash_set.Find(leaf_ptr, in_set);
  if (!in_set) {
    return;
  }

  hash_set.set[hash_set_index].sample_count += 1.0f;
  hash_set.set[hash_set_index].lighting += PackIndirectLighting(indirect_light);
}
