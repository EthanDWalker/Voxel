#include "Include/common.slang"
#define VOXEL_TREE_SET_BINDING 2
#include "Include/camera.slang"
#include "Include/color.slang"
#include "Include/hash_set.slang"
#include "Include/pbr.slang"
#include "Include/voxel_tree.slang"

[[vk::binding(0, 0)]]
RWTexture2D<float4> image;
[[vk::binding(1, 0)]]
RWTexture2D<float> beam_image;

[[vk::binding(0, 1)]]
ConstantBuffer<Camera> camera;

[[vk::binding(1, 1)]]
ConstantBuffer<Camera> last_camera;

[[vk::binding(HASH_SET_HEADER_BINDING, 3)]]
RWStructuredBuffer<HashSetHeader> hash_set_header;

[[vk::binding(HASH_SET_KEY_BINDING, 3)]]
RWStructuredBuffer<AtomicHashSetKey> hash_set_key_arr;

[[vk::binding(HASH_SET_LAST_KEY_BINDING, 3)]]
StructuredBuffer<HashSetKey> last_hash_set_key_arr;

[[vk::binding(HASH_SET_DIFFUSE_DATA_BINDING, 3)]]
RWStructuredBuffer<AtomicLightingData> diffuse_data_arr;

[[vk::binding(HASH_SET_LAST_DIFFUSE_DATA_BINDING, 3)]]
RWStructuredBuffer<LightingData> last_diffuse_data_arr;

[[vk::binding(HASH_SET_SPECULAR_DATA_BINDING, 3)]]
RWStructuredBuffer<AtomicLightingData> specular_data_arr;

[[vk::binding(HASH_SET_LAST_SPECULAR_DATA_BINDING, 3)]]
RWStructuredBuffer<LightingData> last_specular_data_arr;

float3 GetPixelRayDir(const uint32_t2 pixel_index, const uint32_t2 image_size, const float4x4 inv_view_proj) {
  const float2 pixel_center_ndc = float2(float(pixel_index.x + 0.5f) / image_size.x * 2.0f - 1.0f,
                                         float(pixel_index.y + 0.5f) / image_size.y * -2.0f + 1.0f);
  const float4 pixel_center_world = mul(inv_view_proj, float4(pixel_center_ndc, 1.0f, 1.0f));
  return normalize(pixel_center_world.xyz / pixel_center_world.w - camera.position);
}

float FloatHash(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return (float(x) * (1.0f / 4294967295.0f));
}

float2 UniformRandom2(const uint seed) {
  return float2(FloatHash(seed), FloatHash(seed * 1664525U + 1013904223U));
}

float3 TransformFromNormalSpace(const float3 normal, const float3 local_dir) {
  const float3 up = abs(normal.y) < 0.999f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);

  const float3 tangent = normalize(cross(up, normal));
  const float3 bitangent = cross(normal, tangent);

  return normalize(tangent * local_dir.x + normal * local_dir.y + bitangent * local_dir.z);
}

float3 TransformToNormalSpace(const float3 normal, const float3 world_dir) {
  const float3 up = abs(normal.y) < 0.999f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);

  const float3 tangent = normalize(cross(up, normal));
  const float3 bitangent = cross(normal, tangent);

  return normalize(float3(dot(world_dir, tangent), dot(world_dir, normal), dot(world_dir, bitangent)));
}

float3 SampleCosineWeightedHemisphere(const float3 normal, const float2 rand) {
  const float r = sqrt(rand.x);
  const float theta = 2 * PI * rand.y;

  const float2 disk_point = r * float2(cos(theta), sin(theta));

  const float3 local_dir = float3(disk_point.x, sqrt(max(0.0f, 1.0f - r * r)), disk_point.y);

  return TransformFromNormalSpace(normal, local_dir);
}

// w_o away from surface
float3 SampleVNDF_GGX(const float3 normal, float3 w_o, const float roughness, const float2 rand) {
  w_o = TransformToNormalSpace(normal, w_o);
  const float3 v = normalize(float3(w_o.x * roughness, w_o.y, w_o.z * roughness));

  const float3 t_1 =
      (v.y < 0.999f) ? normalize(cross(v, float3(0.0f, 1.0f, 0.0f))) : float3(1.0f, 0.0f, 0.0f);
  const float3 t_2 = cross(t_1, v);

  const float a = 1.0f / (1.0f + v.y);
  const float r = sqrt(rand.x);
  const float theta = (rand.y < a) ? (rand.y / a) * PI : PI + (rand.y - a) / (1.0f - a) * PI;

  const float p_1 = r * cos(theta);
  const float p_2 = r * sin(theta) * ((rand.y < a) ? 1.0f : v.y);

  const float3 n = p_1 * t_1 + p_2 * t_2 + sqrt(max(0.0f, 1.0f - p_1 * p_1 - p_2 * p_2)) * v;

  return TransformFromNormalSpace(normal,
                                  normalize(float3(roughness * n.x, max(0.0f, n.y), roughness * n.z)));
}

float3 SampleSpecularLight(float3 origin, const float3 view_dir, Material material, float3 normal,
                           inout uint32_t seed) {
  const float a2 = material.rough * material.rough;
  float3 reflectance;

  const float3 micro_normal = SampleVNDF_GGX(normal, -view_dir, material.rough, UniformRandom2(seed));

  const float3 ray_dir = reflect(view_dir, micro_normal);

  if (dot(normal, ray_dir) > 0.0f) {
    const float3 f_0 = 0.16f * (material.reflect * material.reflect) * (1.0f - material.metallic) +
                       material.albedo.rgb * material.metallic;

    const float3 F = F_Schlick(f_0, dot(ray_dir, micro_normal));
    const float G_1 = max(G_1_SmithGGX(-view_dir, normal, a2), 1e-6f);
    const float G_2 = G_2_SmithGGX(ray_dir, -view_dir, normal, a2);

    reflectance = F * saturate(G_2 / G_1);
  } else {
    return 0.0f;
  }

  float3 radiance = 0.0f;
  float3 throughput = reflectance;

  RaycastResult res;
  Raycast(origin, ray_dir, res);

  if (res.hit) {
    material = material_arr[res.hit_node.material_index];
    radiance += throughput * material.emissive * material.albedo.rgb;
  } else {
    radiance += throughput * SampleEnvironmentColor(ray_dir);
  }

  return radiance;
}

struct EmissiveSample {
  uint32_t3 tree_index;
  float pdf;
  uint32_t visibility_direction_index;
}

uint32_t GetNthSetBitIndex(uint32_t value, uint32_t n) {
  while (bool(n--)) {
    if (value == 0)
      return 0xFFFFFFFF;

    value &= value - 1;
  }

  return bool(value) ? firstbitlow(value) : 0xFFFFFFFF;
}

// index is the min index of the bound but still inside
float GetSampleWeight(const float3 position, const float3 normal, const uint32_t3 index,
                      const uint32_t level) {
  const float3 min_bound = index * voxel_size + voxel_tree_header.min_bound;
  const float3 max_bound =
      (1 + index + (1 << ((voxel_tree_header.max_depth - (level + 1)) << 1))) * voxel_size +
      voxel_tree_header.min_bound;

  const float3 center = GetWorldPositionCenter(index, level);
  const float3 offset = center - position;

  if (all(position >= min_bound && position <= max_bound)) {
    // in bounds so cone cos_theta cant be computed
    return 1.0f / dot(offset, offset);
  } else {
    const float cos_theta =
        max(dot(normalize(min_bound - position), normal), dot(normalize(max_bound - position), normal));
    return saturate(cos_theta) / dot(offset, offset);
  }
}

Optional<EmissiveSample> GetRandomVisibleEmissive(const float3 origin, const float3 normal,
                                                  inout uint32_t seed) {
  float rand = FloatHash(seed);
  float pdf = 1.0f;

  float total_weight = 0.0f;
  for (uint32_t i = 0; i < 64; i++) {
    total_weight +=
        voxel_tree[0][i].luminance *
        GetSampleWeight(origin, normal, GetLocalOffset(i) << ((voxel_tree_header.max_depth - 1) << 1), 0);
  }

  if (total_weight == 0.0f) {
    return none;
  }

  rand = max((rand * total_weight) - 1e-5f, 0.0f);

  float acc_weight = 0.0f;
  uint32_t local_index;
  BranchNode node;
  for (uint32_t i = 0; i < 64; i++) {
    node = voxel_tree[0][i];
    const float weight =
        node.luminance *
        GetSampleWeight(origin, normal, GetLocalOffset(i) << ((voxel_tree_header.max_depth - 1) << 1), 0);
    acc_weight += weight;
    if (rand <= acc_weight) {
      local_index = i;
      pdf *= (weight / total_weight);
      break;
    }
  }

  if (node.luminance == 0) {
    return none;
  }

  uint32_t3 index = GetLocalOffset(local_index) << ((voxel_tree_header.max_depth - 1) << 1);
  for (uint32_t level = 1;; level++) {
    seed = Hash(seed * local_index);

    const uint32_t page_index = node.child_ptr_data.child_ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset =
        node.child_ptr_data.child_ptr - (page_index << voxel_tree_header.page_size_exp);

    const uint32_t level_shift = (voxel_tree_header.max_depth - (level + 1)) << 1;
    total_weight = 0.0f;

    uint32_t child_mask_low = (uint32_t)node.child_mask;
    uint32_t child_mask_high = (uint32_t)(node.child_mask >> 32);

    while (bool(child_mask_low)) {
      const uint32_t i = firstbitlow(child_mask_low);
      total_weight += voxel_tree[page_index][page_offset + i].luminance *
                      GetSampleWeight(origin, normal, index + (GetLocalOffset(i) << level_shift), level);
      child_mask_low &= child_mask_low - 1;
    }

    while (bool(child_mask_high)) {
      const uint32_t i = firstbitlow(child_mask_high) + 32;
      total_weight += voxel_tree[page_index][page_offset + i].luminance *
                      GetSampleWeight(origin, normal, index + (GetLocalOffset(i) << level_shift), level);
      child_mask_high &= child_mask_high - 1;
    }

    acc_weight = 0.0f;
    rand = max((FloatHash(seed) * total_weight) - 1e-5f, 0.0f);

    child_mask_low = (uint32_t)node.child_mask;
    child_mask_high = (uint32_t)(node.child_mask >> 32);
    while (bool(child_mask_low)) {
      const uint32_t i = firstbitlow(child_mask_low);
      node = voxel_tree[page_index][page_offset + i];
      const float weight =
          node.luminance * GetSampleWeight(origin, normal, index + (GetLocalOffset(i) << level_shift), level);
      acc_weight += weight;
      if (rand <= acc_weight) {
        local_index = i;
        pdf *= (weight / total_weight);
        child_mask_high = 0;
        break;
      }

      child_mask_low &= child_mask_low - 1;
    }

    while (bool(child_mask_high)) {
      const uint32_t i = firstbitlow(child_mask_high) + 32;
      node = voxel_tree[page_index][page_offset + i];
      const float weight =
          node.luminance * GetSampleWeight(origin, normal, index + (GetLocalOffset(i) << level_shift), level);
      acc_weight += weight;
      if (rand <= acc_weight) {
        local_index = i;
        pdf *= (weight / total_weight);
        break;
      }

      child_mask_high &= child_mask_high - 1;
    }

    index += GetLocalOffset(local_index) << level_shift;

    if (node.luminance == 0) {
      return none;
    }

    if (node.child_ptr_data.is_leaf) {
      break;
    }
  }

  const uint32_t leaf_page_index = node.child_ptr_data.child_ptr >> voxel_tree_header.page_size_exp;
  const uint32_t leaf_page_offset =
      node.child_ptr_data.child_ptr - (leaf_page_index << voxel_tree_header.page_size_exp);

  acc_weight = 0.0f;
  total_weight = node.luminance;
  seed = Hash(seed * local_index);
  rand = max((FloatHash(seed) * total_weight) - 1e-5f, 0.0f);
  LeafNode leaf_node = {};
  uint32_t direction_index = 0;
  for (uint32_t i = 0; i < 64; i++) {
    leaf_node = voxel_tree_leaf[leaf_page_index][leaf_page_offset + i];
    if (leaf_node.visibility_mask != 0) {
      const float weight =
          material_arr[leaf_node.material_index].emissive * countbits(leaf_node.visibility_mask);
      acc_weight += weight;

      if (rand <= acc_weight) {
        local_index = i;
        seed = Hash(seed * local_index);
        rand = max((FloatHash(seed) * countbits(leaf_node.visibility_mask)) - 1e-5f, 0.0f);

        const uint32_t bit_index = (uint32_t)floor(rand);
        direction_index = GetNthSetBitIndex(leaf_node.visibility_mask, bit_index);

        break;
      }
    }
  }

  index += GetLocalOffset(local_index);

  EmissiveSample emissive_sample;
  emissive_sample.pdf = pdf;
  emissive_sample.tree_index = index;
  emissive_sample.visibility_direction_index = direction_index;

  return emissive_sample;
}

float3 LightSampleDiffuseLight(const float3 origin, const float3 normal, inout uint32_t seed) {
  float3 throughput = 1.0f;
  float3 radiance = 0.0f;

  const Optional<EmissiveSample> sample_opt = GetRandomVisibleEmissive(origin, normal, seed);
  if (!sample_opt.hasValue) {
    return 0.0f;
  }

  const float3 emissive_normal = visibility_direction_arr[sample_opt.value.visibility_direction_index];
  float3 sample_pos = GetWorldPositionCenter(sample_opt.value.tree_index, voxel_tree_header.max_depth - 1);
  sample_pos += ((voxel_size / 2.0f)) * emissive_normal;

  const float3 sample_dir = normalize(sample_pos - origin);

  const float cos2_theta = dot(sample_dir, normal) * dot(-sample_dir, emissive_normal);

  if (cos2_theta <= 0.0f) {
    return 0.0f;
  }

  const float distance = length(sample_pos - origin);

  RaycastResult res;
  Raycast(origin, sample_dir, res, distance);

  if (res.hit) {
    if (res.t + 1e-1f >= distance && all(res.hit_normal == emissive_normal)) {
      const Material material = material_arr[res.hit_node.material_index];
      radiance = (material.albedo.rgb * material.emissive * cos2_theta) /
                 (PI * sample_opt.value.pdf * distance * distance);
    }
  }

  return radiance;
}

float3 BRDFSampleDiffuseLight(float3 origin, float3 normal, const uint32_t max_bounce_count,
                              inout uint32_t seed) {
  float3 throughput = 1.0f;
  float3 radiance = 0.0f;

  for (uint32_t i = 0; i < max_bounce_count; i++) {
    const float3 ray_dir = SampleCosineWeightedHemisphere(normal, UniformRandom2(seed));

    RaycastResult res;
    Raycast(origin, ray_dir, res);

    if (!res.hit) {
      // radiance += throughput * SampleEnvironmentColor(ray_dir);
      break;
    }

    const Material material = material_arr[res.hit_node.material_index];
    normal = res.hit_normal;

    radiance += throughput * material.emissive * material.albedo.rgb;
    throughput *= material.albedo.rgb * (1.0f - material.metallic);

    if (all(throughput < 1e-2f) || material.emissive != 0.0f) {
      break;
    }

    seed = Hash(seed ^ (res.leaf_ptr) ^ (i * 4312u));
    origin =
        StepTreeRaySingle(GetWorldPositionCenter(res.hit_tree_index, res.hit_level), normal, res.hit_level);
  }

  return radiance;
}

struct PushConstants {
  float diffuse_alpha;
  float specular_alpha;
  uint32_t frame_number;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void CMain(const uint32_t2 dispatch_id: SV_DispatchThreadID, const uniform PushConstants info) {
  uint32_t2 image_size;
  image.GetDimensions(image_size.x, image_size.y);

  if (all(image_size <= dispatch_id))
    return;

  const float4x4 inv_view_proj =
      mul(LookAtInverse(camera.position, camera.position + camera.front, camera.up),
          PerspectiveReverseZInverse(1.0f, 2.0f, camera.fov_y, camera.aspect_ratio));

  const float3 ray_dir = GetPixelRayDir(dispatch_id << INDIRECT_PREPASS_SCALE_EXP, image_size, inv_view_proj);

  const float t = SampleBeamImage(dispatch_id << INDIRECT_PREPASS_SCALE_EXP, beam_image);

  RaycastResult res;
  Raycast(camera.position + t * ray_dir, ray_dir, res);

  if (res.hit) {
    const float hit_distance =
        length(GetWorldPositionCenter(res.hit_tree_index, res.hit_level) - camera.position);
    const float last_hit_distance =
        length(GetWorldPositionCenter(res.hit_tree_index, res.hit_level) - last_camera.position);

    const Optional<uint32_t> last_hash_set_index =
        HashSetFind(PackTreeIndex(res.hit_tree_index & GetLodTreeMask(last_hit_distance)),
                    last_hash_set_key_arr, hash_set_header);

    const LightingData last_diffuse_data =
        last_hash_set_index.hasValue ? last_diffuse_data_arr[last_hash_set_index.value] : LightingData();
    const LightingData last_specular_data =
        last_hash_set_index.hasValue ? last_specular_data_arr[last_hash_set_index.value] : LightingData();

    bool initialized;
    const Optional<uint32_t> hash_set_index =
        HashSetFindOrInitialize(PackTreeIndex(res.hit_tree_index & GetLodTreeMask(hit_distance)), initialized,
                                hash_set_key_arr, hash_set_header);

    if (!hash_set_index.hasValue) {
      // failed to initialize
      return;
    }

    if (initialized) {
      if (last_hash_set_index.hasValue) {
        diffuse_data_arr[hash_set_index.value].acc_lighting_r +=
            last_diffuse_data.acc_lighting.r * (1.0f - info.diffuse_alpha);
        diffuse_data_arr[hash_set_index.value].acc_lighting_g +=
            last_diffuse_data.acc_lighting.g * (1.0f - info.diffuse_alpha);
        diffuse_data_arr[hash_set_index.value].acc_lighting_b +=
            last_diffuse_data.acc_lighting.b * (1.0f - info.diffuse_alpha);
        diffuse_data_arr[hash_set_index.value].sample_count +=
            last_diffuse_data.sample_count * (1.0f - info.diffuse_alpha);

        specular_data_arr[hash_set_index.value].acc_lighting_r +=
            last_specular_data.acc_lighting.r * (1.0f - info.specular_alpha);
        specular_data_arr[hash_set_index.value].acc_lighting_g +=
            last_specular_data.acc_lighting.g * (1.0f - info.specular_alpha);
        specular_data_arr[hash_set_index.value].acc_lighting_b +=
            last_specular_data.acc_lighting.b * (1.0f - info.specular_alpha);
        specular_data_arr[hash_set_index.value].sample_count +=
            last_specular_data.sample_count * (1.0f - info.specular_alpha);
      }
    }

    const float3 voxel_center = GetWorldPositionCenter(res.hit_tree_index, res.hit_level);
    const float3 ray_origin = StepTreeRaySingle(voxel_center, res.hit_normal, res.hit_level);

    uint32_t seed = Hash(res.leaf_ptr ^ ((info.frame_number + 1) * 9781u) ^
                         ((dispatch_id.x + 1) * (dispatch_id.y + 1) * 32197u));

    const Material material = material_arr[res.hit_node.material_index];

    {
      const uint32_t max_bounce_count = 1;

      const float3 diffuse_light_sample =
          false ? BRDFSampleDiffuseLight(ray_origin, res.hit_normal, max_bounce_count, seed)
                : LightSampleDiffuseLight(ray_origin, res.hit_normal, seed);

      diffuse_data_arr[hash_set_index.value].sample_count += 1.0f;
      diffuse_data_arr[hash_set_index.value].acc_lighting_r += diffuse_light_sample.r;
      diffuse_data_arr[hash_set_index.value].acc_lighting_g += diffuse_light_sample.g;
      diffuse_data_arr[hash_set_index.value].acc_lighting_b += diffuse_light_sample.b;
    }

    if (false) {
      const float3 specular_light_sample =
          SampleSpecularLight(ray_origin, ray_dir, material, res.hit_normal, seed);
      specular_data_arr[hash_set_index.value].sample_count += 1.0f;
      specular_data_arr[hash_set_index.value].acc_lighting_r += specular_light_sample.r;
      specular_data_arr[hash_set_index.value].acc_lighting_g += specular_light_sample.g;
      specular_data_arr[hash_set_index.value].acc_lighting_b += specular_light_sample.b;
    }
  }
}
