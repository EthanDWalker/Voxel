#include "Include/common.slang"
#define VOXEL_TREE_ATOMIC_EMISSIVE
#define VOXEL_TREE_READ_WRITE
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
[[vk::binding(1, 1)]]
ConstantBuffer<Camera> last_camera;

[[vk::binding(HASH_SET_HEADER_BINDING, 3)]]
RWStructuredBuffer<HashSetHeader> hash_set_header;

[[vk::binding(HASH_SET_KEY_BINDING, 3)]]
RWStructuredBuffer<AtomicHashSetKey> hash_set_key_arr;

[[vk::binding(HASH_SET_LAST_KEY_BINDING, 3)]]
StructuredBuffer<HashSetKey> last_hash_set_key_arr;

[[vk::binding(HASH_SET_DIFFUSE_DATA_BINDING, 3)]]
RWStructuredBuffer<LightingData> diffuse_data_arr;

[[vk::binding(HASH_SET_LAST_DIFFUSE_DATA_BINDING, 3)]]
RWStructuredBuffer<LightingData> last_diffuse_data_arr;

[[vk::binding(HASH_SET_SPECULAR_DATA_BINDING, 3)]]
RWStructuredBuffer<LightingData> specular_data_arr;

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

void GetNormalBasis(const float3 normal, out float3 tangent, out float3 bitangent) {
  const float3 up = abs(normal.y) < 0.999f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);

  tangent = normalize(cross(up, normal));
  bitangent = cross(normal, tangent);
}

float3 TransformFromNormalSpace(const float3 normal, const float3 local_dir) {
  float3 tangent;
  float3 bitangent;
  GetNormalBasis(normal, tangent, bitangent);

  return normalize(tangent * local_dir.x + normal * local_dir.y + bitangent * local_dir.z);
}

float3 TransformToNormalSpace(const float3 normal, const float3 world_dir) {
  float3 tangent;
  float3 bitangent;
  GetNormalBasis(normal, tangent, bitangent);

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

uint32_t GetNthSetBitIndex(uint32_t value, uint32_t n) {
  while (bool(n--)) {
    value &= value - 1;
  }

  return firstbitlow(value);
}

static const uint32_t SENTINAL = 0xFFFFFFFF;

struct LightSample {
  uint32_t3 tree_index;
  Optional<LeafNode> leaf_node_opt;
};

LightSample SampleLights(inout uint32_t seed) {
  LightSample light_sample;
  light_sample.leaf_node_opt = none;

  const uint32_t sampled_index = (uint32_t)(FloatHash(seed) * voxel_tree_header.emissive_leaf_count);
  seed ^= Hash(seed * sampled_index);
  if (sampled_index == voxel_tree_header.emissive_leaf_count || voxel_tree_header.emissive_leaf_count == 0) {
    return light_sample;
  }

  const uint32_t sampled_page_index = sampled_index >> voxel_tree_header.page_size_exp;
  const uint32_t sampled_page_offset =
      sampled_index - (sampled_page_index << voxel_tree_header.page_size_exp);

  const uint32_t sampled_value =
      voxel_tree_emissive_leaf[sampled_page_index][sampled_page_offset].packed_tree_index.load();

  if (sampled_value == SENTINAL) {
    return light_sample;
  }

  const uint32_t3 sampled_tree_index = UnpackTreeIndex(sampled_value);

  uint32_t ptr, level;
  const Optional<LeafNode> node_opt = GetLeafNodeAtIndex(sampled_tree_index, ptr, level);

  if (!node_opt.hasValue) {
    const uint32_t ret =
        voxel_tree_emissive_leaf[sampled_page_index][sampled_page_offset].packed_tree_index.compareExchange(
            sampled_value, SENTINAL);

    if (ret == sampled_value) {
      // get the tail
      const uint32_t tail_index = voxel_tree_header.emissive_leaf_count - 1;

      // if we are at the tail we can safely decrement
      if (tail_index == sampled_index) {
        voxel_tree_header.DeallocateEmissiveLeafVoxels(1);
        return light_sample;
      }

      // get the tail value
      const uint32_t tail_page_index = tail_index >> voxel_tree_header.page_size_exp;
      const uint32_t tail_page_offset = tail_index - (tail_page_index << voxel_tree_header.page_size_exp);
      const uint32_t tail_value =
          voxel_tree_emissive_leaf[tail_page_index][tail_page_offset].packed_tree_index.load();

      // try to claim the tail value so that we can swap
      const uint32_t ret =
          voxel_tree_emissive_leaf[tail_page_index][tail_page_offset].packed_tree_index.compareExchange(
              tail_value, SENTINAL);

      // if we failed to claim the tail then restore sampled_value so that we can try again
      if (tail_value == SENTINAL || ret == SENTINAL) {
        voxel_tree_emissive_leaf[sampled_page_index][sampled_page_offset].packed_tree_index.store(
            sampled_value);
        return light_sample;
      }

      // if we succeed in claiming the tail value then we can move the tail value to a safe
      // region and decrement
      if (ret == tail_value) {
        voxel_tree_emissive_leaf[sampled_page_index][sampled_page_offset].packed_tree_index.store(tail_value);
        voxel_tree_header.DeallocateEmissiveLeafVoxels(1);
      }
    }

    return light_sample;
  }

  light_sample.leaf_node_opt = node_opt.value;
  light_sample.tree_index = sampled_tree_index;

  return light_sample;
}

Reservoir<float3> ResampleLights(const float3 origin, const float3 normal, const uint32_t sample_count,
                                 inout uint32_t seed) {
  Reservoir<float3> reservoir = {};
  for (uint32_t i = 0; i < sample_count; i++) {
    const LightSample light_sample = SampleLights(seed);

    if (!light_sample.leaf_node_opt.hasValue) {
      continue;
    }

    const LeafNode node = light_sample.leaf_node_opt.value;

    const uint32_t sampled_visiblity_index = (uint32_t)(FloatHash(seed) * countbits(node.visibility_mask));
    const float3 sampled_normal =
        visibility_direction_arr[GetNthSetBitIndex(node.visibility_mask, sampled_visiblity_index)];

    seed ^= Hash(seed * node.visibility_mask);
    float3 tangent, bitangent;
    GetNormalBasis(sampled_normal, tangent, bitangent);
    const float epsilon = 1e-4f;
    const float2 sampled_surface_pos = (voxel_size * (1.0f - epsilon)) * (UniformRandom2(seed) - 0.5f);
    seed ^= Hash(seed);

    const float3 sampled_position =
        GetWorldPositionCenter(light_sample.tree_index, voxel_tree_header.max_depth - 1) +
        sampled_normal * (voxel_size * (0.5f - epsilon)) + tangent * sampled_surface_pos.x +
        bitangent * sampled_surface_pos.y;
    const float cheap_pdf = 1.0f / (voxel_tree_header.emissive_leaf_count * countbits(node.visibility_mask) *
                                    voxel_size * voxel_size);

    const float3 sampled_offset = sampled_position - origin;
    const float3 sample_dir = normalize(sampled_offset);

    const float cos_terms = saturate(dot(sample_dir, normal)) * saturate(dot(-sample_dir, sampled_normal));
    if (cos_terms == 0.0f) {
      continue;
    }

    const float geometric_factor = cos_terms / dot(sampled_offset, sampled_offset);

    // approximating f_d * G * L_e
    const float target_pdf = length((material_arr[node.material_index].albedo.rgb *
                                     material_arr[node.material_index].emissive * geometric_factor) /
                                    PI);

    reservoir.Update(sampled_position, target_pdf / cheap_pdf, FloatHash(seed));
  }

  return reservoir;
}

struct PushConstants {
  float diffuse_alpha;
  float specular_alpha;
  uint32_t frame_number;
};

uint32_t countbits(const uint32_t4 v) {
  return countbits(v.x) + countbits(v.y) + countbits(v.z) + countbits(v.w);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void CMain(const uint32_t2 dispatch_id: SV_DispatchThreadID, const uniform PushConstants info) {
  uint32_t2 image_size;
  image.GetDimensions(image_size.x, image_size.y);

  if (all(image_size <= (dispatch_id << DIFFUSE_PREPASS_SCALE_EXP))) {
    return;
  }

  const float4x4 inv_view_proj =
      mul(LookAtInverse(camera.position, camera.position + camera.front, camera.up),
          PerspectiveReverseZInverse(1.0f, 2.0f, camera.fov_y, camera.aspect_ratio));

  const float3 ray_dir = GetPixelRayDir(dispatch_id << DIFFUSE_PREPASS_SCALE_EXP, image_size, inv_view_proj);

  const float t = SampleBeamImage(dispatch_id << DIFFUSE_PREPASS_SCALE_EXP, beam_image);

  RaycastResult res;
  Raycast(camera.position + t * ray_dir, ray_dir, res);

  if (res.hit) {
    const float3 voxel_center = GetWorldPositionCenter(res.hit_tree_index, res.hit_level);

    bool initialized;
    const Optional<uint32_t> hash_set_index = HashSetFindOrInitialize(
        PackTreeIndex(res.hit_tree_index), initialized, hash_set_key_arr, hash_set_header);

    if (initialized) {
      const float last_distance = length(last_camera.position - voxel_center);

      const Optional<uint32_t> last_hash_set_index =
          HashSetFind(PackTreeIndex(res.hit_tree_index), last_hash_set_key_arr, hash_set_header);

      if (last_hash_set_index.hasValue) {
        const LightingData last_diffuse_data = last_diffuse_data_arr[last_hash_set_index.value];

        diffuse_data_arr[hash_set_index.value].acc_lighting +=
            last_diffuse_data.acc_lighting * (1.0f - info.diffuse_alpha);
        diffuse_data_arr[hash_set_index.value].sample_count +=
            last_diffuse_data.sample_count * (1.0f - info.diffuse_alpha);
      }

      const float3 ray_origin = StepTreeRaySingle(voxel_center, res.hit_normal, res.hit_level);

      uint32_t seed = Hash(res.leaf_ptr ^ ((info.frame_number + 1) * 9781u) ^
                           ((dispatch_id.x + 1) * (dispatch_id.y + 1) * 32197u));

      const Material material = material_arr[res.hit_node.material_index];

      {
        const uint32_t SAMPLE_COUNT = 32;
        const Reservoir<float3> reservoir = ResampleLights(ray_origin, res.hit_normal, SAMPLE_COUNT, seed);
 
        const float3 sample_offset = reservoir.selected_sample - ray_origin;
        const float3 sample_dir = normalize(sample_offset);
        const uint32_t3 sample_tree_index = GetTreeIndex(reservoir.selected_sample);
        const float3 sample_normal =
            GetNormal(GetWorldPositionCenter(sample_tree_index, voxel_tree_header.max_depth - 1),
                      reservoir.selected_sample);
        RaycastResult light_res;
        Raycast(ray_origin, sample_dir, light_res);

        float3 radiance = 0.0f;
        if (light_res.hit && all(sample_tree_index == light_res.hit_tree_index) &&
            all(light_res.hit_normal == sample_normal)) {
          const Material hit_material = material_arr[light_res.hit_node.material_index];
          const float geometric_factor =
              (saturate(dot(sample_dir, res.hit_normal) * dot(-sample_dir, light_res.hit_normal))) /
              (dot(sample_offset, sample_offset));

          const float3 color = (hit_material.albedo.rgb * hit_material.emissive * geometric_factor) / PI;
          const float target_pdf = length(color);

          const float W = reservoir.total_weight / (SAMPLE_COUNT * target_pdf);

          if (all(color >= 0.0f)) {
            radiance = color * W;
          }
        }

        diffuse_data_arr[hash_set_index.value].sample_count += 1.0f;
        diffuse_data_arr[hash_set_index.value].acc_lighting += radiance;
      }
    }
  }
}
