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

float2 Random2(const uint seed) { return float2(FloatHash(seed), FloatHash(seed * 1664525U + 1013904223U)); }

float3 TransformFromNormalSpace(const float3 normal, const float3 local_dir) {
  const float3 up = abs(normal.y) < 0.999f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);

  const float3 tangent = normalize(cross(up, normal));
  const float3 bitangent = cross(normal, tangent);

  return tangent * local_dir.x + normal * local_dir.y + bitangent * local_dir.z;
}

float3 TransformToNormalSpace(const float3 normal, const float3 world_dir) {
  const float3 up = abs(normal.y) < 0.999f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);

  const float3 tangent = normalize(cross(up, normal));
  const float3 bitangent = cross(normal, tangent);

  return float3(dot(world_dir, tangent), dot(world_dir, normal), dot(world_dir, bitangent));
}

float3 CosineWeightedHemisphere(const float3 normal, const float2 rand) {
  const float r = rand.x;
  const float theta = 2 * PI * rand.y;

  const float2 disk_point = r * float2(cos(theta), sin(theta));

  const float3 local_dir =
      float3(disk_point.x, sqrt(max(0.0f, 1.0f + dot(disk_point, -disk_point))), disk_point.y);

  return TransformFromNormalSpace(normal, local_dir);
}

// w_o away from surface
float3 VNDF_GGX(const float3 normal, float3 w_o, const float roughness, const float2 rand) {
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
  float3 ray_dir;

  if (true) {
    const float3 micro_normal = VNDF_GGX(normal, -view_dir, material.rough, Random2(seed));

    ray_dir = reflect(view_dir, micro_normal);

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
  } else {
    const float2 uniform_rand_2 = Random2(seed);
    const float phi = atan(material.rough * sqrt(uniform_rand_2.x / (1.0f - uniform_rand_2.x)));
    const float theta = 2 * PI * uniform_rand_2.y;

    const float3 micro_normal =
        TransformFromNormalSpace(normal, float3(sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta)));

    ray_dir = reflect(view_dir, micro_normal);

    if (dot(normal, ray_dir) > 0.0f && dot(ray_dir, micro_normal) > 0.0f) {
      const float3 f_0 = 0.16f * (material.reflect * material.reflect) * (1.0f - material.metallic) +
                         material.albedo.rgb * material.metallic;

      const float3 F = F_Schlick(f_0, dot(ray_dir, micro_normal));

      const float G = G_2_SmithGGX(ray_dir, -view_dir, normal, a2);

      const float weight =
          abs(dot(-view_dir, micro_normal)) / max(dot(normal, -view_dir) * dot(normal, micro_normal), 1e-4f);

      reflectance = F * G * weight;
    } else {
      return 0.0f;
    }
  }

  float3 radiance = 0.0f;
  float3 throughput = reflectance;

  RaycastResult res;
  Raycast(origin, ray_dir, res);

  if (res.hit) {
    material = material_arr[res.hit_material_index];
    radiance += throughput * material.emmisive * material.albedo.rgb;

    const Optional<uint32_t> hit_last_hash_set_index =
        HashSetFind(PackTreeIndex(res.hit_tree_index), last_hash_set_key_arr, hash_set_header);
    if (hit_last_hash_set_index.hasValue) {
      const LightingData diffuse_data = last_diffuse_data_arr[hit_last_hash_set_index.value];
      const LightingData specular_data = last_specular_data_arr[hit_last_hash_set_index.value];

      radiance += throughput * material.albedo.rgb * (1.0f - material.metallic) *
                  (diffuse_data.acc_lighting / diffuse_data.sample_count);
    }
  } else {
    radiance += throughput * SampleEnvironmentColor(ray_dir);
  }

  return radiance;
}

float RaySphereIntersection(const float3 sphere_origin, const float sphere_radius, float3 ray_origin,
                            const float3 ray_dir) {
  ray_origin = ray_origin - sphere_origin;

  const float a = 1.0f;
  const float b = 2.0f * dot(ray_origin, ray_dir);
  const float c = dot(ray_origin, ray_origin) - sphere_radius * sphere_radius;

  const float delta_2 = b * b - 4.0f * a * c;
  if (delta_2 < 0.0f) {
    return -1.0f;
  }
  const float delta = sqrt(delta_2);

  const float t_0 = (-b - delta) / (2.0f * a);
  const float t_1 = (-b + delta) / (2.0f * a);

  return min(t_0, t_1);
}

float3 SampleDiffuseLight(float3 origin, float3 normal, Material material, const uint32_t max_bounce_count,
                          inout uint32_t seed) {
  float3 throughput = 1.0f;
  float3 radiance = 0.0f;

  for (uint32_t i = 0; i < max_bounce_count; i++) {
    const float3 ray_dir = CosineWeightedHemisphere(normal, Random2(seed));

    RaycastResult res;
    Raycast(origin, ray_dir, res);

    if (!res.hit) {
      radiance += throughput * SampleEnvironmentColor(ray_dir);
      break;
    }

    material = material_arr[res.hit_material_index];
    normal = res.hit_normal;

    // lambert diffuse and cosine weighted sampling method's pdf cancel
    radiance += throughput * material.emmisive * material.albedo.rgb;
    throughput *= material.albedo.rgb * (1.0f - material.metallic);

    if (all(throughput < 1e-2f) || material.emmisive != 0.0f) {
      break;
    }

    const Optional<uint32_t> hit_hash_set_index =
        HashSetFind(PackTreeIndex(res.hit_tree_index), last_hash_set_key_arr, hash_set_header);
    if (hit_hash_set_index.hasValue) {
      const LightingData diffuse_data = last_diffuse_data_arr[hit_hash_set_index.value];
      radiance += (1.0f - material.metallic) * material.albedo.rgb *
                  (diffuse_data.acc_lighting / max(diffuse_data.sample_count, 0.0f));
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
  image_size = image_size >> INDIRECT_PREPASS_SCALE_EXP;

  if (all(image_size <= dispatch_id))
    return;

  const float4x4 inv_view_proj =
      mul(LookAtInverse(camera.position, camera.position + camera.front, camera.up),
          PerspectiveReverseZInverse(1.0f, 2.0f, camera.fov_y, camera.aspect_ratio));

  const float3 ray_dir = GetPixelRayDir(dispatch_id, image_size, inv_view_proj);

  const float t = SampleBeamImage(dispatch_id << INDIRECT_PREPASS_SCALE_EXP, beam_image);

  RaycastResult res;
  Raycast(camera.position + t * ray_dir, ray_dir, res);

  if (res.hit) {
    const Optional<uint32_t> last_hash_set_index =
        HashSetFind(PackTreeIndex(res.hit_tree_index), last_hash_set_key_arr, hash_set_header);

    const LightingData last_diffuse_data =
        last_hash_set_index.hasValue ? last_diffuse_data_arr[last_hash_set_index.value] : LightingData();
    const LightingData last_specular_data =
        last_hash_set_index.hasValue ? last_specular_data_arr[last_hash_set_index.value] : LightingData();

    bool initialized;
    const Optional<uint32_t> hash_set_index = HashSetFindOrInitialize(
        PackTreeIndex(res.hit_tree_index), initialized, hash_set_key_arr, hash_set_header);

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

    const Material material = material_arr[res.hit_material_index];

    {
      const uint32_t max_bounce_count = 1;

      float3 diffuse_light_sample =
          SampleDiffuseLight(ray_origin, res.hit_normal, material, max_bounce_count, seed);

      diffuse_data_arr[hash_set_index.value].sample_count += 1.0f;
      diffuse_data_arr[hash_set_index.value].acc_lighting_r += diffuse_light_sample.r;
      diffuse_data_arr[hash_set_index.value].acc_lighting_g += diffuse_light_sample.g;
      diffuse_data_arr[hash_set_index.value].acc_lighting_b += diffuse_light_sample.b;
    }

    {
      const float3 specular_light_sample =
          SampleSpecularLight(ray_origin, ray_dir, material, res.hit_normal, seed);
      specular_data_arr[hash_set_index.value].sample_count += 1.0f;
      specular_data_arr[hash_set_index.value].acc_lighting_r += specular_light_sample.r;
      specular_data_arr[hash_set_index.value].acc_lighting_g += specular_light_sample.g;
      specular_data_arr[hash_set_index.value].acc_lighting_b += specular_light_sample.b;
    }
  }
}
