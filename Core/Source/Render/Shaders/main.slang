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

void RaycastTest(const float3 ray_origin, float3 ray_dir, out RaycastResult result,
                 const float t_max = 1e10f) {
  result.hit = false;
  result.t = 0.0f;
  result.iterations = 0;
  ray_dir += 1e-4f;
  float ray_t_min, ray_t_max;
  const bool intersect =
      RayAABBIntersect(ray_dir, ray_origin, voxel_tree_header.min_bound * (1.0f - 1e-5f),
                       (voxel_tree_header.min_bound + box_size) * (1.0f - 1e-5f), ray_t_min, ray_t_max);
  if (!intersect) {
    return;
  }

  ray_t_min = max(ray_t_min, 0.0f);
  result.t += ray_t_min;

  float3 position = ray_origin + ray_dir * ray_t_min;

  const float3 inv_ray_dir = 1.0f / ray_dir;
  const float3 ray_dir_step = step(0.0f, ray_dir);
  const int32_t3 voxel_step = sign(ray_dir);

  uint32_t3 current_index = GetTreeIndex(position);

  uint32_t3 last_index = 0xFFFFFFFF;
  bool is_empty = true;
  BranchNode node;
  uint32_t local_index;
  uint32_t ptr;
  uint32_t level;

  while (all(position >= voxel_tree_header.min_bound) &&
         all(position <= (voxel_tree_header.min_bound + box_size))) {
    const uint32_t3 index_diff = last_index ^ current_index;
    const uint32_t invalidated_stack_size = firstbithigh(index_diff.x | index_diff.y | index_diff.z) >> 1;

    if (invalidated_stack_size < PTR_STACK_SIZE) {
      level = initial_ptr_stack_depth + ((PTR_STACK_SIZE - 1) - invalidated_stack_size);
      ptr = ptr_stack_mem[memory_index][level - initial_ptr_stack_depth];
    } else {
      level = 1;
      ptr = GetLocalIndex(current_index, 0);
    }

    TraverseTreeFromLevelPtr(ptr, level, node, is_empty, local_index, current_index);

    if ((!is_empty && node.luminance != 0.0f) || result.t > t_max) {
      break;
    }

    position = StepTreeRay(ray_dir_step, ray_dir, inv_ray_dir, voxel_step, position, current_index,
                           node.child_mask, local_index, level, result.t);

    last_index = current_index;
    current_index = GetTreeIndex(position);

    result.iterations++;
  }

  if (is_empty) {
    result.hit = false;
    return;
  }
  result.hit = true;

  result.hit_position = position;
  result.hit_tree_index = current_index;
  result.hit_level = level;

  ptr = node.child_ptr_data.child_ptr + local_index;
  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

  const LeafNode leaf_node = voxel_tree_leaf[page_index][page_offset];
  result.hit_node = leaf_node;
  result.leaf_ptr = ptr;

  const float3 voxel_center = GetWorldPositionCenter(result.hit_tree_index, result.hit_level);
  const float3 hit_offset = (result.hit_position - voxel_center);
  const float3 abs_hit_offset = abs(result.hit_position - voxel_center);
  result.hit_normal =
      sign(hit_offset) *
      (abs_hit_offset.x > abs_hit_offset.y && abs_hit_offset.x > abs_hit_offset.z
           ? float3(1.0f, 0.0f, 0.0f)
           : (abs_hit_offset.y > abs_hit_offset.z ? float3(0.0f, 1.0f, 0.0f) : float3(0.0f, 0.0f, 1.0f)));
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
        const uint32_t2 indirect_pixel_index = dispatch_id & (~0b1);
        const float3 indirect_light_dir = GetPixelRayDir(indirect_pixel_index, image_size, inv_view_proj);
        const float indirect_t = SampleBeamImage(indirect_pixel_index, beam_image);

        RaycastResult lighting_res;
        Raycast(ray_origin + indirect_t * indirect_light_dir, indirect_light_dir, lighting_res);

        const Optional<uint32_t> closest_hash_set_index =
            HashSetFind(PackTreeIndex(lighting_res.hit_tree_index), hash_set_key_arr, hash_set_header);

        if (closest_hash_set_index.hasValue) {
          diffuse_data = diffuse_data_arr[closest_hash_set_index.value];
          specular_data = diffuse_data_arr[closest_hash_set_index.value];
        }
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
