#include "Include/common.slang"
#define VOXEL_TREE_SET_BINDING 2
#include "Include/camera.slang"
#include "Include/voxel_tree.slang"

[[vk::binding(0, 0)]]
RWTexture2D<float4> image;
[[vk::binding(1, 0)]]
RWTexture2D<float> beam_image;

[[vk::binding(0, 1)]]
ConstantBuffer<Camera> camera;

[[vk::binding(0, 3)]]
CountedBuffer<DirectionalLight> directional_light_arr;

void RadianceConecast(const float3 ray_origin, float3 ray_dir, const float half_angle,
                      out RaycastResult result) {
  result.hit = false;
  result.error = false;
  result.t = voxel_size;
  result.hit_color = 0.0f;
  result.iterations = 0;
  ray_dir += 1e-4f;
  float ray_t_min, ray_t_max;
  const bool intersect = RayAABBIntersect(ray_dir, ray_origin, voxel_tree_header.min_bound * (1.0f - 1e-5f),
                                          voxel_tree_header.max_bound * (1.0f - 1e-5f), ray_t_min, ray_t_max);
  if (!intersect) {
    return;
  }

  ray_t_min = max(ray_t_min, 0.0f);
  result.t += ray_t_min;

  float3 position = ray_origin + ray_dir * result.t;
  const float3 inv_ray_dir = 1.0f / ray_dir;
  const float3 abs_inv_ray_dir = abs(inv_ray_dir);
  const float3 ray_dir_step = step(0.0f, ray_dir);
  const int32_t3 voxel_step = sign(ray_dir);
  const float tan_half_angle = tan(half_angle);

  uint32_t3 current_index = GetTreeIndex(position);

  uint32_t3 last_index = 0xFFFFFFFF;
  bool is_empty = true;
  BranchNode node;
  uint32_t local_index;
  uint32_t ptr;
  while (all(position >= voxel_tree_header.min_bound) && all(position <= voxel_tree_header.max_bound)) {
    const uint32_t3 index_diff = last_index ^ current_index;
    const uint32_t invalidated_stack_size = firstbithigh(index_diff.x | index_diff.y | index_diff.z) >> 1;

    uint32_t level = invalidated_stack_size < PTR_STACK_SIZE
                         ? initial_ptr_stack_depth + ((PTR_STACK_SIZE - 1) - invalidated_stack_size)
                         : 1;
    ptr = invalidated_stack_size < PTR_STACK_SIZE
              ? ray_ptr_stack_mem[memory_index][level - initial_ptr_stack_depth]
              : GetLocalIndex(current_index, 0);

    if (invalidated_stack_size == 0) {
      local_index = GetLocalIndex(current_index, level);

      is_empty = ((node.child_mask >> local_index) & 0b1) == 0;
    } else {
      for (;; level++) {
        const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
        const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

        node = voxel_tree[page_index + voxel_tree_header.GetLevelPageOffset(level - 1)][page_offset];

        local_index = GetLocalIndex(current_index, level);

        is_empty = ((node.child_mask >> local_index) & 0b1) == 0;

        if (level >= initial_ptr_stack_depth) {
          ray_ptr_stack_mem[memory_index][level - initial_ptr_stack_depth] = ptr;
        }

        if (is_empty || level == (voxel_tree_header.max_depth - uint32_t(min(log2(result.t), 2))))
          break;

        ptr = node.child_ptr + local_index;
      }
    }

    last_index = current_index;

    if (!is_empty) {
      const uint32_t radiance_ptr = node.child_radiance_ptr;

      uint32_t radiance_page_index;
      uint32_t3 radiance_image_pixel_offset;
      GetRadianceImagePixelOffset(radiance_ptr, radiance_page_index, radiance_image_pixel_offset);

      const uint32_t3 local_offset = GetLocalOffset(current_index, level);

      const float4 color = radiance_pages[radiance_page_index].SampleLevel(
          radiance_sampler, (radiance_image_pixel_offset + local_offset + 1) / radiance_image_size, 0.0f);

      result.hit_color.rgb += color.xyz * (1.0f - result.hit_color.a);
      result.hit_color.a += (1.0f - result.hit_color.a) * color.a;
      if (result.hit_color.a >= 1.0f) {
        break;
      }

      const float delta_t = 2.0f * tan_half_angle * result.t;
      result.t += delta_t;

      position += ray_dir * delta_t;
    } else {
      const uint32_t level_shift = ((voxel_tree_header.max_depth - 1) - level) * 2 +
                                   uint32_t((node.child_mask >> (local_index & 0b101010) & 0x00330033) == 0);
      const float scale = (1 << level_shift) * voxel_size;

      const uint32_t3 level_index = current_index >> level_shift;
      const float3 voxel_min_bound = voxel_tree_header.min_bound + level_index * scale;
      const float3 intersect_pos = voxel_min_bound + ray_dir_step * scale;

      const float3 t_max = (intersect_pos - position) * inv_ray_dir;

      const float3 mask = step(t_max, min(t_max.yzx, t_max.zxy));

      const float3 neighbor_min_bound = voxel_min_bound + (voxel_step * scale * mask);
      const float3 neighbor_max_bound = neighbor_min_bound + scale;

      const float delta_t = dot(mask, t_max);
      result.t += delta_t;

      const float voxel_clamp_epsilon = 1e-3f;
      position = clamp(position + delta_t * ray_dir, neighbor_min_bound + voxel_clamp_epsilon,
                       neighbor_max_bound - voxel_clamp_epsilon);
    }

    current_index = GetTreeIndex(position);
  }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void CMain(const uint32_t2 dispatch_id: SV_DispatchThreadID) {
  uint32_t2 image_size;
  image.GetDimensions(image_size.x, image_size.y);

  if (all(image_size <= dispatch_id))
    return;

  const float2 beam_image_location_float = dispatch_id / float(BEAM_PREPASS_SCALE);
  const int32_t2 beam_image_location = dispatch_id / BEAM_PREPASS_SCALE;
  const bool2 is_whole = (beam_image_location_float == beam_image_location);

  float t;
  if (all(is_whole)) {
    t = beam_image.Load(beam_image_location);
    t = min(t, beam_image.Load(beam_image_location + int32_t2(0, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 0)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 1)));
  } else if (is_whole.x) {
    t = beam_image.Load(beam_image_location);
    t = min(t, beam_image.Load(beam_image_location + int32_t2(0, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 0)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(-1, 0)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(-1, 1)));
  } else if (is_whole.y) {
    t = beam_image.Load(beam_image_location);
    t = min(t, beam_image.Load(beam_image_location + int32_t2(0, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(0, -1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 0)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, -1)));
  } else {
    t = beam_image.Load(beam_image_location);
    t = min(t, beam_image.Load(beam_image_location + int32_t2(0, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 1)));
    t = min(t, beam_image.Load(beam_image_location + int32_t2(1, 0)));
  }

  const float4x4 inv_view_proj =
      mul(LookAtInverse(camera.position, camera.position + camera.front, camera.up),
          PerspectiveReverseZInverse(camera.z_near, camera.z_near + 1.0f, camera.fov_y, camera.aspect_ratio));

  const float2 ndc = float2(float(dispatch_id.x + 0.5f) / float(image_size.x) * 2.0f - 1.0f,
                            float(dispatch_id.y + 0.5f) / float(image_size.y) * -2.0f + 1.0f);

  const float3 ndc_min = float3(ndc, 0.0f);
  const float3 ndc_max = float3(ndc, 1.0f);

  const float4 world_min = mul(inv_view_proj, float4(ndc_min, 1.0f));
  const float4 world_max = mul(inv_view_proj, float4(ndc_max, 1.0f));

  const float3 ray_dir = normalize((world_max.xyz / world_max.w) - (world_min.xyz / world_min.w));
  const float3 ray_origin = camera.position;

  RaycastResult res;
  Raycast(ray_origin + t * ray_dir, ray_dir, res);

  if (res.hit) {
    const float3 cone_dirs[5] = {
      float3(0.0f, 0.0f, 1.0f),
      float3(0.0f, 0.707106781f, 0.707106781f),
      float3(0.0f, -0.707106781f, 0.707106781f),
      float3(0.707106781f, 0.0f, 0.707106781f),
      float3(-0.707106781f, 0.0f, 0.707106781f),
    };
    const float cone_weights[5] = { 0.28, 0.18, 0.18, 0.18, 0.18 };

    const float3 tangent = normalize(cross(res.hit_normal, normalize(float3(0.5f, 0.5f, 0.5f))));
    const float3 bit_tangent = normalize(cross(tangent, res.hit_normal));
    const float3x3 tbn = float3x3(float3(tangent.x, bit_tangent.x, res.hit_normal.x),
                                  float3(tangent.y, bit_tangent.y, res.hit_normal.y),
                                  float3(tangent.z, bit_tangent.z, res.hit_normal.z));

    const float3 L = -directional_light_arr[0].direction;
    const float light_impact = max(dot(L, res.hit_normal), 0.3f);

    float3 acc = 0.0f;

    for (uint32_t i = 0; i < 4; i++) {
      const float3 cone_dir = normalize(mul(tbn, cone_dirs[i]));

      RaycastResult cone_res;
      RadianceConecast(res.hit_position, cone_dir, radians(30.0f), cone_res);

      acc += cone_weights[i] * cone_res.hit_color.rgb;
    }
    image.Store(dispatch_id, float4(acc + (res.hit_color.rgb * light_impact), 1.0f));

  } else {
    image.Store(dispatch_id, float4(float3(0.0f), 1.0f));
  }
}
