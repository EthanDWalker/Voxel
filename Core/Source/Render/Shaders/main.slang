#include "Include/common.slang"
#define VOXEL_TREE_READ_WRITE
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

void ConecastAA(const float3 cone_origin, float3 cone_dir, const float tan_half_angle, const float initial_t,
                inout RaycastResult result) {
  result.hit = false;
  result.error = false;
  result.hit_color = 0.0f;
  result.t = initial_t;
  result.iterations = 0;
  cone_dir += 1e-4f;
  float ray_t_min, ray_t_max;
  const bool intersect =
      RayAABBIntersect(cone_dir, cone_origin, voxel_tree_header.min_bound * (1.0f - 1e-5f),
                       (voxel_tree_header.min_bound + box_size) * (1.0f - 1e-5f), ray_t_min, ray_t_max);
  if (!intersect) {
    return;
  }

  ray_t_min = max(ray_t_min, 0.0f);
  result.t += ray_t_min;

  float3 position = cone_origin + cone_dir * ray_t_min;

  const float3 inv_cone_dir = 1.0f / cone_dir;
  const float3 abs_inv_cone_dir = abs(inv_cone_dir);
  const float3 cone_dir_step = step(0.0f, cone_dir);
  const int32_t3 voxel_step = sign(cone_dir);

  uint32_t3 current_index = GetTreeIndex(position);

  uint32_t3 last_index = 0xFFFFFFFF;
  bool is_empty = true;
  BranchNode node;
  uint32_t local_index;
  uint32_t ptr;
  uint32_t level;

  const uint32_t sub_pixel_offset_count = 8;

  while (all(position >= voxel_tree_header.min_bound) &&
         all(position <= (voxel_tree_header.min_bound + box_size))) {
    const uint32_t3 index_diff = last_index ^ current_index;
    const uint32_t invalidated_stack_size = firstbithigh(index_diff.x | index_diff.y | index_diff.z) >> 1;

    level = invalidated_stack_size < PTR_STACK_SIZE
                ? initial_ptr_stack_depth + ((PTR_STACK_SIZE - 1) - invalidated_stack_size)
                : 1;
    ptr = invalidated_stack_size < PTR_STACK_SIZE
              ? ray_ptr_stack_mem[memory_index][level - initial_ptr_stack_depth]
              : GetLocalIndex(current_index, 0);

    if (invalidated_stack_size == 0) {
      local_index = GetLocalIndex(current_index, level);

      is_empty = ((node.child_mask >> local_index) & 0b1) == 0;
    } else {
      TraverseTreeFromLevelPtr(ptr, level, node, is_empty, local_index, current_index,
                               voxel_tree_header.max_depth);
    }

    if (!is_empty) {
      break;
    }

    position = StepTreeRay(cone_dir_step, cone_dir, inv_cone_dir, voxel_step, position, current_index,
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

  ptr = node.child_ptr + local_index;
  const uint32_t leaf_page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t leaf_page_offset = ptr - (leaf_page_index << voxel_tree_header.page_size_exp);
  result.hit_color = UnpackRGBA8(voxel_tree_leaf[leaf_page_index][leaf_page_offset].color);
  result.hit_normal = UnpackNormal(voxel_tree_leaf[leaf_page_index][leaf_page_offset].normal);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void CMain(const uint32_t3 dispatch_id: SV_DispatchThreadID) {
  uint32_t2 image_size;
  image.GetDimensions(image_size.x, image_size.y);

  if (all(image_size <= dispatch_id.xy))
    return;

  const float2 beam_image_location_float = dispatch_id.xy / float(BEAM_PREPASS_SCALE);
  const int32_t2 beam_image_location = dispatch_id.xy / BEAM_PREPASS_SCALE;
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
  t -= 1.0f;

  const float4x4 inv_view_proj =
      mul(LookAtInverse(camera.position, camera.position + camera.front, camera.up),
          PerspectiveReverseZInverse(1.0f, 2.0f, camera.fov_y, camera.aspect_ratio));

  const float3 cone_origin = camera.position;

  const float2 center_ndc = float2(float(dispatch_id.x + 0.5) / image_size.x * 2.0 - 1.0,
                                   float(dispatch_id.y + 0.5) / image_size.y * -2.0 + 1.0);
  const float4 center_world = mul(inv_view_proj, float4(center_ndc, 1.0, 1.0));
  const float3 cone_dir = normalize(center_world.xyz / center_world.w - cone_origin);

  const float2 pixel_corner_offsets[4] = {
    float2(0.0f, 0.0f),
    float2(1.0f, 0.0f),
    float2(0.0f, 1.0f),
    float2(1.0f, 1.0f),
  };

  float max_r = 0.0f;
  for (uint32_t i = 0; i < 4; i++) {
    const float2 corner_ndc =
        float2(float(dispatch_id.x + pixel_corner_offsets[i].x) / float(image_size.x) * 2.0f - 1.0f,
               float(dispatch_id.y + pixel_corner_offsets[i].y) / float(image_size.y) * -2.0f + 1.0f);

    const float4 world_pixel_corner = mul(inv_view_proj, float4(corner_ndc, 1.0f, 1.0f));
    max_r =
        max(max_r, length(world_pixel_corner.xyz / world_pixel_corner.w - center_world.xyz / center_world.w));
  }

  RaycastResult res;
  ConecastAA(cone_origin + t * cone_dir, cone_dir, max_r, t, res);

  if (res.hit) {
    const float3 L = -directional_light_arr[0].direction;
    const float light_impact = max(dot(L, res.hit_normal), 0.3f);

    const float3 acc = res.hit_color.rgb * light_impact;

    image.Store(dispatch_id.xy, float4(float3(acc), 1.0f));
  }
}
