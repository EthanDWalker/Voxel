import common;
import voxelize;

[[vk::binding(1, 0)]]
RWTexture2D<float> beam_image;

[[vk::binding(0, 1)]]
ConstantBuffer<Camera> camera;

[[vk::binding(0, 2)]]
StructuredBuffer<BranchNode> voxel_tree[];

[[vk::binding(1, 2)]]
StructuredBuffer<LeafNode> voxel_tree_leaf[];

[[vk::binding(2, 2)]]
TreeHeader voxel_tree_header;

static const float box_size = voxel_tree_header.max_bound - voxel_tree_header.min_bound;
static const float grid_n = 1 << (voxel_tree_header.max_depth * 2);
static const float voxel_size = box_size / grid_n;

vector<uint32_t, N> GetTreeIndex<let N : int>(vector<float, N> p) {
  const vector<float, N> offset = vector<float, N>(p - voxel_tree_header.min_bound);
  return vector<uint32_t, N>(floor((offset / box_size) * float(grid_n)));
}

uint32_t GetLocalIndex(const uint32_t3 index, const uint32_t depth) {
  return dot(1, (((index >> ((voxel_tree_header.max_depth - 1) - depth) * 2)) & 0x3) << uint32_t3(0, 2, 4));
}

static const uint32_t initial_ptr_stack_depth = voxel_tree_header.max_depth - PTR_STACK_SIZE;

[numthreads(8, 8, 1)]
void CMain(const uint32_t2 dispatch_id: SV_DispatchThreadID) {
  uint32_t2 image_size;
  beam_image.GetDimensions(image_size.x, image_size.y);

  if (all(image_size <= dispatch_id))
    return;

  const float2 ndc = float2(float(dispatch_id.x) / float(image_size.x) * 2.0f - 1.0f,
                            float(dispatch_id.y) / float(image_size.y) * -2.0f + 1.0f);

  const float3 ndc_min = float3(ndc, 0.0f);
  const float3 ndc_max = float3(ndc, 1.0f);

  const float4 view_min =
      mul(PerspectiveReverseZInverse(camera.z_near, camera.z_near + 1.0f, camera.fov_y, camera.aspect_ratio),
          float4(ndc_min, 1.0f));
  const float4 view_max =
      mul(PerspectiveReverseZInverse(camera.z_near, camera.z_near + 1.0f, camera.fov_y, camera.aspect_ratio),
          float4(ndc_max, 1.0f));

  const float3 world_min = mul(LookAtInverse(camera.position, camera.position + camera.front, camera.up),
                               float4(view_min.xyz / view_min.w, 1.0f))
                               .xyz;
  const float3 world_max = mul(LookAtInverse(camera.position, camera.position + camera.front, camera.up),
                               float4(view_max.xyz / view_max.w, 1.0f))
                               .xyz;

  const float3 ray_dir = normalize(world_max - world_min);
  const float3 ray_origin = camera.position;

  float ray_t_min, ray_t_max;
  const bool intersect = RayAABBIntersect(ray_dir, ray_origin, voxel_tree_header.min_bound * (1.0f - 1e-5f),
                                          voxel_tree_header.max_bound * (1.0f - 1e-5f), ray_t_min, ray_t_max);
  if (!intersect) {
    return;
  }

  ray_t_min = max(ray_t_min, 0.0f);

  float3 position = ray_origin + ray_dir * ray_t_min;

  const float3 inv_ray_dir = 1.0f / ray_dir;
  const float3 abs_inv_ray_dir = abs(inv_ray_dir);
  const float3 ray_dir_step = step(0.0f, ray_dir);
  const int32_t3 voxel_step = sign(ray_dir);

  uint32_t3 current_index = GetTreeIndex(position);

  uint32_t iterations = 0;

  const uint32_t memory_index = spirv_asm {
        result:$$uint = OpLoad builtin(LocalInvocationIndex:uint);
    };

  static groupshared uint32_t ptr_stack_mem[64][PTR_STACK_SIZE];

  uint32_t3 last_index = 0xFFFFFFFF;
  bool is_empty = true;
  while (all(position >= voxel_tree_header.min_bound) && all(position <= voxel_tree_header.max_bound)) {
    const uint32_t3 index_diff = last_index ^ current_index;
    const uint32_t invalidated_stack_size = firstbithigh(index_diff.x | index_diff.y | index_diff.z) >> 1;

    uint32_t level = invalidated_stack_size < PTR_STACK_SIZE
                         ? initial_ptr_stack_depth + ((PTR_STACK_SIZE - 1) - invalidated_stack_size)
                         : 1;
    uint32_t ptr = invalidated_stack_size < PTR_STACK_SIZE
                       ? ptr_stack_mem[memory_index][level - initial_ptr_stack_depth]
                       : GetLocalIndex(current_index, 0);
    uint32_t local_index;
    BranchNode node;

    BranchNode test;

    for (;; level++) {
      const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
      const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

      node = voxel_tree[page_index + voxel_tree_header.GetLevelPageOffset(level - 1)][page_offset];

      local_index = GetLocalIndex(current_index, level);

      is_empty = ((node.child_mask >> local_index) & 0b1) == 0;

      if (level >= initial_ptr_stack_depth) {
        ptr_stack_mem[memory_index][level - initial_ptr_stack_depth] = ptr;
      }

      if (is_empty || level == voxel_tree_header.max_depth - 1)
        break;

      ptr = node.child_ptr + local_index;
    }

    if (!is_empty) {
      break;
    }

    const uint32_t level_shift = ((voxel_tree_header.max_depth - 1) - level) * 2 +
                                 uint32_t((node.child_mask >> (local_index & 0b101010) & 0x00330033) == 0);
    const float scale = (1 << level_shift) * voxel_size;

    const uint32_t3 level_index = current_index >> level_shift;
    const float3 voxel_min_bound = voxel_tree_header.min_bound + level_index * scale;
    const float3 intersect_pos = voxel_min_bound + ray_dir_step * scale;

    const float3 t_max = (intersect_pos - position) * inv_ray_dir;
    const float3 t_delta = scale * abs_inv_ray_dir;

    const float3 mask = step(t_max, min(t_max.yzx, t_max.zxy));

    const float3 neighbor_min_bound = voxel_min_bound + (voxel_step * scale * mask);
    const float3 neighbor_max_bound = neighbor_min_bound + scale;

    const float voxel_clamp_epsilon = 1e-3f;
    position = clamp(position + dot(mask, t_max) * ray_dir, neighbor_min_bound + voxel_clamp_epsilon,
                     neighbor_max_bound - voxel_clamp_epsilon);

    last_index = current_index;
    current_index = GetTreeIndex(position);

    if (iterations > 100) {
      break;
    }
    iterations++;
  }

  const float3 voxel_min_bound =
      (voxel_tree_header.max_bound - voxel_tree_header.min_bound) * (float3(current_index) / float3(grid_n)) +
      voxel_tree_header.min_bound;
  const float3 voxel_max_bound = voxel_min_bound + voxel_size;
  float voxel_t_min, voxel_t_max;
  RayAABBIntersect(ray_dir, ray_origin, voxel_min_bound, voxel_max_bound, voxel_t_min, voxel_t_max);

  beam_image.Store(dispatch_id, voxel_t_min);
}
