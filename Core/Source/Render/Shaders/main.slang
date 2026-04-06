import common;
import voxelize;

[[vk::binding(0, 0)]]
RWTexture2D<float4> image;

[[vk::binding(0, 1)]]
ConstantBuffer<Camera> camera;

[[vk::binding(0, 2)]]
StructuredBuffer<BranchNode> voxel_tree[];

[[vk::binding(1, 2)]]
TreeHeader voxel_tree_header;

[[vk::binding(0, 3)]]
CountedBuffer<DirectionalLight> directional_light_arr;

bool RayAABBIntersect(const float3 ray_dir, const float3 ray_origin, const float3 bounds_min,
                      const float3 bounds_max, inout float t_min, inout float t_max) {
  float t_min_y, t_max_y, t_min_z, t_max_z;
  const float x_inv_dir = 1 / ray_dir.x;
  if (x_inv_dir >= 0) {
    t_min = (bounds_min.x - ray_origin.x) * x_inv_dir;
    t_max = (bounds_max.x - ray_origin.x) * x_inv_dir;
  } else {
    t_min = (bounds_max.x - ray_origin.x) * x_inv_dir;
    t_max = (bounds_min.x - ray_origin.x) * x_inv_dir;
  }

  const float y_inv_dir = 1 / ray_dir.y;
  if (y_inv_dir >= 0) {
    t_min_y = (bounds_min.y - ray_origin.y) * y_inv_dir;
    t_max_y = (bounds_max.y - ray_origin.y) * y_inv_dir;
  } else {
    t_min_y = (bounds_max.y - ray_origin.y) * y_inv_dir;
    t_max_y = (bounds_min.y - ray_origin.y) * y_inv_dir;
  }

  if (t_min > t_max_y || t_min_y > t_max)
    return false;
  if (t_min_y > t_min)
    t_min = t_min_y;
  if (t_max_y < t_max)
    t_max = t_max_y;

  const float z_inv_dir = 1 / ray_dir.z;
  if (z_inv_dir >= 0) {
    t_min_z = (bounds_min.z - ray_origin.z) * z_inv_dir;
    t_max_z = (bounds_max.z - ray_origin.z) * z_inv_dir;
  } else {
    t_min_z = (bounds_max.z - ray_origin.z) * z_inv_dir;
    t_max_z = (bounds_min.z - ray_origin.z) * z_inv_dir;
  }

  if (t_min > t_max_z || t_min_z > t_max)
    return false;
  if (t_min_z > t_min)
    t_min = t_min_z;
  if (t_max_z < t_max)
    t_max = t_max_z;
  return t_max > 0.0f;
}

struct RaycastResult {
  float4 hit_color;
  float3 hit_position;
  bool hit;
  bool error;
  float3 hit_normal;
};

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

const static uint32_t ptr_stack_size = 1;
const static uint32_t initial_stack_depth = voxel_tree_header.max_depth - ptr_stack_size;

void BuildPtrStackFromIndex(inout uint32_t ptr_stack[ptr_stack_size], const uint32_t3 index,
                            const uint32_t initial_depth, inout bool is_empty, inout uint32_t empty_depth) {
  const uint32_t start_depth = initial_depth < initial_stack_depth ? 1 : initial_depth;
  uint32_t ptr = initial_depth < initial_stack_depth ? GetLocalIndex(index, 0)
                                                     : ptr_stack[start_depth - initial_stack_depth];

  for (empty_depth = start_depth;;) {
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    const BranchNode node =
        voxel_tree[page_index + voxel_tree_header.GetLevelPageOffset(empty_depth - 1)][page_offset];

    const uint32_t local_index = GetLocalIndex(index, empty_depth);

    is_empty = ((node.child_mask >> local_index) & 0b1) == 0;

    if (empty_depth >= initial_stack_depth) {
      ptr_stack[empty_depth - initial_stack_depth] = ptr;
    }

    if (is_empty || empty_depth >= voxel_tree_header.max_depth - 1)
      break;

    ptr = node.child_ptr + local_index;

    empty_depth++;
  }
}

void Raycast(const float3 ray_origin, const float3 ray_dir, out RaycastResult result) {
  result.hit = false;
  result.error = false;
  float ray_t_min, ray_t_max;
  const bool intersect = RayAABBIntersect(ray_dir, ray_origin, voxel_tree_header.min_bound * (1.0f - 1e-5f),
                                          voxel_tree_header.max_bound * (1.0f - 1e-5f), ray_t_min, ray_t_max);
  if (!intersect) {
    return;
  }

  ray_t_min = max(ray_t_min, 0.0f);

  float3 position = ray_origin + ray_dir * ray_t_min;

  const float3 inv_ray_dir = 1.0f / ray_dir;

  const int32_t3 voxel_step = select(abs(ray_dir) < 1e-5f, 0, select(ray_dir < 0.0f, -1, 1));

  uint32_t3 current_index = GetTreeIndex(position);
  uint32_t3 last_index = 0xFFFFFFFF;

  float3 t_voxel = select(ray_dir < 0.0f, current_index / grid_n, (current_index + 1) / grid_n);
  float3 voxel_max = voxel_tree_header.min_bound + t_voxel * box_size;
  float3 t_max = ray_t_min + (voxel_max - position) * inv_ray_dir;

  const float3 t_delta = voxel_size * abs(inv_ray_dir);

  bool is_empty = true;

  uint32_t iterations = 0;

  const uint32_t memory_index = spirv_asm {
        result:$$uint = OpLoad builtin(LocalInvocationIndex:uint);
    };

  static groupshared uint32_t ptr_stack_mem[64][ptr_stack_size];

  while (all(position >= voxel_tree_header.min_bound) && all(position <= voxel_tree_header.max_bound)) {

    const uint32_t3 index_diff = current_index ^ last_index;
    const uint32_t invalid_stack_size = firstbithigh(index_diff.x | index_diff.y | index_diff.z);

    uint32_t level;
    if (invalid_stack_size >= ptr_stack_size || true) {
      BuildPtrStackFromIndex(ptr_stack_mem[memory_index], current_index, 1, is_empty, level);
    } else {
      BuildPtrStackFromIndex(ptr_stack_mem[memory_index], current_index,
                             initial_stack_depth + ((ptr_stack_size - 1) - invalid_stack_size), is_empty,
                             level);
    }

    if (!is_empty) {
      result.hit_color = UnpackRGBA8(0xFFFFFFFF);
      break;
    }

    const uint32_t scale = (1 << (((voxel_tree_header.max_depth - 1) - level) * 2));
    const float3 voxel_min_bound =
        voxel_tree_header.min_bound +
        (current_index >> (((voxel_tree_header.max_depth - 1) - level) * 2)) * voxel_size * scale;

    const float3 intersect_pos = voxel_min_bound + step(0.0f, ray_dir) * voxel_size * scale;
    const float3 intersect_t = (intersect_pos - position) * inv_ray_dir;
    const float t_step = min3(intersect_t.x, intersect_t.y, intersect_t.z);

    const float3 neighbor_min_bound =
        select(t_step == intersect_t, voxel_min_bound + voxel_step * voxel_size * scale, voxel_min_bound);
    const float3 neighbor_max_bound = neighbor_min_bound + voxel_size * scale;

    const float voxel_clamp_epsilon = 1e-3f;
    position = clamp(position + t_step * ray_dir, neighbor_min_bound + voxel_clamp_epsilon,
                     neighbor_max_bound - voxel_clamp_epsilon);

    last_index = current_index;
    current_index = GetTreeIndex(position);

    if (iterations > 1000) {
      result.error = true;
      return;
    }
    iterations++;
  }

  const float3 voxel_min_bound =
      (voxel_tree_header.max_bound - voxel_tree_header.min_bound) * (float3(current_index) / float3(grid_n)) +
      voxel_tree_header.min_bound;
  const float3 voxel_max_bound = voxel_min_bound + voxel_size;
  float voxel_t_min, voxel_t_max;
  RayAABBIntersect(ray_dir, ray_origin, voxel_min_bound, voxel_max_bound, voxel_t_min, voxel_t_max);

  if (is_empty) {
    result.hit = false;
    return;
  }
  result.hit = true;

  result.hit_position = (ray_origin + ray_dir * voxel_t_min);

  const float3 rel_hit_pos = result.hit_position - (voxel_min_bound + voxel_size / 2.0f);
  const float3 abs_hit_pos = abs(rel_hit_pos);

  result.hit_normal = abs_hit_pos.x > abs_hit_pos.y && abs_hit_pos.x > abs_hit_pos.z
                          ? float3(sign(rel_hit_pos.x), 0.0f, 0.0f)
                          : (abs_hit_pos.y > abs_hit_pos.z ? float3(0.0f, sign(rel_hit_pos.y), 0.0f)
                                                           : float3(0.0f, 0.0f, sign(rel_hit_pos.z)));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void CMain(const uint32_t2 dispatch_id: SV_DispatchThreadID) {
  uint32_t2 image_size;
  image.GetDimensions(image_size.x, image_size.y);

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

  RaycastResult res;
  Raycast(ray_origin, ray_dir, res);

  if (res.error) {
    image.Store(dispatch_id, float4(float3(1.0f, 0.0f, 0.0f), 1.0f));
    return;
  }

  if (res.hit) {
    const float3 L = -directional_light_arr[0].direction + 1e-5f;
    const float light_impact = max(dot(L, res.hit_normal), 0.3f);

    image.Store(dispatch_id, float4(res.hit_color.rgb * light_impact, 1.0f));
  } else {
    image.Store(dispatch_id, float4(float3(0.0f), 1.0f));
  }
}
