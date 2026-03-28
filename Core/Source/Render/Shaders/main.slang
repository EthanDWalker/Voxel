import common;

[[vk::binding(0, 0)]]
RWTexture2D<float4> image;

[[vk::binding(0, 1)]]
ConstantBuffer<Camera> camera;

[[vk::binding(0, 2)]]
StructuredBuffer<uint32_t> voxel_tree[];

public struct TreeHeader {
  private ByteAddressBuffer buffer;

  public property uint32_t voxel_count {
    get { return buffer.Load<uint32_t>(0); }
  }

  public property uint32_t allocated_page_count {
    get { return buffer.Load<uint32_t>(4); }
  }

  public property uint32_t page_size {
    get { return buffer.Load<uint32_t>(8); }
  };

  public property float min_bound {
    get { return buffer.Load<float>(12); }
  };

  public property float max_bound {
    get { return buffer.Load<float>(16); }
  };

  public property uint32_t max_depth {
    get { return buffer.Load<uint32_t>(20); }
  };
};

[[vk::binding(1, 2)]]
TreeHeader voxel_tree_header;

float4x4 LookAtInverse(const float3 eye, const float3 center, const float3 up) {
  const float3 f = normalize(center - eye);
  const float3 s = normalize(cross(f, up));
  const float3 u = normalize(cross(s, f));

  // clang-format off
  const float4x4 inv_view = float4x4(
    float4(s.x, u.x, -f.x, eye.x),
    float4(s.y, u.y, -f.y, eye.y),
    float4(s.z, u.z, -f.z, eye.z),
    float4(0.0f, 0.0f, 0.0f, 1.0f)
  );
  // clang-format on

  return inv_view;
}

float4x4 PerspectiveReverseZInverse(const float z_near, const float z_far, const float fov_y,
                                    const float aspect_ratio) {
  const float tan_half_fov_y = tan(fov_y / 2.0f);

  float4x4 inv_proj = float4x4(0.0f);
  inv_proj[0][0] = aspect_ratio * tan_half_fov_y;
  inv_proj[1][1] = tan_half_fov_y;
  inv_proj[2][3] = -1.0f;
  inv_proj[3][2] = (-z_far + z_near) / (z_far * z_near);
  inv_proj[3][3] = 1.0f / z_near;
  return inv_proj;
}

bool RaySphereIntersection(const float3 ray_dir, const float3 ray_origin) {
  const float sphere_radius = 1.0f;
  const float3 sphere_center = float3(0.0f);

  const float3 L = sphere_center - ray_origin;
  const float tca = dot(L, ray_dir);
  const float d2 = dot(L, L) - tca * tca;
  if (d2 > sphere_radius * sphere_radius)
    return false;

  const float thc = sqrt(sphere_radius * sphere_radius - d2);
  const float t0 = tca - thc;
  const float t1 = tca + thc;

  if (t0 < 0.0f && t1 < 0.0f)
    return false;

  return true;
}

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
static const float grid_n = 1 << voxel_tree_header.max_depth;
static const float voxel_size = box_size / grid_n;

vector<uint32_t, N> GetTreeIndex<let N : int>(vector<float, N> p) {
  vector<uint32_t, N> tree_index =
      vector<uint32_t, N>(floor(((p - voxel_tree_header.min_bound) / box_size) * grid_n)) + 1;
  tree_index = select(tree_index == grid_n + 1, tree_index - 1, tree_index);
  return tree_index - 1;
}

const static uint32_t ptr_stack_size = 8;
const static uint32_t MAX_DEPTH = voxel_tree_header.max_depth;
const static uint32_t initial_stack_depth = (MAX_DEPTH - 1) - (ptr_stack_size - 1);

void BuildPtrStackFromIndex(inout uint32_t ptr_stack[ptr_stack_size], const uint32_t3 index,
                            const uint32_t initial_depth, inout bool is_empty, inout uint32_t empty_depth) {
  uint32_t ptr;
  uint32_t start_depth;
  if (initial_depth <= initial_stack_depth) {
    const uint32_t3 initial_index = index >> (MAX_DEPTH - 1);

    ptr = (initial_index.x & 1) + ((initial_index.y & 1) << 1) + ((initial_index.z & 1) << 2);
    start_depth = 1;
  } else {
    start_depth = initial_depth;
    ptr = ptr_stack[start_depth - initial_stack_depth];
  }

  for (empty_depth = start_depth; empty_depth < MAX_DEPTH;) {
    const uint32_t3 level_index = index >> ((MAX_DEPTH - 1) - empty_depth);
    const uint32_t local_index =
        (level_index.x & 1) + ((level_index.y & 1) << 1) + ((level_index.z & 1) << 2);

    const uint32_t page_index = ptr / voxel_tree_header.page_size;
    const uint32_t node = voxel_tree[page_index][ptr - page_index * voxel_tree_header.page_size];

    if (empty_depth >= initial_stack_depth) {
      ptr_stack[empty_depth - initial_stack_depth] = ptr;
    }

    ptr += (node & 0xFFFFFF) + local_index;

    is_empty = !bool((node >> 24) & (0b1 << local_index));

    empty_depth++;

    if (is_empty)
      break;
  }
}

void Raycast(float3 ray_origin, const float3 ray_dir, out RaycastResult result) {
  result.hit = false;
  result.error = false;
  float ray_t_min, ray_t_max;
  const bool intersect = RayAABBIntersect(ray_dir, ray_origin, voxel_tree_header.min_bound * (1.0f - 1e-5f),
                                          voxel_tree_header.max_bound * (1.0f - 1e-5f), ray_t_min, ray_t_max);
  if (!intersect) {
    return;
  }

  ray_t_min = max(ray_t_min, 0.0f);

  ray_origin = ray_origin + ray_dir * ray_t_min;

  const float3 inv_ray_dir = 1.0f / ray_dir;

  const int32_t3 voxel_step = select(abs(ray_dir) < 1e-5f, 0, select(ray_dir < 0.0f, -1, 1));

  uint32_t3 current_index = GetTreeIndex(ray_origin);
  uint32_t3 last_index = 0xFFFFFFFF;

  float3 t_voxel = select(ray_dir < 0.0f, current_index / grid_n, (current_index + 1) / grid_n);
  float3 voxel_max = voxel_tree_header.min_bound + t_voxel * box_size;
  float3 t_max = ray_t_min + (voxel_max - ray_origin) * inv_ray_dir;

  const float3 t_delta = voxel_size * abs(inv_ray_dir);

  if (false) {
    const float3 mask = step(t_max, min(t_max.yzx, t_max.zxy));

    current_index += int3(mask) * voxel_step;
    t_max += mask * t_delta;
  }

  bool is_empty = true;

  uint32_t iterations = 0;

  float3 position = ray_origin;

  const uint32_t memory_index = spirv_asm {
        result:$$uint = OpLoad builtin(LocalInvocationIndex:uint);
    };

  static groupshared uint32_t ptr_stack_mem[64][ptr_stack_size];

  while (all(position >= voxel_tree_header.min_bound) && all(position <= voxel_tree_header.max_bound)) {

    const uint32_t3 index_diff = current_index ^ last_index;
    const uint32_t invalid_stack_size = firstbithigh(index_diff.x | index_diff.y | index_diff.z);

    uint32_t level;
    if (invalid_stack_size >= ptr_stack_size) {
      BuildPtrStackFromIndex(ptr_stack_mem[memory_index], current_index, 1, is_empty, level);
    } else {
      BuildPtrStackFromIndex(ptr_stack_mem[memory_index], current_index,
                             initial_stack_depth + ((ptr_stack_size - 1) - invalid_stack_size), is_empty,
                             level);
    }

    if (!is_empty) {
      const uint32_t ptr = ptr_stack_mem[memory_index][ptr_stack_size - 1];

      const uint32_t local_index =
          (current_index.x & 1) + ((current_index.y & 1) << 1) + ((current_index.z & 1) << 2);
      const uint32_t page_index = ptr / voxel_tree_header.page_size;
      const uint32_t page_offset = ptr - page_index * voxel_tree_header.page_size;

      const uint32_t leaf_ptr = ptr + local_index + (voxel_tree[page_index][page_offset] & 0xFFFFFF);

      const uint32_t leaf_page_index = leaf_ptr / voxel_tree_header.page_size;
      const uint32_t leaf_node =
          voxel_tree[leaf_page_index][leaf_ptr - leaf_page_index * voxel_tree_header.page_size];

      result.hit_color = UnpackRGBA8(leaf_node);
      break;
    }

    const uint32_t scale = (1 << (MAX_DEPTH - level));
    const float3 voxel_min_bound =
        voxel_tree_header.min_bound + (current_index >> (MAX_DEPTH - level)) * voxel_size * scale;

    const float3 intersect_pos = voxel_min_bound + step(0.0f, ray_dir) * voxel_size * scale;
    const float3 intersect_t = (intersect_pos - position) * inv_ray_dir;
    const float t_step = min3(intersect_t.x, intersect_t.y, intersect_t.z);

    const float3 neighbor_min_bound =
        select(t_step == intersect_t, voxel_min_bound + voxel_step * voxel_size * scale, voxel_min_bound);
    const float3 neighbor_max_bound = neighbor_min_bound + voxel_size * scale;

    const float voxel_clamp_epsilon = 1e-4f * voxel_size;
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
                               float4(view_min.xyz / view_min.w + float3(0.5f), 1.0f))
                               .xyz;
  const float3 world_max = mul(LookAtInverse(camera.position, camera.position + camera.front, camera.up),
                               float4(view_max.xyz / view_max.w + float3(0.5f), 1.0f))
                               .xyz;

  const float3 ray_dir = normalize(world_max - world_min);
  const float3 ray_origin = camera.position;

  RaycastResult res;
  Raycast(ray_origin, ray_dir, res);

  if (res.error) {
    image.Store(dispatch_id, float4(float3(1.0f, 0.0f, 0.0f), 1.0f));
    return;
  }

  const float3 dir_light = normalize(float3(0.6f, -1.0f, 0.2f));
  if (res.hit) {
    const float3 L = -dir_light;
    const float light_impact = max(dot(L, res.hit_normal), 0.1f);

    RaycastResult shadow_res;
    Raycast(res.hit_position + (1e-2f * L), L, shadow_res);
    const float shadow = shadow_res.hit ? 0.1f : 1.0f;

    if (shadow_res.error) {
      image.Store(dispatch_id, float4(float3(1.0f, 0.0f, 0.0f), 1.0f));
      return;
    }

    image.Store(dispatch_id, float4(res.hit_color.rgb * light_impact * shadow, 1.0f));
  } else {
    image.Store(dispatch_id, float4(float3(0.0f), 1.0f));
  }
}
