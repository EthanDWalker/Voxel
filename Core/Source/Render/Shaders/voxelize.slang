import common;

public struct TreeHeader {
  private RWByteAddressBuffer buffer;

  public property uint32_t voxel_count {
    get {
      uint32_t val;
      buffer.InterlockedAdd(0, 0, val);
      return val;
    }
  }

  const uint32_t AllocatedVoxels(const uint32_t count) {
    uint32_t val;
    buffer.InterlockedAdd(0, count, val);
    return val;
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

struct Vertex {
  float3 position;
  uint32_t uv_data;

  property float2 uv {
    get { return float2(f16tof32(uv_data >> 16), f16tof32(uv_data & 0xFFFF)); }
  }
}

static const uint32_t VOXEL_PTR_OFFSET = 24;

[[vk::binding(0, 0)]]
StructuredBuffer<uint32_t> index_arr;
[[vk::binding(1, 0)]]
StructuredBuffer<Vertex> vertex_arr;

[[vk::binding(0, 1)]]
RWStructuredBuffer<Atomic<uint32_t>> voxel_tree[];

[[vk::binding(1, 1)]]
TreeHeader voxel_tree_header;

struct PushConstants {
  uint32_t triangle_count;
};

static const float box_size = voxel_tree_header.max_bound - voxel_tree_header.min_bound;
static const float grid_n = 1 << voxel_tree_header.max_depth;
static const float voxel_size = box_size / grid_n;

float3 GetBarycentric<let N : int>(const vector<float, N> positions[3], const vector<float, N> point) {
  const vector<float, N> v0 = positions[2] - positions[0];
  const vector<float, N> v1 = positions[1] - positions[0];
  const vector<float, N> v2 = point - positions[0];
  const float d00 = dot(v0, v0);
  const float d01 = dot(v0, v1);
  const float d11 = dot(v1, v1);
  const float d20 = dot(v2, v0);
  const float d21 = dot(v2, v1);
  const float denom = d00 * d11 - d01 * d01;

  float3 coords;
  coords.x = (d11 * d20 - d01 * d21) / denom;
  coords.y = (d00 * d21 - d01 * d20) / denom;
  coords.z = 1.0f - coords.x - coords.y;
  return coords;
}

vector<uint32_t, N> GetTreeIndex<let N : int>(vector<float, N> p) {
  vector<uint32_t, N> tree_index =
      vector<uint32_t, N>(floor(((p - voxel_tree_header.min_bound) / box_size) * grid_n)) + 1;
  tree_index = select(tree_index == grid_n + 1, tree_index - 1, tree_index);
  return tree_index - 1;
}

const bool RayBox2dIntersection(const float2 ray_origin, const float2 ray_dir, const float2 box_min,
                                const float2 box_max) {
  const float2 inv_dir = 1.0f / ray_dir;

  const float2 t0 = (box_min - ray_origin) * inv_dir;
  const float2 t1 = (box_max - ray_origin) * inv_dir;

  const float2 tmin2 = min(t0, t1);
  const float2 tmax2 = max(t0, t1);

  const float tmin = max(tmin2.x, tmin2.y);
  const float tmax = min(tmax2.x, tmax2.y);

  return tmax >= max(tmin, 0.0f);
}

void VoxelizeBarycentric(const Vertex vertices[3]) {
  const float3 positions[3] = {
    vertices[0].position,
    vertices[1].position,
    vertices[2].position,
  };

  const float2 uvs[3] = {
    vertices[0].uv,
    vertices[1].uv,
    vertices[2].uv,
  };

  const float3 normal = cross(positions[2] - positions[0], positions[1] - positions[0]);
  const float3 abs_normal = abs(cross(positions[1] - positions[0], positions[2] - positions[0]));
  const uint32_t projected_axis = (uint32_t)dot(float3(0, 1, 2), step(abs_normal.yzx, abs_normal.xyz) *
                                                                     step(abs_normal.zxy, abs_normal.xyz));

  const uint32_t2 non_projected_axises_map[3] = {
    /*x:*/ uint32_t2(1, 2),
    /*y:*/ uint32_t2(0, 2),
    /*z:*/ uint32_t2(1, 0),
  };

  const uint32_t2 non_projected_axises = non_projected_axises_map[projected_axis];

  const float2 plane_slope =
      -float2(normal[non_projected_axises[0]], normal[non_projected_axises[1]]) / normal[projected_axis];

  const float half_voxel_size = voxel_size * 0.5f;

  const float2 projected_positions[3] = {
    float2(positions[0][non_projected_axises[0]], positions[0][non_projected_axises[1]]),
    float2(positions[1][non_projected_axises[0]], positions[1][non_projected_axises[1]]),
    float2(positions[2][non_projected_axises[0]], positions[2][non_projected_axises[1]]),
  };

  const float2 projected_edges[3] = {
    projected_positions[1] - projected_positions[0],
    projected_positions[2] - projected_positions[1],
    projected_positions[0] - projected_positions[2],
  };

  const float min_x = min3(projected_positions[0].x, projected_positions[1].x, projected_positions[2].x);
  const float min_y = min3(projected_positions[0].y, projected_positions[1].y, projected_positions[2].y);

  const float max_x = max3(projected_positions[0].x, projected_positions[1].x, projected_positions[2].x);
  const float max_y = max3(projected_positions[0].y, projected_positions[1].y, projected_positions[2].y);

  const uint32_t2 min_tree_index = GetTreeIndex(float2(min_x, min_y));
  const uint32_t2 max_tree_index = GetTreeIndex(float2(max_x, max_y));

  for (uint32_t x = min_tree_index.x; x <= max_tree_index.x; x++) {
    for (uint32_t y = min_tree_index.y; y <= max_tree_index.y; y++) {
      const float2 voxel_position = float2(x, y) * voxel_size + voxel_tree_header.min_bound + half_voxel_size;

      const float3 barycentric = GetBarycentric(projected_positions, voxel_position);
      uint32_t3 tree_index;
      tree_index[non_projected_axises[0]] = x;
      tree_index[non_projected_axises[1]] = y;
      tree_index[projected_axis] = GetTreeIndex(positions[0][projected_axis] +
                                                dot(voxel_position - projected_positions[0], plane_slope));

      if (min3(barycentric.x, barycentric.y, barycentric.z) >= 0.0f) {
        const float2 uv = uvs[0] * barycentric.x + uvs[1] * barycentric.y + uvs[2] * barycentric.z;
        AddVoxel(tree_index, float4(uv, 0.0f, 1.0f));
      }
    }
  }
}

[[shader("compute")]]
[numthreads(64, 1, 1)]
void CMain(const uint32_t triangle_id: SV_DispatchThreadID, const uniform PushConstants pc) {
  if (triangle_id >= pc.triangle_count)
    return;

  // get vertices of dispatch
  const uint32_t index_offset = triangle_id * 3;
  const Vertex vertices[3] = {
    vertex_arr[index_offset + 0],
    vertex_arr[index_offset + 1],
    vertex_arr[index_offset + 2],
  };

  VoxelizeBarycentric(vertices);
}

void AddVoxel(int32_t3 index, const float4 color) {
  uint32_t3 level_index = index >> ((voxel_tree_header.max_depth - 1));
  uint32_t local_index = (level_index.x & 1) + ((level_index.y & 1) << 1) + ((level_index.z & 1) << 2);
  uint32_t child_ptr = local_index;

  bool retry;
  bool exit;

  for (uint32_t i = 1; i < voxel_tree_header.max_depth; i++) {
    level_index = index >> ((voxel_tree_header.max_depth - 1) - i);
    local_index = (level_index.x & 1) + ((level_index.y & 1) << 1) + ((level_index.z & 1) << 2);

    const uint32_t new_child_ptr = AddChild(local_index, child_ptr, retry, exit);

    if (exit) {
      return;
    } else if (retry) {
      i--;
    } else {
      child_ptr = new_child_ptr;
    }
  }

  AddLeafNode(child_ptr, PackRGBA8(color));
}

void AddLeafNode(const uint32_t tree_index, const uint32_t data) {
  const uint32_t page_index = tree_index / voxel_tree_header.page_size;

  if (page_index >= voxel_tree_header.allocated_page_count) {
    return;
  }

  const uint32_t page_offset = tree_index - page_index * voxel_tree_header.page_size;

  voxel_tree[page_index][page_offset].store(data);
}

uint32_t AddChild(const uint32_t local_index, const uint32_t tree_index, out bool retry, out bool exit) {
  retry = false;
  exit = false;
  const uint32_t page_index = tree_index / voxel_tree_header.page_size;

  if (page_index >= voxel_tree_header.allocated_page_count) {
    exit = true;
    return 0;
  }

  const uint32_t page_offset = tree_index - page_index * voxel_tree_header.page_size;

  uint32_t value = voxel_tree[page_index][page_offset].compareExchange(0, 0xFFFFFF) & 0xFFFFFF;

  if (value == 0) {
    const uint32_t ptr = voxel_tree_header.AllocatedVoxels(8);

    value = (ptr - tree_index) & 0xFFFFFF;

    voxel_tree[page_index][page_offset] = value;
  }

  if (value == 0xFFFFFF) {
    retry = true;
    return 0;
  }

  voxel_tree[page_index][page_offset].or(1 << (local_index + VOXEL_PTR_OFFSET)); // set valid node

  return value + local_index + tree_index;
}

