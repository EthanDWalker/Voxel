import common;
public module voxelize;

static const uint32_t CHILD_PTR_OFFSET_MASK = 0xFFFFFF;
static const uint32_t MAX_CHILD_PTR_NO_FAR = 0x7FFFFF;
static const uint32_t FAR_PTR_BIT = 0x800000;
static const uint32_t CHILD_MASK_OFFSET = 24;
static const uint32_t UINT32_MAX = 0xFFFFFFFF;
static const bool CONSERVATINVE_VOXELIZATION = false;

struct RWVoxelizationData {
  private RWByteAddressBuffer buffer;

  void DocumentAllocation() { buffer.InterlockedAdd(0, 1); }
  void DocumentFarPtr() { buffer.InterlockedAdd(4, 1); }
  uint32_t AllocateNewVoxelPtr() {
    uint32_t val;
    buffer.InterlockedAdd(8, 2, val);
    return val;
  }
};

struct Vertex {
  float3 position;
  private uint32_t uv_data;

  property float2 uv {
    get { return float2(f16tof32(uv_data >> 16), f16tof32(uv_data & 0xFFFF)); }
  }
}

bool RayBox2dIntersection(const float2 ray_origin, const float2 ray_dir, const float2 box_min,
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

vector<uint32_t, N> GetTreeIndex<let N : int>(const RWTreeHeader voxel_tree_header, vector<float, N> p) {
  vector<uint32_t, N> tree_index =
      vector<uint32_t, N>(floor(
          ((p - voxel_tree_header.min_bound) / (voxel_tree_header.max_bound - voxel_tree_header.min_bound)) *
          (1 << voxel_tree_header.max_depth))) +
      1;
  tree_index = select(tree_index == (1 << voxel_tree_header.max_depth) + 1, tree_index - 1, tree_index);
  return tree_index - 1;
}

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

interface IVoxelizeFn {
  void operator()(const uint32_t3 index, const uint32_t depth, const float2 uv);
};

void Voxelize(const Vertex vertices[3], const uint32_t depth, const RWTreeHeader voxel_tree_header,
              const IVoxelizeFn voxelize_fn) {
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
  const uint32_t projected_axis = (abs_normal.x > abs_normal.y && abs_normal.x > abs_normal.z) ? 0
                                  : (abs_normal.y > abs_normal.z)                              ? 1
                                                                                               : 2;
  const uint32_t2 non_projected_axises_map[3] = {
    /*x:*/ uint32_t2(1, 2),
    /*y:*/ uint32_t2(0, 2),
    /*z:*/ uint32_t2(1, 0),
  };

  const uint32_t2 non_projected_axises = non_projected_axises_map[projected_axis];

  const float2 plane_slope =
      -float2(normal[non_projected_axises[0]], normal[non_projected_axises[1]]) / normal[projected_axis];

  const float box_size = voxel_tree_header.max_bound - voxel_tree_header.min_bound;
  const float grid_n = 1 << voxel_tree_header.max_depth;
  const float voxel_size = box_size / grid_n;

  const float half_voxel_size = voxel_size * 0.5f;

  const float2 projected_positions[3] = {
    float2(positions[0][non_projected_axises[0]], positions[0][non_projected_axises[1]]),
    float2(positions[1][non_projected_axises[0]], positions[1][non_projected_axises[1]]),
    float2(positions[2][non_projected_axises[0]], positions[2][non_projected_axises[1]]),
  };

  const float2 projected_edges[3] = {
    normalize(projected_positions[1] - projected_positions[0]),
    normalize(projected_positions[2] - projected_positions[1]),
    normalize(projected_positions[0] - projected_positions[2]),
  };

  const float min_x = min3(projected_positions[0].x, projected_positions[1].x, projected_positions[2].x);
  const float min_y = min3(projected_positions[0].y, projected_positions[1].y, projected_positions[2].y);

  const float max_x = max3(projected_positions[0].x, projected_positions[1].x, projected_positions[2].x);
  const float max_y = max3(projected_positions[0].y, projected_positions[1].y, projected_positions[2].y);

  const uint32_t2 min_tree_index = GetTreeIndex(voxel_tree_header, float2(min_x, min_y));
  const uint32_t2 max_tree_index = GetTreeIndex(voxel_tree_header, float2(max_x, max_y));

  uint32_t3 last_index = 0xFFFFFFFF;

  for (uint32_t x = min_tree_index.x; x <= max_tree_index.x; x++) {
    for (uint32_t y = min_tree_index.y; y <= max_tree_index.y; y++) {
      const float2 voxel_min_bound = float2(x, y) * voxel_size + voxel_tree_header.min_bound;
      const float2 voxel_max_bound = voxel_min_bound + voxel_size;
      const float2 voxel_center = voxel_min_bound + half_voxel_size;

      const float3 barycentric = GetBarycentric(projected_positions, voxel_center);

      bool edge_hit = false;

      if (CONSERVATINVE_VOXELIZATION) {
        for (uint32_t i = 0; i < 3; i++) {
          if (RayBox2dIntersection(projected_positions[i], projected_edges[i], voxel_min_bound,
                                   voxel_max_bound)) {
            edge_hit = true;
            break;
          }
        }
      }

      if (min3(barycentric.x, barycentric.y, barycentric.z) >= 0.0f || edge_hit) {
        const float2 uv = uvs[0] * barycentric.x + uvs[1] * barycentric.y + uvs[2] * barycentric.z;

        uint32_t3 tree_index;
        tree_index[non_projected_axises[0]] = x;
        tree_index[non_projected_axises[1]] = y;
        tree_index[projected_axis] =
            GetTreeIndex(voxel_tree_header, positions[0][projected_axis] +
                                                dot(voxel_center - projected_positions[0], plane_slope));

        const uint32_t3 diff = tree_index ^ last_index;
        const uint32_t invalidated_depth = firstbithigh(diff.x | diff.y | diff.z);

        if (invalidated_depth >= (voxel_tree_header.max_depth - (depth + 1))) {
          voxelize_fn(tree_index, depth, uv);
          last_index = tree_index;
        }
      }
    }
  }
}
