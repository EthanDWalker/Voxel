import common;
import voxelize;

[[vk::binding(1, 0)]]
StructuredBuffer<Vertex> vertex_arr;

[[vk::binding(0, 1)]]
RWStructuredBuffer<AtomicBranchNode> voxel_tree[];

[[vk::binding(2, 1)]]
RWTreeHeader voxel_tree_header;

struct PushConstants {
  uint32_t triangle_count;
  uint32_t depth;
};

uint32_t GetLocalIndex(const uint32_t3 index, const uint32_t depth) {
  return dot(1, (((index >> ((voxel_tree_header.max_depth - 1) - depth) * 2)) & 0x3) << uint32_t3(0, 2, 4));
}

void Allocate(const uint32_t3 index, const uint32_t depth) {
  uint32_t ptr = GetLocalIndex(index, 0);

  for (uint32_t i = 1; i < depth; i++) {
    const uint32_t page_level_offset = voxel_tree_header.GetLevelPageOffset(i - 1);
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    const uint32_t local_index = GetLocalIndex(index, i);

    const uint32_t child_ptr = voxel_tree[page_index + page_level_offset][page_offset].child_ptr.load();

    ptr = child_ptr + local_index;
  }

  const uint32_t page_level_offset = voxel_tree_header.GetLevelPageOffset(depth - 1);
  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;

  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

  if (voxel_tree[page_index + page_level_offset][page_offset].child_ptr.compareExchange(SENTINAL, 0) ==
      SENTINAL) {
    if (depth == voxel_tree_header.max_depth - 1) {
      voxel_tree[page_index + page_level_offset][page_offset].child_ptr =
          voxel_tree_header.AllocateLeafVoxels(64);
    } else {
      voxel_tree[page_index + page_level_offset][page_offset].child_ptr =
          voxel_tree_header.AllocateVoxels(depth, 64);
    }
  }
}

[numthreads(64, 1, 1)]
void CMain(const uint32_t triangle_id: SV_DispatchThreadID, const uniform PushConstants pc) {
  if (triangle_id >= pc.triangle_count) {
    return;
  }

  const Vertex vertices[3] = {
    vertex_arr[triangle_id * 3 + 0],
    vertex_arr[triangle_id * 3 + 1],
    vertex_arr[triangle_id * 3 + 2],
  };

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
  const float grid_n = 1 << (voxel_tree_header.max_depth * 2);
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

      if (min3(barycentric.x, barycentric.y, barycentric.z) >= 0.0f) {
        uint32_t3 tree_index;
        tree_index[non_projected_axises[0]] = x;
        tree_index[non_projected_axises[1]] = y;
        tree_index[projected_axis] =
            GetTreeIndex(voxel_tree_header, positions[0][projected_axis] +
                                                dot(voxel_center - projected_positions[0], plane_slope));

        const uint32_t3 diff = last_index ^ tree_index;
        const uint32_t invalidated_depth = firstbithigh(diff.x | diff.y | diff.z) >> 1;
        if (invalidated_depth >= (voxel_tree_header.max_depth - 1) - pc.depth) {
          Allocate(tree_index, pc.depth);
          last_index = tree_index;
        }
      }
    }
  }
}
