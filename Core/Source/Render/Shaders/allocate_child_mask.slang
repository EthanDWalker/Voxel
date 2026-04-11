import common;
import voxelize;

[[vk::binding(0, 0)]]
StructuredBuffer<uint32_t> index_arr;
[[vk::binding(1, 0)]]
StructuredBuffer<Vertex> vertex_arr;
[[vk::binding(2, 0)]]
Texture2D<float4> albedo_texture;
[[vk::binding(3, 0)]]
SamplerState sampler;
[[vk::binding(4, 0)]]
RWVoxelizationData voxelization_data;

[[vk::binding(0, 1)]]
RWStructuredBuffer<BranchNode> voxel_tree[];

[[vk::binding(1, 1)]]
RWStructuredBuffer<AtomicLeafNode> voxel_tree_leaf[];

[[vk::binding(2, 1)]]
RWTreeHeader voxel_tree_header;

uint32_t GetLocalIndex(const uint32_t3 index, const uint32_t depth) {
  return dot(1, (((index >> ((voxel_tree_header.max_depth - 1) - depth) * 2)) & 0x3) << uint32_t3(0, 2, 4));
}

void AllocateChildMasks(const uint32_t3 index, const uint32_t depth, const float2 uv,
                        const uint32_t invalidated_depth) {

  const uint32_t memory_index = spirv_asm {
        result:$$uint = OpLoad builtin(LocalInvocationIndex:uint);
    };

  uint32_t ptr = GetLocalIndex(index, 0);
  uint32_t local_index;

  for (uint32_t i = 1; i <= depth; i++) {
    const uint32_t level_page_offset = voxel_tree_header.GetLevelPageOffset(i - 1);
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    local_index = GetLocalIndex(index, i);

    InterlockedOr(voxel_tree[page_index + level_page_offset][page_offset].child_mask, uint64_t(1)
                                                                                          << local_index);

    ptr = voxel_tree[page_index + level_page_offset][page_offset].child_ptr + local_index;
  }

  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);
  voxel_tree_leaf[page_index][page_offset].color =
      PackRGBA8(albedo_texture.SampleLevel(sampler, uv.yx, 0.0f));
}

struct PushConstants {
  uint32_t triangle_count;
  uint32_t depth;
};

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

  static groupshared uint32_t ptr_stack_mem[64][PTR_STACK_SIZE];

  static const uint32_t initial_ptr_stack_depth = voxel_tree_header.max_depth - PTR_STACK_SIZE;

  const uint32_t memory_index = spirv_asm {
        result:$$uint = OpLoad builtin(LocalInvocationIndex:uint);
    };

  for (uint32_t x = min_tree_index.x; x <= max_tree_index.x; x++) {
    for (uint32_t y = min_tree_index.y; y <= max_tree_index.y; y++) {
      const float2 voxel_min_bound = float2(x, y) * voxel_size + voxel_tree_header.min_bound;
      const float2 voxel_max_bound = voxel_min_bound + voxel_size;
      const float2 voxel_center = voxel_min_bound + half_voxel_size;

      const float3 barycentric = GetBarycentric(projected_positions, voxel_center);

      if (min3(barycentric.x, barycentric.y, barycentric.z) >= 0.0f) {
        const float2 uv = uvs[0] * barycentric.x + uvs[1] * barycentric.y + uvs[2] * barycentric.z;

        uint32_t3 tree_index;
        tree_index[non_projected_axises[0]] = x;
        tree_index[non_projected_axises[1]] = y;
        tree_index[projected_axis] =
            GetTreeIndex(voxel_tree_header, positions[0][projected_axis] +
                                                dot(voxel_center - projected_positions[0], plane_slope));

        const uint32_t3 diff = last_index ^ tree_index;
        const uint32_t invalidated_stack_size = firstbithigh(diff.x | diff.y | diff.z) >> 1;

        uint32_t start_depth = invalidated_stack_size < PTR_STACK_SIZE
                                   ? initial_ptr_stack_depth + ((PTR_STACK_SIZE - 1) - invalidated_stack_size)
                                   : 1;
        uint32_t ptr = invalidated_stack_size < PTR_STACK_SIZE
                           ? ptr_stack_mem[memory_index][start_depth - initial_ptr_stack_depth]
                           : GetLocalIndex(tree_index, 0);
        uint32_t local_index;

        for (uint32_t i = start_depth; i < voxel_tree_header.max_depth; i++) {
          const uint32_t level_page_offset = voxel_tree_header.GetLevelPageOffset(i - 1);
          const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
          const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

          local_index = GetLocalIndex(tree_index, i);

          InterlockedOr(voxel_tree[page_index + level_page_offset][page_offset].child_mask,
                        uint64_t(1) << local_index);

          if (i >= initial_ptr_stack_depth) {
            ptr_stack_mem[memory_index][i - initial_ptr_stack_depth] = ptr;
          }

          ptr = voxel_tree[page_index + level_page_offset][page_offset].child_ptr + local_index;
        }

        const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
        const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);
        voxel_tree_leaf[page_index][page_offset].color =
            PackRGBA8(albedo_texture.SampleLevel(sampler, uv.yx, 0.0f));

        last_index = tree_index;
      }
    }
  }
}

