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
RWTreeHeader voxel_tree_header;

[[vk::binding(2, 1)]]
RWStructuredBuffer<uint32_t> voxel_tree_leaf[];

uint32_t GetLocalIndex(const uint32_t3 index, const uint32_t depth) {
  return dot(1, (((index >> ((voxel_tree_header.max_depth - 1) - depth) * 2)) & 0x3) << uint32_t3(0, 2, 4));
}

struct Func : IVoxelizeFn {
  void operator()(const uint32_t3 index, const uint32_t depth, const float2 uv) {
    AllocateChildMasks(index, depth, uv);
  }
};

void AllocateChildMasks(const uint32_t3 index, const uint32_t depth, const float2 uv) {
  uint32_t ptr = GetLocalIndex(index, 0);
  uint32_t local_index;

  for (uint32_t i = 1; i <= depth; i++) {
    const uint32_t level_page_offset = voxel_tree_header.GetLevelPageOffset(i - 1);
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    local_index = GetLocalIndex(index, i);

    InterlockedOr(voxel_tree[page_index + level_page_offset][page_offset].child_mask, uint64_t(1)
                                                                                          << local_index);

    ptr = voxel_tree[page_index + level_page_offset][page_offset].child_ptr +
          ((i == depth) ? (local_index / 8) : local_index);
  }

  const uint32_t level_page_offset = voxel_tree_header.GetLevelPageOffset(depth);
  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);
  const uint32_t node_offset = local_index & 0x3;
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

  Func func;
  Voxelize(vertices, pc.depth, voxel_tree_header, func);
}
