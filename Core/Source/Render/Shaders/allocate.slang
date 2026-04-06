import common;
import voxelize;

[[vk::binding(1, 0)]]
StructuredBuffer<Vertex> vertex_arr;

[[vk::binding(0, 1)]]
RWStructuredBuffer<AtomicBranchNode> voxel_tree[];

[[vk::binding(1, 1)]]
RWTreeHeader voxel_tree_header;

struct PushConstants {
  uint32_t triangle_count;
  uint32_t depth;
};

struct Func : IVoxelizeFn {
  void operator()(const uint32_t3 index, const uint32_t depth, const float2 uv) { Allocate(index, depth); }
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
    voxel_tree[page_index + page_level_offset][page_offset].child_ptr =
        voxel_tree_header.AllocateVoxels(depth, (depth == voxel_tree_header.max_depth - 1) ? 64: 64);
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

  Func func;

  Voxelize(vertices, pc.depth, voxel_tree_header, func);
}
