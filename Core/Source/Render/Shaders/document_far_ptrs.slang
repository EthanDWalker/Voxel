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
[[vk::binding(5, 0)]]
RWStructuredBuffer<uint32_t> new_voxel_ptrs;

[[vk::binding(0, 1)]]
RWStructuredBuffer<Atomic<uint32_t>> voxel_tree[];

[[vk::binding(1, 1)]]
RWStructuredBuffer<uint32_t> far_ptr_arr[];

[[vk::binding(2, 1)]]
RWTreeHeader voxel_tree_header;

static const float box_size = voxel_tree_header.max_bound - voxel_tree_header.min_bound;
static const float grid_n = 1 << voxel_tree_header.max_depth;
static const float voxel_size = box_size / grid_n;

vector<uint32_t, N> GetTreeIndex<let N : int>(vector<float, N> p) {
  vector<uint32_t, N> tree_index =
      vector<uint32_t, N>(floor(((p - voxel_tree_header.min_bound) / box_size) * grid_n)) + 1;
  tree_index = select(tree_index == grid_n + 1, tree_index - 1, tree_index);
  return tree_index - 1;
}

uint32_t GetLocalIndex(const uint32_t3 index, const uint32_t depth) {
  return dot(1, ((index >> ((voxel_tree_header.max_depth - 1) - depth)) & 1) << uint32_t3(0, 1, 2));
}

struct PushConstants {
  uint32_t triangle_count;
  uint32_t depth;
};

struct Func : IVoxelizeFn {
  void operator()(const uint32_t3 index, const uint32_t depth, const float2 uv) { Allocate(index, depth); }
}

void Allocate(const uint32_t3 index, const uint32_t depth) {
  uint32_t ptr = GetLocalIndex(index, 0);

  for (uint32_t i = 1; i < depth; i++) {
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    const uint32_t node =
        voxel_tree[page_index + voxel_tree_header.GetLevelPageOffset(i - 1)][page_offset].load();

    const uint32_t local_index = GetLocalIndex(index, i);

    if ((node & FAR_PTR_BIT) == 0) {
      ptr = (node & MAX_CHILD_PTR_NO_FAR) + local_index;
    } else {
      const uint32_t far_ptr = node & MAX_CHILD_PTR_NO_FAR;
      const uint32_t far_ptr_page_index = far_ptr >> voxel_tree_header.far_ptr_page_size_exp;
      const uint32_t far_ptr_page_offset =
          far_ptr - (far_ptr_page_index << voxel_tree_header.far_ptr_page_size_exp);

      ptr = far_ptr_arr[far_ptr_page_index][far_ptr_page_offset] + local_index;
    }
  }

  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

  const uint32_t res =
      voxel_tree[page_index + voxel_tree_header.GetLevelPageOffset(depth - 1)][page_offset].compareExchange(
          UINT32_MAX, 0);

  if (res == UINT32_MAX) {
    const uint32_t alloc_ptr = voxel_tree_header.AllocateVoxels(depth, 8);
    if (alloc_ptr > MAX_CHILD_PTR_NO_FAR) {
      voxelization_data.DocumentFarPtr();
    }

    const uint32_t index = voxelization_data.AllocateNewVoxelPtr();
    new_voxel_ptrs[index] = ptr;
    new_voxel_ptrs[index + 1] = alloc_ptr;
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
