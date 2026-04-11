import common;
public module voxelize;

static const bool CONSERVATINVE_VOXELIZATION = false;
static const uint32_t SENTINAL = 0xFFFFFFFF;

struct BranchNode {
  uint64_t child_mask;
  uint32_t child_ptr;
  uint32_t radiance;
};

struct AtomicBranchNode {
  Atomic<uint64_t> child_mask;
  Atomic<uint32_t> child_ptr;
  Atomic<uint32_t> radiance;
};

struct LeafNode {
  uint32_t color;
};

struct AtomicLeafNode {
  Atomic<uint32_t> color;
};

struct RWVoxelizationData {
  private RWByteAddressBuffer buffer;

  void DocumentAllocation() { buffer.InterlockedAdd(0, 1); }
  void Notify() { buffer.InterlockedAdd(4, 1); }
};

struct Vertex {
  float3 position;
  private uint32_t uv_data;

  property float2 uv {
    get { return float2(f16tof32(uv_data >> 16), f16tof32(uv_data & 0xFFFF)); }
  }
}

vector<uint32_t, N> GetTreeIndex<let N : int>(const RWTreeHeader voxel_tree_header, vector<float, N> p) {
  const uint32_t grid_n = 1 << (voxel_tree_header.max_depth * 2);
  const float box_size = voxel_tree_header.max_bound - voxel_tree_header.min_bound;
  const vector<float, N> offset = vector<float, N>(p - voxel_tree_header.min_bound);
  return vector<uint32_t, N>(floor((offset / box_size) * float(grid_n)));
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
  coords.y = (d00 * d21 - d01 * d20) / denom;
  coords.z = (d11 * d20 - d01 * d21) / denom;
  coords.x = 1.0f - coords.z - coords.y;
  return coords;
}
