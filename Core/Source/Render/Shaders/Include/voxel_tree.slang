#pragma once
#include "common.slang"

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
  uint32_t radiance;
};

struct AtomicLeafNode {
  Atomic<uint32_t> color;
  Atomic<uint32_t> radiance;
};

struct RWVoxelizationData {
  private RWByteAddressBuffer buffer;

  void DocumentAllocation() { buffer.InterlockedAdd(0, 1); }
  void Notify() { buffer.InterlockedAdd(4, 1); }
};

struct BaseTreeHeader<T>
    where T : IBuffer {
  T buffer;

  property float min_bound {
    get { return buffer.Load<float, 0>(); }
  };

  property float max_bound {
    get { return buffer.Load<float, 4>(); }
  };

  property uint32_t max_depth {
    get { return buffer.Load<uint32_t, 8>(); }
  };

  property uint32_t page_size_exp {
    get { return buffer.Load<uint32_t, 12>(); }
  };
};

struct RWTreeHeader : BaseTreeHeader<RWByteBuffer> {
  const uint32_t GetLevelPageOffset(const uint32_t level) {
    return buffer.data.Load<uint32_t>(20 + (level * sizeof(uint32_t)));
  }

  const uint32_t AllocateLeafVoxels(const uint32_t amount) {
    uint32_t val;
    buffer.data.InterlockedAdd(16, amount, val);
    return val;
  }

  const uint32_t AllocateVoxels(const uint32_t level, const uint32_t amount) {
    uint32_t val;
    buffer.data.InterlockedAdd(20 + ((max_depth - 1) + level) * sizeof(uint32_t), amount, val);
    return val;
  }
};

struct TreeHeader : BaseTreeHeader<ByteBuffer> {
  const uint32_t GetLevelPageOffset(const uint32_t level) {
    return buffer.data.Load<uint32_t>(20 + (level * sizeof(uint32_t)));
  }
}

// clang-format off
#ifndef VOXEL_TREE_SET_BINDING
  #define VOXEL_TREE_SET_BINDING -1
#endif

#ifdef VOXEL_TREE_ATOMIC

  [[vk::binding(0, VOXEL_TREE_SET_BINDING)]]
  RWStructuredBuffer<AtomicBranchNode> voxel_tree[];
  [[vk::binding(1, VOXEL_TREE_SET_BINDING)]]
  RWStructuredBuffer<AtomicLeafNode> voxel_tree_leaf[];

#else

  #ifdef VOXEL_TREE_READ_WRITE

    [[vk::binding(0, VOXEL_TREE_SET_BINDING)]]
    RWStructuredBuffer<BranchNode> voxel_tree[];
    [[vk::binding(1, VOXEL_TREE_SET_BINDING)]]
    RWStructuredBuffer<LeafNode> voxel_tree_leaf[];

  #else

    [[vk::binding(0, VOXEL_TREE_SET_BINDING)]]
    StructuredBuffer<BranchNode> voxel_tree[];
    [[vk::binding(1, VOXEL_TREE_SET_BINDING)]]
    StructuredBuffer<LeafNode> voxel_tree_leaf[];

  #endif

#endif

#ifdef VOXEL_TREE_READ_WRITE

  [[vk::binding(2, VOXEL_TREE_SET_BINDING)]]
  RWTreeHeader voxel_tree_header;

#else

  [[vk::binding(2, VOXEL_TREE_SET_BINDING)]]
  TreeHeader voxel_tree_header;

#endif
// clang-format on

static const float box_size = voxel_tree_header.max_bound - voxel_tree_header.min_bound;
static const float grid_n = 1 << (voxel_tree_header.max_depth * 2);
static const float voxel_size = box_size / grid_n;
static const uint32_t initial_ptr_stack_depth = voxel_tree_header.max_depth - PTR_STACK_SIZE;
static groupshared uint32_t ray_ptr_stack_mem[64][PTR_STACK_SIZE];

vector<uint32_t, N> GetTreeIndex<let N : int>(vector<float, N> p) {
  const vector<float, N> offset = vector<float, N>(p - voxel_tree_header.min_bound);
  return vector<uint32_t, N>((offset / box_size) * float(grid_n));
}

uint32_t GetLocalIndex(const uint32_t3 index, const uint32_t depth) {
  return dot(1, (((index >> ((voxel_tree_header.max_depth - 1) - depth) * 2)) & 0x3) << uint32_t3(0, 2, 4));
}

#ifndef VOXEL_TREE_ATOMIC

struct RaycastResult {
  float4 hit_color;
  float3 hit_position;
  bool hit;
  float3 hit_normal;
  bool error;
  uint32_t3 tree_index;
  float t;
  uint32_t iterations;
};

void Raycast(const float3 ray_origin, float3 ray_dir, out RaycastResult result) {
  result.hit = false;
  result.error = false;
  result.t = 0.0f;
  result.hit_color = 0.0f;
  result.iterations = 0;
  ray_dir += 1e-4f;
  float ray_t_min, ray_t_max;
  const bool intersect = RayAABBIntersect(ray_dir, ray_origin, voxel_tree_header.min_bound * (1.0f - 1e-5f),
                                          voxel_tree_header.max_bound * (1.0f - 1e-5f), ray_t_min, ray_t_max);
  if (!intersect) {
    return;
  }

  ray_t_min = max(ray_t_min, 0.0f);
  result.t += ray_t_min;

  float3 position = ray_origin + ray_dir * ray_t_min;

  const float3 inv_ray_dir = 1.0f / ray_dir;
  const float3 abs_inv_ray_dir = abs(inv_ray_dir);
  const float3 ray_dir_step = step(0.0f, ray_dir);
  const int32_t3 voxel_step = sign(ray_dir);

  uint32_t3 current_index = GetTreeIndex(position);

  const uint32_t memory_index = spirv_asm {
        result:$$uint = OpLoad builtin(LocalInvocationIndex:uint);
    };

  uint32_t3 last_index = 0xFFFFFFFF;
  bool is_empty = true;
  BranchNode node;
  while (all(position >= voxel_tree_header.min_bound) && all(position <= voxel_tree_header.max_bound)) {
    const uint32_t3 index_diff = last_index ^ current_index;
    const uint32_t invalidated_stack_size = firstbithigh(index_diff.x | index_diff.y | index_diff.z) >> 1;

    uint32_t level = invalidated_stack_size < PTR_STACK_SIZE
                         ? initial_ptr_stack_depth + ((PTR_STACK_SIZE - 1) - invalidated_stack_size)
                         : 1;
    uint32_t ptr = invalidated_stack_size < PTR_STACK_SIZE
                       ? ray_ptr_stack_mem[memory_index][level - initial_ptr_stack_depth]
                       : GetLocalIndex(current_index, 0);
    uint32_t local_index;

    if (invalidated_stack_size == 0) {
      local_index = GetLocalIndex(current_index, level);

      is_empty = ((node.child_mask >> local_index) & 0b1) == 0;
    } else {
      for (;; level++) {
        const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
        const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

        node = voxel_tree[page_index + voxel_tree_header.GetLevelPageOffset(level - 1)][page_offset];

        local_index = GetLocalIndex(current_index, level);

        is_empty = ((node.child_mask >> local_index) & 0b1) == 0;

        if (level >= initial_ptr_stack_depth) {
          ray_ptr_stack_mem[memory_index][level - initial_ptr_stack_depth] = ptr;
        }

        if (is_empty || level == voxel_tree_header.max_depth - 1)
          break;

        ptr = node.child_ptr + local_index;
      }
    }

    if (!is_empty) {
      ptr = node.child_ptr + local_index;
      const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
      const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

      result.hit_color = UnpackRGBA8(voxel_tree_leaf[page_index][page_offset].color);
      break;
    }

    const uint32_t level_shift = ((voxel_tree_header.max_depth - 1) - level) * 2 +
                                 uint32_t((node.child_mask >> (local_index & 0b101010) & 0x00330033) == 0);
    const float scale = (1 << level_shift) * voxel_size;

    const uint32_t3 level_index = current_index >> level_shift;
    const float3 voxel_min_bound = voxel_tree_header.min_bound + level_index * scale;
    const float3 intersect_pos = voxel_min_bound + ray_dir_step * scale;

    const float3 t_max = (intersect_pos - position) * inv_ray_dir;

    const float3 mask = step(t_max, min(t_max.yzx, t_max.zxy));

    const float3 neighbor_min_bound = voxel_min_bound + (voxel_step * scale * mask);
    const float3 neighbor_max_bound = neighbor_min_bound + scale;

    const float delta_t = dot(mask, t_max);
    result.t += delta_t;

    const float voxel_clamp_epsilon = 1e-3f;
    position = clamp(position + delta_t * ray_dir, neighbor_min_bound + voxel_clamp_epsilon,
                     neighbor_max_bound - voxel_clamp_epsilon);

    last_index = current_index;
    current_index = GetTreeIndex(position);

    if (result.iterations > 1000) {
      result.error = true;
      return;
    }
    result.iterations++;
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
  result.tree_index = current_index;
}

#endif
