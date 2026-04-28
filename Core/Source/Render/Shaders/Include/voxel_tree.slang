#pragma once
#include "common.slang"

static const uint32_t SENTINAL = 0xFFFFFFFF;

struct BranchNode {
  uint64_t child_mask;
  uint32_t child_ptr;
};

struct AtomicBranchNode {
  Atomic<uint64_t> child_mask;
  Atomic<uint32_t> child_ptr;
};

struct LeafNode {
  uint32_t color;
  uint32_t normal;
};

struct AtomicLeafNode {
  Atomic<uint32_t> color;
  Atomic<uint32_t> normal;
};

struct VoxelizeInfo {
  float4 position : SV_Position;
  float3 world_position;
  float16_t2 uv;
  float3 normal;
};

struct BaseTreeHeader<T>
    where T : IBuffer {
  T buffer;

  property float min_bound {
    get { return buffer.Load<float, 0>(); }
  };

  property uint32_t box_size_exp {
    get { return buffer.Load<uint32_t, 4>(); }
  };

  property uint32_t max_depth {
    get { return buffer.Load<uint32_t, 8>(); }
  };

  property uint32_t page_size_exp {
    get { return buffer.Load<uint32_t, 12>(); }
  };
};

struct RWTreeHeader : BaseTreeHeader<RWByteBuffer> {
  const uint32_t AllocateLeafVoxels(const uint32_t amount) {
    uint32_t val;
    buffer.data.InterlockedAdd(16, amount, val);
    return val;
  }

  const uint32_t GetLevelPageOffset(const uint32_t level) {
    return buffer.data.Load<uint32_t>(20 + (level * sizeof(uint32_t)));
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

  [[vk::binding(4, VOXEL_TREE_SET_BINDING)]]
  RWTreeHeader voxel_tree_header;

#else

  [[vk::binding(4, VOXEL_TREE_SET_BINDING)]]
  TreeHeader voxel_tree_header;

#endif
// clang-format on

vector<float, N> FloatDivideBy2ToN<let N : int>(const vector<float, N> x, const uint32_t n) {
  return asfloat(asuint(x) - (n << 23));
}

vector<float, N> FloatMultiplyBy2ToN<let N : int>(const vector<float, N> x, const uint32_t n) {
  return asfloat(asuint(x) + (n << 23));
}

static const float box_size = 1 << voxel_tree_header.box_size_exp;
static const float grid_n = 1 << (voxel_tree_header.max_depth << 1);
static const float voxel_size = FloatDivideBy2ToN(box_size, voxel_tree_header.max_depth << 1);
static const float voxel_radius = sqrt(voxel_size * voxel_size * voxel_size * 0.5f);
static const uint32_t initial_ptr_stack_depth = voxel_tree_header.max_depth - PTR_STACK_SIZE;
static groupshared uint32_t ray_ptr_stack_mem[64][PTR_STACK_SIZE];
static const uint32_t memory_index = spirv_asm {
        result:$$uint = OpLoad builtin(LocalInvocationIndex:uint);
    };

vector<uint32_t, N> GetTreeIndex<let N : int>(const vector<float, N> p) {
  const vector<float, N> offset = vector<float, N>(p - voxel_tree_header.min_bound);
  // return vector<uint32_t, N>(asfloat(asuint(offset) - (voxel_tree_header.box_size_exp << 23)) *
  // float(grid_n));
  // clang-format off
  return vector<uint32_t, N>(
    asfloat(
      (
        asuint(offset) - (voxel_tree_header.box_size_exp << 23)
      ) 
        + ((voxel_tree_header.max_depth << 1) << 23)
    )
  );
  // clang-format on
}

uint32_t GetLocalIndex(const uint32_t3 index, const uint32_t depth) {
  return SumComponents((((index >> (((voxel_tree_header.max_depth - 1) - depth) << 1))) & 0x3)
                       << uint32_t3(0, 2, 4));
}

uint32_t3 GetLocalOffset(const uint32_t local_index) {
  return (uint32_t3(local_index) >> uint32_t3(0, 2, 4)) & uint32_t3(0x3);
}

uint32_t3 GetLocalOffset(const uint32_t3 index, const uint32_t depth) {
  return (index >> (((voxel_tree_header.max_depth - 1) - depth) << 1)) & 0x3;
}

void TraverseTreeFromLevelPtr(inout uint32_t ptr, inout uint32_t level, inout BranchNode node,
                              inout bool is_empty, inout uint32_t local_index, const uint32_t3 index,
                              const uint32_t max_depth) {
  for (;; level++) {
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

#ifdef VOXEL_TREE_ATOMIC
    node.child_ptr = voxel_tree[page_index + voxel_tree_header.GetLevelPageOffset(level - 1)][page_offset]
                         .child_ptr.load();
    node.child_mask = voxel_tree[page_index + voxel_tree_header.GetLevelPageOffset(level - 1)][page_offset]
                          .child_mask.load();
#else
    node = voxel_tree[page_index + voxel_tree_header.GetLevelPageOffset(level - 1)][page_offset];
#endif

    local_index = GetLocalIndex(index, level);

    is_empty = ((node.child_mask >> local_index) & 0b1) == 0;

    if (level >= initial_ptr_stack_depth) {
      ray_ptr_stack_mem[memory_index][level - initial_ptr_stack_depth] = ptr;
    }

    if (is_empty || level == (max_depth - 1))
      break;

    ptr = node.child_ptr + local_index;
  }
}

#ifndef VOXEL_TREE_ATOMIC

struct RaycastResult {
  float4 hit_color;
  float3 hit_position;
  uint32_t iterations;
  float3 hit_normal;
  float t;
  uint32_t3 hit_tree_index;
  bool error;
  bool hit;
};

float3 StepTreeRay(const float3 ray_dir_step, const float3 ray_dir, const float3 inv_ray_dir,
                   const float3 voxel_step, const float3 position, const uint32_t3 index,
                   const uint64_t child_mask, const uint32_t local_index, const uint32_t level,
                   inout float t) {
  const uint32_t level_shift = ((voxel_tree_header.max_depth - 1) - level) * 2 +
                               uint32_t((child_mask >> (local_index & 0b101010) & 0x00330033) == 0);
  const float scale = (1 << level_shift) * voxel_size;

  const uint32_t3 level_index = index >> level_shift;
  const float3 voxel_min_bound = voxel_tree_header.min_bound + level_index * scale;
  const float3 intersect_pos = voxel_min_bound + ray_dir_step * scale;

  const float3 t_max = (intersect_pos - position) * inv_ray_dir;

  const float3 mask = step(t_max, min(t_max.yzx, t_max.zxy));

  const float3 neighbor_min_bound = voxel_min_bound + (voxel_step * scale * mask);
  const float3 neighbor_max_bound = neighbor_min_bound + scale;

  const float delta_t = dot(mask, t_max);
  t += delta_t;

  const float voxel_clamp_epsilon = 1e-3f;
  return clamp(position + delta_t * ray_dir, neighbor_min_bound + voxel_clamp_epsilon,
               neighbor_max_bound - voxel_clamp_epsilon);
}

void Raycast(const float3 ray_origin, float3 ray_dir, out RaycastResult result, const float t_max = 1e10f) {
  result.hit = false;
  result.error = false;
  result.hit_color = 0.0f;
  result.t = 0.0f;
  result.iterations = 0;
  ray_dir += 1e-4f;
  float ray_t_min, ray_t_max;
  const bool intersect =
      RayAABBIntersect(ray_dir, ray_origin, voxel_tree_header.min_bound * (1.0f - 1e-5f),
                       (voxel_tree_header.min_bound + box_size) * (1.0f - 1e-5f), ray_t_min, ray_t_max);
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

  uint32_t3 last_index = 0xFFFFFFFF;
  bool is_empty = true;
  BranchNode node;
  uint32_t local_index;
  uint32_t ptr;

  while (all(position >= voxel_tree_header.min_bound) &&
         all(position <= (voxel_tree_header.min_bound + box_size))) {
    const uint32_t3 index_diff = last_index ^ current_index;
    const uint32_t invalidated_stack_size = firstbithigh(index_diff.x | index_diff.y | index_diff.z) >> 1;

    uint32_t level = invalidated_stack_size < PTR_STACK_SIZE
                         ? initial_ptr_stack_depth + ((PTR_STACK_SIZE - 1) - invalidated_stack_size)
                         : 1;
    ptr = invalidated_stack_size < PTR_STACK_SIZE
              ? ray_ptr_stack_mem[memory_index][level - initial_ptr_stack_depth]
              : GetLocalIndex(current_index, 0);

    if (invalidated_stack_size == 0) {
      local_index = GetLocalIndex(current_index, level);

      is_empty = ((node.child_mask >> local_index) & 0b1) == 0;
    } else {
      TraverseTreeFromLevelPtr(ptr, level, node, is_empty, local_index, current_index,
                               voxel_tree_header.max_depth);
    }

    if (!is_empty || result.t > t_max) {
      break;
    }

    position = StepTreeRay(ray_dir_step, ray_dir, inv_ray_dir, voxel_step, position, current_index,
                           node.child_mask, local_index, level, result.t);

    last_index = current_index;
    current_index = GetTreeIndex(position);

    result.iterations++;
  }

  if (is_empty) {
    result.hit = false;
    return;
  }
  result.hit = true;

  result.hit_position = position;
  result.hit_tree_index = current_index;

  ptr = node.child_ptr + local_index;
  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

  result.hit_color = UnpackRGBA8(voxel_tree_leaf[page_index][page_offset].color);
  result.hit_normal = UnpackNormal(voxel_tree_leaf[page_index][page_offset].normal);
}

#endif

