#pragma once
#include "common.slang"

static const uint32_t CHILD_PTR_MASK = 0x7FFFFFFF;
static const uint32_t IS_LEAF_BIT_OFFSET = 31;
static const uint32_t IS_LEAF_BIT = 0x80000000;

static const uint32_t PTR_STACK_SIZE = 4;

struct VoxelVolume {
  uint32_t3 min_tree_index;
  uint32_t depth;
  uint32_t3 max_tree_index;
  uint32_t material_index;

  bool Contains(const uint32_t3 tree_index) {
    return all(tree_index >= min_tree_index && tree_index < max_tree_index);
  }
};

struct ChildPtrData {
  uint32_t data;

  __init(const uint32_t data) { this.data = data; }

  property uint32_t child_ptr {
    get { return data & CHILD_PTR_MASK; }
  }

  property bool is_leaf {
    get { return bool(data >> IS_LEAF_BIT_OFFSET); }
  }
};

struct BranchNode {
  uint64_t child_mask;
  ChildPtrData child_ptr_data;
}

struct Material {
  float4 albedo;
  float rough;
  float metallic;
  float emissive;
  float reflect;
};

struct AtomicBranchNode {
  Atomic<uint64_t> child_mask;
  Atomic<uint32_t> child_ptr_data;
};

struct LeafNode {
  uint16_t material_index;
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

  property uint32_t allocated_leaf_count {
    get { return buffer.Load<uint32_t, 24>(); }
  };

  property uint32_t allocated_branch_count {
    get { return buffer.Load<uint32_t, 28>(); }
  };
};

struct RWTreeHeader : BaseTreeHeader<RWByteBuffer> {
  const uint32_t AllocateLeafVoxels(const uint32_t amount) {
    uint32_t val;
    buffer.data.InterlockedAdd(16, amount, val);
    return val;
  }

  const uint32_t AllocateBranchVoxels(const uint32_t amount) {
    uint32_t val;
    buffer.data.InterlockedAdd(20, amount, val);
    return val;
  }
};

struct TreeHeader : BaseTreeHeader<ByteBuffer> {}

// clang-format off
#ifndef VOXEL_TREE_SET_BINDING
  #define VOXEL_TREE_SET_BINDING -1
#endif

#ifdef VOXEL_TREE_ATOMIC

  [[vk::binding(0, VOXEL_TREE_SET_BINDING)]]
  RWStructuredBuffer<AtomicBranchNode> voxel_tree[];
  [[vk::binding(1, VOXEL_TREE_SET_BINDING)]]
  RWStructuredBuffer<LeafNode> voxel_tree_leaf[];

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

[[vk::binding(3, VOXEL_TREE_SET_BINDING)]]
CountedBuffer<Material> material_arr;

static const float box_size = 1 << voxel_tree_header.box_size_exp;
static const uint32_t grid_n = 1 << (voxel_tree_header.max_depth << 1);
static const float voxel_size = FloatDivideBy2ToN(box_size, voxel_tree_header.max_depth << 1);
static const uint32_t initial_ptr_stack_depth = voxel_tree_header.max_depth - PTR_STACK_SIZE;
static groupshared uint32_t ptr_stack_mem[64][PTR_STACK_SIZE];
static const uint32_t memory_index = spirv_asm {
        result:$$uint = OpLoad builtin(LocalInvocationIndex:uint);
    };

uint32_t PackTreeIndex(const uint32_t3 tree_index) {
  return SumComponents(
      tree_index << uint32_t3(0, voxel_tree_header.max_depth << 1, (voxel_tree_header.max_depth << 1) << 1));
}

float3 GetWorldPositionCenter(const uint32_t3 tree_index, const uint32_t level) {
  const uint32_t level_shift = ((voxel_tree_header.max_depth - 1) - level) << 1;
  const float level_voxel_size = FloatMultiplyBy2ToN(voxel_size, level_shift);
  return (tree_index >> level_shift) * level_voxel_size + voxel_tree_header.min_bound +
         (level_voxel_size * 0.5f);
}

vector<uint32_t, N> GetTreeIndex<let N : int>(const vector<float, N> p) {
  // clang-format off
  return vector<uint32_t, N>(
    asfloat(
      (
        asuint(vector<float, N>(p - voxel_tree_header.min_bound)) - (voxel_tree_header.box_size_exp << 23)
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

float3 StepTreeRaySingle(const float3 position, float3 ray_dir, const uint32_t level) {
  ray_dir += 1e-3f;

  const uint32_t level_shift = ((voxel_tree_header.max_depth - 1) - level) * 2;
  const float scale = (1 << level_shift) * voxel_size;

  const float3 inv_ray_dir = 1.0f / ray_dir;
  const float3 ray_dir_step = step(0.0f, ray_dir);
  const int32_t3 voxel_step = sign(ray_dir);

  const uint32_t3 level_index = GetTreeIndex(position) >> level_shift;
  const float3 voxel_min_bound = voxel_tree_header.min_bound + level_index * scale;
  const float3 intersect_pos = voxel_min_bound + ray_dir_step * scale;

  const float3 t_max = (intersect_pos - position) * inv_ray_dir;

  const float3 mask = step(t_max, min(t_max.yzx, t_max.zxy));

  const float3 neighbor_min_bound = voxel_min_bound + (voxel_step * scale * mask);
  const float3 neighbor_max_bound = neighbor_min_bound + scale;

  const float delta_t = dot(mask, t_max);

  const float voxel_clamp_epsilon = 2e-5f;
  return clamp(position + delta_t * ray_dir, neighbor_min_bound + voxel_clamp_epsilon,
               neighbor_max_bound - voxel_clamp_epsilon);
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

  const float voxel_clamp_epsilon = 2e-5f;
  return clamp(position + delta_t * ray_dir, neighbor_min_bound + voxel_clamp_epsilon,
               neighbor_max_bound - voxel_clamp_epsilon);
}

LeafNode GetLeafNodeAtFilledIndex(const uint32_t3 tree_index, out uint32_t ptr, out uint32_t level) {
  uint32_t local_index = GetLocalIndex(tree_index, 0);
  BranchNode node;

  ptr = local_index;
  for (level = 1;; level++) {
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

#ifdef VOXEL_TREE_ATOMIC
    node.child_ptr_data = ChildPtrData(voxel_tree[page_index][page_offset].child_ptr_data.load());
    node.child_mask = voxel_tree[page_index][page_offset].child_mask.load();
#else
    node = voxel_tree[page_index][page_offset];
#endif

    local_index = GetLocalIndex(tree_index, level);

    ptr = node.child_ptr_data.child_ptr + local_index;

    if (node.child_ptr_data.is_leaf) {
      break;
    }
  }

  const uint32_t leaf_page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t leaf_page_offset = ptr - (leaf_page_index << voxel_tree_header.page_size_exp);
  return voxel_tree_leaf[leaf_page_index][leaf_page_offset];
}

void TraverseTreeFromLevelPtr(inout uint32_t ptr, inout uint32_t level, inout BranchNode node,
                              inout bool is_empty, inout uint32_t local_index, const uint32_t3 index) {
  for (;; level++) {
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

#ifdef VOXEL_TREE_ATOMIC
    node.child_ptr_data = ChildPtrData(voxel_tree[page_index][page_offset].child_ptr_data.load());
    node.child_mask = voxel_tree[page_index][page_offset].child_mask.load();
#else
    node = voxel_tree[page_index][page_offset];
#endif

    local_index = GetLocalIndex(index, level);

    is_empty = ((node.child_mask >> local_index) & 0b1) == 0;

    if (level >= initial_ptr_stack_depth) {
      ptr_stack_mem[memory_index][level - initial_ptr_stack_depth] = ptr;
    }

    if (is_empty || node.child_ptr_data.is_leaf)
      break;

    ptr = node.child_ptr_data.child_ptr + local_index;
  }
}

#ifndef VOXEL_TREE_ATOMIC

struct RaycastResult {
  float3 hit_position;
  uint32_t iterations;
  float3 hit_normal;
  float t;
  uint32_t3 hit_tree_index;
  uint32_t leaf_ptr;
  uint32_t hit_material_index;
  uint32_t hit_level;
  bool hit;
};

void Raycast(const float3 ray_origin, float3 ray_dir, out RaycastResult result, const float t_max = 1e10f) {
  result.hit = false;
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
  const float3 ray_dir_step = step(0.0f, ray_dir);
  const int32_t3 voxel_step = sign(ray_dir);

  uint32_t3 current_index = GetTreeIndex(position);

  uint32_t3 last_index = 0xFFFFFFFF;
  bool is_empty = true;
  BranchNode node;
  uint32_t local_index;
  uint32_t ptr;
  uint32_t level;

  while (all(position >= voxel_tree_header.min_bound) &&
         all(position <= (voxel_tree_header.min_bound + box_size))) {
    const uint32_t3 index_diff = last_index ^ current_index;
    const uint32_t invalidated_stack_size = firstbithigh(index_diff.x | index_diff.y | index_diff.z) >> 1;

    if (invalidated_stack_size < PTR_STACK_SIZE) {
      level = initial_ptr_stack_depth + ((PTR_STACK_SIZE - 1) - invalidated_stack_size);
      ptr = ptr_stack_mem[memory_index][level - initial_ptr_stack_depth];
    } else {
      level = 1;
      ptr = GetLocalIndex(current_index, 0);
    }

    TraverseTreeFromLevelPtr(ptr, level, node, is_empty, local_index, current_index);

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
  result.hit_level = level;

  ptr = node.child_ptr_data.child_ptr + local_index;
  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

  const LeafNode leaf_node = voxel_tree_leaf[page_index][page_offset];
  result.hit_material_index = leaf_node.material_index;
  result.leaf_ptr = ptr;

  const float3 voxel_center = GetWorldPositionCenter(result.hit_tree_index, result.hit_level);
  const float3 hit_offset = (result.hit_position - voxel_center);
  const float3 abs_hit_offset = abs(result.hit_position - voxel_center);
  result.hit_normal =
      sign(hit_offset) *
      (abs_hit_offset.x > abs_hit_offset.y && abs_hit_offset.x > abs_hit_offset.z
           ? float3(1.0f, 0.0f, 0.0f)
           : (abs_hit_offset.y > abs_hit_offset.z ? float3(0.0f, 1.0f, 0.0f) : float3(0.0f, 0.0f, 1.0f)));
}

#endif

static const int32_t3 offset_masks[26] = {
  // top
  int32_t3(0, 1, 0),
  int32_t3(1, 1, 0),
  int32_t3(-1, 1, 0),
  int32_t3(0, 1, 1),
  int32_t3(0, 1, -1),

  int32_t3(1, 1, -1),
  int32_t3(1, 1, 1),
  int32_t3(-1, 1, -1),
  int32_t3(-1, 1, 1),

  // middle (no (0, 0, 0))
  int32_t3(1, 0, 0),
  int32_t3(-1, 0, 0),
  int32_t3(0, 0, 1),
  int32_t3(0, 0, -1),

  int32_t3(1, 0, -1),
  int32_t3(1, 0, 1),
  int32_t3(-1, 0, -1),
  int32_t3(-1, 0, 1),

  // bottom
  int32_t3(0, -1, 0),
  int32_t3(1, -1, 0),
  int32_t3(-1, -1, 0),
  int32_t3(0, -1, 1),
  int32_t3(0, -1, -1),

  int32_t3(1, -1, -1),
  int32_t3(1, -1, 1),
  int32_t3(-1, -1, -1),
  int32_t3(-1, -1, 1),
};

