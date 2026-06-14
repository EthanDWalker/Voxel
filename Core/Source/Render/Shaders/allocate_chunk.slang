#define VOXEL_TREE_ATOMIC
#define VOXEL_TREE_READ_WRITE
#define VOXEL_TREE_SET_BINDING 0
#include "Include/noise.slang"
#include "Include/voxel_tree.slang"

struct ChunkAllocationInfo {
  uint32_t3 chunk_index;
  uint32_t lod;
  uint32_t alloc_leaf;
  uint32_t color;
  uint32_t seed;
};

void Allocate(const uint32_t3 index, const uint32_t depth, const bool leaf) {
  uint32_t ptr = GetLocalIndex(index, 0);

  for (uint32_t i = 1; i < depth; i++) {
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    const uint32_t local_index = GetLocalIndex(index, i);

    const ChildPtrData child_ptr_data =
        ChildPtrData(voxel_tree[page_index][page_offset].child_ptr_data.load());

    if (child_ptr_data.is_leaf) {
      break;
    } else {
      ptr = child_ptr_data.child_ptr + local_index;
    }
  }

  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

  const uint32_t local_index = GetLocalIndex(index, depth);

  voxel_tree[page_index][page_offset].child_mask.or(uint64_t(1) << local_index);

  if (voxel_tree[page_index][page_offset].child_ptr_data.compareExchange(SENTINAL, 0) == SENTINAL) {
    if (leaf) {
      voxel_tree[page_index][page_offset].child_ptr_data =
          voxel_tree_header.AllocateLeafVoxels(64) | IS_LEAF_BIT;
    } else {
      voxel_tree[page_index][page_offset].child_ptr_data = voxel_tree_header.AllocateBranchVoxels(64);
    }
  }
}

float3 GetNormal(const uint32_t3 tree_index) {
  uint32_t local_index = GetLocalIndex(tree_index, 0);
  uint32_t ptr = local_index;
  uint32_t level = 1;
  bool is_empty = false;
  BranchNode node;
  float3 normal = 0.0f;

  for (int32_t x = -1; x <= 1; x++) {
    for (int32_t y = -1; y <= 1; y++) {
      for (int32_t z = -1; z <= 1; z++) {
        const int32_t3 offset = int32_t3(x, y, z);
        if (all(offset == 0))
          continue;

        const int32_t3 current_index = tree_index + offset;

        if (any(current_index < 0) || any(current_index >= grid_n)) {
          is_empty = true;
        } else {

          level = 1;
          ptr = GetLocalIndex(current_index, 0);

          TraverseTreeFromLevelPtr(ptr, level, node, is_empty, local_index, current_index);
        }

        if (is_empty) {
          normal += normalize(offset);
        }
      }
    }
  }

  return all(normal == 0.0f) ? float3(0.0f, 1.0f, 0.0f) : normalize(normal);
}

void AllocateLeaf(const uint32_t3 tree_index, const float3 color) {
  uint32_t local_index = GetLocalIndex(tree_index, 0);
  uint32_t ptr = local_index;
  uint32_t level = 1;
  bool is_empty = false;
  BranchNode node;

  TraverseTreeFromLevelPtr(ptr, level, node, is_empty, local_index, tree_index);

  if (is_empty) {
    return;
  }

  ptr = node.child_ptr_data.child_ptr + local_index;

  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);
  LeafNode leaf_node;

  leaf_node.normal = GetNormal(tree_index);
  leaf_node.color = float4(color, 1.0f);

  voxel_tree_leaf[page_index][page_offset].data.max(PackLeafNode(leaf_node));
}

void UpdateLeafNormal(const uint32_t3 tree_index) {
  uint32_t local_index = GetLocalIndex(tree_index, 0);
  uint32_t ptr = local_index;
  uint32_t level = 1;
  bool is_empty = false;
  BranchNode node;

  TraverseTreeFromLevelPtr(ptr, level, node, is_empty, local_index, tree_index);

  if (is_empty) {
    return;
  }

  ptr = node.child_ptr_data.child_ptr + local_index;

  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);
  LeafNode leaf_node = UnpackLeafNode(voxel_tree_leaf[page_index][page_offset].data.load());

  leaf_node.normal = GetNormal(tree_index);

  voxel_tree_leaf[page_index][page_offset].data.store(PackLeafNode(leaf_node));
}

static const float3 SNOW_COLOR = float3(0.96f, 0.86f, 0.79f);
static const float3 GRASS_COLOR = float3(0.08f, 0.43f, 0.08f);
static const float3 STONE_COLOR = float3(136, 140, 141) / 255.0f;
static const float3 DIRT_COLOR = float3(137, 81, 41) / 255.0f;

[numthreads(4, 4, 4)]
void CMain(const int32_t3 dispatch_id: SV_DispatchThreadID, const uniform ChunkAllocationInfo info) {
  const uint32_t3 index = (info.chunk_index << ((voxel_tree_header.max_depth - 1) << 1)) + dispatch_id;
  const float3 world_position = GetWorldPositionCenter(index, voxel_tree_header.max_depth - 1);

  const float noise_height = SampleNoise(info.seed, world_position.xz);

  if (world_position.y > noise_height) {
    return;
  }

  if (info.color == true) {
    const bool3 min_edge = (dispatch_id == 0) && (info.chunk_index != 0);
    const bool3 max_edge =
        (dispatch_id == (1 << ((voxel_tree_header.max_depth - 2) * 2)) * 4 - 1) && (info.chunk_index != 3);

    if (any(min_edge || max_edge)) {
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

      for (uint32_t i = 0; i < 26; i++) {
        const bool3 min_edge = (dispatch_id + offset_masks[i] < 0);
        const bool3 max_edge =
            ((dispatch_id + offset_masks[i]) > ((1 << ((voxel_tree_header.max_depth - 2) * 2)) * 4 - 1));
        if (any(min_edge || max_edge)) {
          UpdateLeafNormal(index + offset_masks[i]);
        }
      }
    }

    float3 color;
    if (world_position.y + voxel_size > noise_height) {
      color = GRASS_COLOR;
    } else {
      color = DIRT_COLOR;
    }

    const float dh_dx =
        (SampleNoise(info.seed, world_position.xz + float2(voxel_size, 0)) - noise_height) / voxel_size;
    const float dh_dy =
        (SampleNoise(info.seed, world_position.xz + float2(0, voxel_size)) - noise_height) / voxel_size;

    const float slant = 1.0f - dot(float3(0.0f, 1.0f, 0.0f), normalize(float3(-dh_dy, 1.0f, -dh_dx)));

    if (slant > 0.6f) {
      color = DIRT_COLOR;
    }

    AllocateLeaf(index, color);
  } else {
    Allocate(index, info.lod, info.alloc_leaf == true);
  }
}
