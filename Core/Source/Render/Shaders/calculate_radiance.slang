import common;
import voxelize;

[[vk::binding(0, 0)]]
RWStructuredBuffer<BranchNode> voxel_tree[];

[[vk::binding(1, 0)]]
RWStructuredBuffer<LeafNode> voxel_tree_leaf[];

[[vk::binding(2, 0)]]
RWTreeHeader voxel_tree_header;

struct PushConstants {
  uint32_t dispatch_count;
  uint32_t level;
};

[numthreads(64, 1, 1)]
void CMain(const uint32_t dispatch_id: SV_DispatchThreadID, const uniform PushConstants pc) {
  if (dispatch_id >= pc.dispatch_count) {
    return;
  }

  const uint32_t page_index = dispatch_id >> voxel_tree_header.page_size_exp;
  const uint32_t page_offset = dispatch_id - (page_index << voxel_tree_header.page_size_exp);

  BranchNode *node = &voxel_tree[page_index + voxel_tree_header.GetLevelPageOffset(pc.level)][page_offset];

  const uint32_t child_page_index = node.child_ptr >> voxel_tree_header.page_size_exp;
  const uint32_t child_page_offset = node.child_ptr - (child_page_index << voxel_tree_header.page_size_exp);

  uint32_t4 acc = 0;
  if (pc.level == voxel_tree_header.max_depth - 2) {
    for (uint32_t i = 0; i < 64; i++) {
      acc += (uint32_t4(voxel_tree_leaf[child_page_index][child_page_offset + i].color) >>
              uint32_t4(24, 16, 8, 0)) &
             0xFF;
    }
  } else {
    for (uint32_t i = 0; i < 64; i++) {
      acc += (uint32_t4(voxel_tree[child_page_index + voxel_tree_header.GetLevelPageOffset(pc.level + 1)]
                                  [child_page_offset + i]
                                      .radiance) >>
              uint32_t4(24, 16, 8, 0)) &
             0xFF;
    }
  }
  acc = acc >> 6;

  node.radiance = (acc.r << 24) | (acc.g << 16) | (acc.b << 8) | (acc.a << 0);
}
