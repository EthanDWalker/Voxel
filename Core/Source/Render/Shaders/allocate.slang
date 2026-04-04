import common;
import voxelize;

[[vk::binding(4, 0)]]
RWVoxelizationData voxelization_data;
[[vk::binding(5, 0)]]
RWStructuredBuffer<uint32_t> new_voxel_ptrs;

[[vk::binding(0, 1)]]
RWStructuredBuffer<uint32_t> voxel_tree[];

[[vk::binding(1, 1)]]
RWStructuredBuffer<uint32_t> far_ptr_arr[];

[[vk::binding(2, 1)]]
RWTreeHeader voxel_tree_header;

struct PushConstants {
  uint32_t size;
  uint32_t depth;
};

[numthreads(64, 1, 1)]
void CMain(const uint32_t dispatch_id: SV_DispatchThreadID, const uniform PushConstants pc) {
  if (dispatch_id >= pc.size) {
    return;
  }

  const uint32_t ptr = new_voxel_ptrs[dispatch_id * 2];
  const uint32_t child_ptr = new_voxel_ptrs[dispatch_id * 2 + 1];

  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

  if (child_ptr > MAX_CHILD_PTR_NO_FAR) {
    const uint32_t far_ptr = voxel_tree_header.AllocateFarPtr();
    const uint32_t far_ptr_page_index = far_ptr >> voxel_tree_header.far_ptr_page_size_exp;
    const uint32_t far_ptr_page_offset =
        far_ptr - (far_ptr_page_index << voxel_tree_header.far_ptr_page_size_exp);
    far_ptr_arr[far_ptr_page_index][far_ptr_page_offset] = child_ptr;
    voxel_tree[page_index][page_offset] = far_ptr | FAR_PTR_BIT;
  } else {
    voxel_tree[page_index][page_offset] = child_ptr;
  }
}
