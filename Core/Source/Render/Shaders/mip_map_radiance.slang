#include "Include/common.slang"
#define VOXEL_TREE_SET_BINDING 0
#define VOXEL_TREE_READ_WRITE

#include "Include/voxel_tree.slang"

[shader("compute")]
[[numthreads(64, 1, 1)]]
void CMain(const uint32_t dispatch_id: SV_DispatchThreadID, const uniform uint32_t depth) {
  const uint32_t ptr = dispatch_id / 64;
  const uint32_t local_index = dispatch_id % 64;

  if (ptr >= voxel_tree_header.AllocateVoxels(0, depth))
    return;

  const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
  const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

  const BranchNode node = voxel_tree[page_index + voxel_tree_header.GetLevelPageOffset(depth)][page_offset];

  if (((node.child_mask >> local_index) & 0b1) == 0)
    return;

  const uint32_t child_page_index = node.child_ptr >> voxel_tree_header.page_size_exp;
  const uint32_t child_page_offset = node.child_ptr - (child_page_index << voxel_tree_header.page_size_exp);

  const uint32_t child_radiance_ptr =
      voxel_tree[child_page_index + voxel_tree_header.GetLevelPageOffset(depth + 1)]
                [child_page_offset + local_index]
                    .child_radiance_ptr;

  uint32_t child_radiance_page_index;
  uint32_t3 child_radiance_image_pixel_offset;
  GetRadianceImagePixelOffset(child_radiance_ptr, child_radiance_page_index,
                              child_radiance_image_pixel_offset);

  float4 color = 0.0f;
  for (uint32_t i = 0; i < 64; i++) {
    color += radiance_pages[child_radiance_page_index].Load(child_radiance_image_pixel_offset + 1 +
                                                            GetLocalOffset(i));
  }
  color /= 64.0f;

  uint32_t radiance_page_index;
  uint32_t3 radiance_image_pixel_offset;
  GetRadianceImagePixelOffset(node.child_radiance_ptr, radiance_page_index, radiance_image_pixel_offset);

  /*
  radiance_pages[radiance_page_index].Store(radiance_image_pixel_offset + GetLocalOffset(local_index) + 1,
                                            color);
  */
}
