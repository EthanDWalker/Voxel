#include "Include/common.slang"
#define VOXEL_TREE_SET_BINDING 0
#define VOXEL_TREE_READ_WRITE
#include "Include/voxel_tree.slang"

[[vk::binding(0, 1)]]
CountedBuffer<DirectionalLight> directional_light_arr;

import glsl;

BranchNode TraverseTree(const uint32_t3 index, const uint32_t depth) {
  BranchNode node;
  uint32_t ptr = GetLocalIndex(index, 0);

  for (uint32_t i = 1; i < depth; i++) {
    const uint32_t level_page_offset = voxel_tree_header.GetLevelPageOffset(i - 1);
    const uint32_t page_index = ptr >> voxel_tree_header.page_size_exp;
    const uint32_t page_offset = ptr - (page_index << voxel_tree_header.page_size_exp);

    const uint32_t local_index = GetLocalIndex(index, i);

    node = voxel_tree[page_index + level_page_offset][page_offset];

    ptr = node.child_ptr + local_index;
  }

  return node;
}

void StoreRadiance(const uint32_t3 index, const float4 color) {
  uint32_t val;

  // clang-format off
  const uint32_t3 offsets[8] = {
    uint32_t3(0, 0, 0),
    uint32_t3(1, 0, 0),
    uint32_t3(0, 1, 0),
    uint32_t3(0, 0, 1),

    uint32_t3(1, 1, 0), 
    uint32_t3(1, 0, 1), 
    uint32_t3(0, 1, 1), 

    uint32_t3(1, 1, 1),
  };
  // clang-format on

  {
    const uint32_t3 local_offset = GetLocalOffset(index, voxel_tree_header.max_depth - 1);

    const bool3 edge = (local_offset == 0 || local_offset == 3);
    const int32_t3 tree_offset = select(edge, select(local_offset == 0, -1, 1), 0);
    const int32_t3 radiance_offset = select(edge, select(local_offset == 0, 4, -4), 0);

    for (uint32_t i = 0; i < 8; i++) {
      if (i != 0 && all(offsets[i] * tree_offset == 0))
        continue;

      const uint32_t radiance_ptr =
          TraverseTree(index + tree_offset * offsets[i], voxel_tree_header.max_depth).child_radiance_ptr;

      uint32_t radiance_page_index;
      uint32_t3 radiance_image_pixel_offset;
      GetRadianceImagePixelOffset(radiance_ptr, radiance_page_index, radiance_image_pixel_offset);

      const uint32_t ret =
          imageAtomicCompSwap(radiance_pages[radiance_page_index],
                              radiance_image_pixel_offset + (local_offset + 1) + radiance_offset * offsets[i],
                              0, PackRGBA8(color));
      if (i == 0) {
        val = ret;
      }
    }
  }

  if (val == 0) {
    for (uint32_t depth = 1; depth < voxel_tree_header.max_depth; depth++) {
      const uint32_t3 local_offset = GetLocalOffset(index, depth - 1);
      const uint32_t level_shift = (voxel_tree_header.max_depth - depth) * 2;
      const uint32_t3 level_index = index >> level_shift;

      const bool3 edge = (local_offset == 0 || local_offset == 3);
      const int32_t3 tree_offset = select(edge, select(local_offset == 0, -1, 1), 0);
      const int32_t3 radiance_offset = select(edge, select(local_offset == 0, 4, -4), 0);

      for (uint32_t i = 0; i < 4; i++) {
        if (i == 0 || any(offsets[i] * tree_offset != 0)) {
          const BranchNode branch_node =
              TraverseTree((level_index + tree_offset * offsets[i]) << level_shift, depth);

          uint32_t radiance_page_index;
          uint32_t3 radiance_image_pixel_offset;
          GetRadianceImagePixelOffset(branch_node.child_radiance_ptr, radiance_page_index,
                                      radiance_image_pixel_offset);

          const float mip_scale = 1 << ((voxel_tree_header.max_depth - depth) * 6);
          const float4 mipped_color = color * (1.0f / mip_scale);

          imageAtomicAdd(radiance_pages[radiance_page_index],
                         radiance_image_pixel_offset + (local_offset + 1) + radiance_offset * offsets[i],
                         PackRGBA8(mipped_color));
        }
      }
    }
  }
}

[shader("compute")]
[[numthreads(8, 8, 1)]]
void CMain(const uint32_t2 dispatch_id: SV_DispatchThreadID) {
  if (any(dispatch_id > grid_n))
    return;

  const float3 ray_origin =
      voxel_size * float3(dispatch_id.x, grid_n, dispatch_id.y) + voxel_tree_header.min_bound;

  const float3 ray_dir = directional_light_arr[0].direction;

  const uint32_t sample_count = 10;

  const float3 sample_offsets[sample_count] = {
    float3(0.0f, 0.0f, 0.0f),

    float3(voxel_size * 0.25f, 0.0f, 0.0f),
    float3(0.0f, 0.0f, voxel_size * 0.25f),
    float3(voxel_size * 0.25f, 0.0f, voxel_size * 0.25f),

    float3(voxel_size * 0.5f, 0.0f, 0.0f),
    float3(0.0f, 0.0f, voxel_size * 0.5f),
    float3(voxel_size * 0.5f, 0.0f, voxel_size * 0.5f),

    float3(voxel_size * 0.75f, 0.0f, 0.0f),
    float3(0.0f, 0.0f, voxel_size * 0.75f),
    float3(voxel_size * 0.75f, 0.0f, voxel_size * 0.75f),
  };

  for (uint32_t i = 0; i < sample_count; i++) {
    RaycastResult res;
    Raycast(ray_origin + sample_offsets[i], ray_dir, res);

    const uint32_t3 local_offset = GetLocalOffset(res.tree_index, voxel_tree_header.max_depth - 1);

    const float4 color = res.hit_color;

    StoreRadiance(res.tree_index, color);
  }
}
