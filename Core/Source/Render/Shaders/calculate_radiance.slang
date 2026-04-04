import common;
import voxelize;

[[vk::binding(0, 1)]]
RWStructuredBuffer<uint32_t> voxel_tree[];

[[vk::binding(1, 1)]]
RWStructuredBuffer<uint32_t> far_ptr_arr[];

[[vk::binding(2, 1)]]
RWTreeHeader voxel_tree_header;

[[vk::binding(0, 3)]]
CountedBuffer<DirectionalLight> directional_light_arr;

struct PushConstants {
  uint32_t voxel_count;
  uint32_t depth;
};

[[numthreads(64, 1, 1)]]
void CMain(const uint32_t dispatch_id: SV_DispatchThreadID, const uniform PushConstants pc) {
  if (dispatch_id >= pc.voxel_count) {
    return;
  }
}

