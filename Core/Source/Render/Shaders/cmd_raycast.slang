#define VOXEL_TREE_SET_BINDING 0
#include "Include/voxel_tree.slang"

struct RaycastCmd {
  float3 origin;
  float t_max;
  float3 dir;
  uint32_t _p1;
};

[[vk::binding(0, 1)]]
StructuredBuffer<RaycastCmd> cmds;

[[vk::binding(1, 1)]]
RWStructuredBuffer<RaycastResult> results;

[[numthreads(1, 1, 1)]]
void CMain(const uint32_t dispatch_id: SV_DispatchThreadID) {
  const RaycastCmd cmd = cmds[dispatch_id];
  RaycastResult res;
  Raycast(cmd.origin, cmd.dir, res, cmd.t_max);
  results[dispatch_id] = res;
}
