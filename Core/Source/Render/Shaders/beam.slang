#include "Include/camera.slang"
#include "Include/common.slang"
#include "Include/voxel.slang"

[[vk::binding(1, 0)]]
RWTexture2D<float> beam_image;

[[vk::binding(0, 1)]]
ConstantBuffer<RaytracingAccelerationStructure> acceleration_structure;

[[vk::binding(1, 1)]]
StructuredBuffer<BranchNode> mesh_branch_node_arr[];

[[vk::binding(2, 1)]]
CountedBuffer<AABB> mesh_voxel_aabb_arr_arr[];

[[vk::binding(3, 1)]]
CountedBuffer<AABB> mesh_aabb_arr;

[[vk::binding(0, 2)]]
ConstantBuffer<Camera> camera;

[[vk::binding(0, 3)]]
StructuredBuffer<PackedLeafNode> leaf_node_page_arr[];

[[vk::binding(1, 3)]]
StructuredBuffer<LeafHeader> leaf_header;

float3 GetPixelRayDir(const uint32_t2 pixel_index, const uint32_t2 image_size, const float4x4 inv_view_proj) {
  const float2 pixel_center_ndc = float2(float(pixel_index.x + 0.5f) / image_size.x * 2.0f - 1.0f,
                                         float(pixel_index.y + 0.5f) / image_size.y * -2.0f + 1.0f);
  const float4 pixel_center_world = mul(inv_view_proj, float4(pixel_center_ndc, 1.0f, 1.0f));
  return normalize(pixel_center_world.xyz / pixel_center_world.w - camera.position);
}

bool TraverseVoxelAABB(const float3 hit_position, const float3 ray_dir, const AABB aabb,
                       const BranchNode branch_node, inout float t) {

  const float3 ray_step = step(0.0f, ray_dir);
  const float3 ray_sign = sign(ray_dir);
  const float3 inv_ray_dir = 1.0f / ray_dir;

  uint32_t3 local_index = uint32_t3(FloatDivideBy2ToN(hit_position - aabb.min_bound, VOXEL_SIZE_EXP));

  const float3 t_delta = abs(FloatMultiplyBy2ToN(inv_ray_dir, VOXEL_SIZE_EXP));

  const float3 next_voxel = FloatMultiplyBy2ToN(local_index + ray_step, VOXEL_SIZE_EXP) + aabb.min_bound;
  float3 t_max = (next_voxel - hit_position) * inv_ray_dir;

  while (all(local_index <= 3)) {
    const uint32_t local_child_mask_index = SumComponents(local_index << uint32_t3(0, 2, 4));

    const float t_next = min3(t_max.x, t_max.y, t_max.z);

    const float3 mask = step(t_max, t_next);

    if ((branch_node.child_mask & (uint64_t(1) << local_child_mask_index)) != 0) {
      return true;
    }

    t_max += mask * t_delta;
    t = t_next;
    local_index += int32_t3(ray_sign * mask);
  }

  return false;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void CMain(const uint32_t2 dispatch_id: SV_DispatchThreadID) {
  uint32_t2 image_size;
  beam_image.GetDimensions(image_size.x, image_size.y);

  if (all(image_size <= dispatch_id))
    return;

  const float4x4 inv_view_proj =
      mul(LookAtInverse(camera.position, camera.position + camera.front, camera.up),
          PerspectiveReverseZInverse(1.0f, 2.0f, camera.fov_y, camera.aspect_ratio));

  const float3 ray_dir = GetPixelRayDir(dispatch_id, image_size, inv_view_proj);

  RayDesc ray_desc;
  ray_desc.TMin = 0.0f;
  ray_desc.TMax = 1e6f;
  ray_desc.Origin = camera.position;
  ray_desc.Direction = ray_dir;

  RayQuery ray_query;
  ray_query.TraceRayInline(acceleration_structure, RAY_FLAG_NONE, 0xFF, ray_desc);

  uint32_t i = 0;
  float t = 1e6f;
  bool hit = false;
  float stepped_t = 0.0f;
  LeafNode node;
  while (ray_query.Proceed()) {
    const uint32_t mesh_index = ray_query.CandidateInstanceID();
    if (ray_query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
      const uint32_t primitive_index = ray_query.CandidatePrimitiveIndex();

      const AABB aabb = mesh_voxel_aabb_arr_arr[mesh_index][primitive_index];

      float t_min, t_max;
      if (RayAABBIntersect(ray_query.CandidateObjectRayDirection(), ray_query.CandidateObjectRayOrigin(),
                           aabb.min_bound, aabb.max_bound, t_min, t_max)) {

        const float3 hit_position =
            (t_min + 1e-3f) * ray_query.CandidateObjectRayDirection() + ray_query.CandidateObjectRayOrigin();

        const uint32_t3 voxel_aabb_index =
            uint32_t3(hit_position - mesh_aabb_arr[mesh_index].min_bound) >> VOXEL_AABB_SIZE_EXP;

        AABB voxel_aabb;
        voxel_aabb.min_bound =
            mesh_aabb_arr[mesh_index].min_bound + (voxel_aabb_index << VOXEL_AABB_SIZE_EXP);
        voxel_aabb.max_bound = voxel_aabb.min_bound + VOXEL_AABB_SIZE;

        const uint32_t3 voxel_aabb_max_index =
            uint32_t3(mesh_aabb_arr[mesh_index].max_bound - mesh_aabb_arr[mesh_index].min_bound) >>
            VOXEL_AABB_SIZE_EXP;

        const uint32_t arr_index = voxel_aabb_index.x + (voxel_aabb_index.y * voxel_aabb_max_index.x) +
                                   (voxel_aabb_index.z * voxel_aabb_max_index.x * voxel_aabb_max_index.y);

        const BranchNode branch_node = mesh_branch_node_arr[mesh_index][arr_index];

        float traversed_t = 0.0f;
        if (TraverseVoxelAABB(hit_position, ray_query.CandidateObjectRayDirection(), voxel_aabb, branch_node,
                              traversed_t)) {
          if (t_min + traversed_t < 0.0f)
            continue;

          t = min(t, t_min + traversed_t);
          hit = true;

          ray_query.CommitProceduralPrimitiveHit(t_min + traversed_t);
        }
      }
    }
  }

  const float3 world_position = t * ray_dir + camera.position;

  if (hit) {
    beam_image.Store(dispatch_id, t * 0.99f);
  }
}
