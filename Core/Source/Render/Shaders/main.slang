#include "Include/camera.slang"
#include "Include/common.slang"

[[vk::binding(0, 0)]]
RWTexture2D<float4> image;

[[vk::binding(0, 1)]]
ConstantBuffer<RaytracingAccelerationStructure> acceleration_structure;

struct BranchNode {
  uint64_t child_mask;
};

[[vk::binding(1, 1)]]
StructuredBuffer<BranchNode> mesh_branch_node_arr;

[[vk::binding(0, 2)]]
ConstantBuffer<Camera> camera;

[[vk::binding(0, 3)]]
CountedBuffer<DirectionalLight> directional_light_arr;

float3 GetPixelRayDir(const uint32_t2 pixel_index, const uint32_t2 image_size, const float4x4 inv_view_proj) {
  const float2 pixel_center_ndc = float2(float(pixel_index.x + 0.5f) / image_size.x * 2.0f - 1.0f,
                                         float(pixel_index.y + 0.5f) / image_size.y * -2.0f + 1.0f);
  const float4 pixel_center_world = mul(inv_view_proj, float4(pixel_center_ndc, 1.0f, 1.0f));
  return normalize(pixel_center_world.xyz / pixel_center_world.w - camera.position);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void CMain(const uint32_t2 dispatch_id: SV_DispatchThreadID) {
  uint32_t2 image_size;
  image.GetDimensions(image_size.x, image_size.y);

  if (all(image_size <= dispatch_id))
    return;

  const float4x4 inv_view_proj =
      mul(LookAtInverse(camera.position, camera.position + camera.front, camera.up),
          PerspectiveReverseZInverse(1.0f, 2.0f, camera.fov_y, camera.aspect_ratio));

  const float3 ray_dir = GetPixelRayDir(dispatch_id, image_size, inv_view_proj);

  RayDesc ray;
  ray.TMin = 0.001f;
  ray.TMax = 1e6f;
  ray.Origin = camera.position;
  ray.Direction = ray_dir;

  RayQuery ray_query;

  ray_query.TraceRayInline(acceleration_structure, RAY_FLAG_NONE, 0xFF, ray);

  while (ray_query.Proceed()) {
    if (ray_query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
      if (ray_query.CandidateInstanceID() != 0)
        ray_query.CommitProceduralPrimitiveHit(0.0f);
    }
  }

  if (ray_query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
    image.Store(dispatch_id, float4(float3(0.0f), 1.0f));
  } else {
    image.Store(dispatch_id, float4(SampleEnvironmentColor(ray_dir, directional_light_arr[0]), 1.0f));
  }
}
