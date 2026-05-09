#include "Include/camera.slang"
#include "Include/common.slang"
#define DENSE_SET_SET_BINDING 4
#include "Include/dense_set.slang"

[[vk::binding(0, 0)]]
RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0, 1)]]
ConstantBuffer<Camera> camera;

[[vk::binding(0, 2)]]
RWTexture2D<float> main_image;

[[vk::binding(0, 3)]]
StructuredBuffer<uint32_t> mesh_triangle_offsets;

[[vk::binding(3, 3)]]
StructuredBuffer<uint32_t> triangle_voxelized_depth_arr[];

static const uint32_t NO_HIT = 0xFFFFFFFF;

struct PayloadData {
  uint32_t triangle_id;
  uint32_t mesh_id;
};

[[shader("raygeneration")]]
void RayGenMain() {
  uint32_t2 image_size;
  main_image.GetDimensions(image_size.x, image_size.y);
  image_size /= PREPASS_SCALE;

  const uint32_t2 dispatch_id = DispatchRaysIndex().xy;

  if (any(dispatch_id >= image_size))
    return;

  const float4x4 inv_view_proj =
      mul(LookAtInverse(camera.position, camera.position + camera.front, camera.up),
          PerspectiveReverseZInverse(1.0f, 2.0f, camera.fov_y, camera.aspect_ratio));

  const float3 ray_origin = camera.position;

  const float2 center_ndc = float2(float(dispatch_id.x + 0.5) / image_size.x * 2.0 - 1.0,
                                   float(dispatch_id.y + 0.5) / image_size.y * -2.0 + 1.0);
  const float4 center_world = mul(inv_view_proj, float4(center_ndc, 1.0, 1.0));
  const float3 ray_dir = normalize(center_world.xyz / center_world.w - ray_origin);

  RayDesc ray_desc;
  ray_desc.TMin = 0.0f;
  ray_desc.TMax = 1e10f;
  ray_desc.Origin = camera.position;
  ray_desc.Direction = ray_dir;

  PayloadData payload;
  payload.triangle_id = NO_HIT;
  payload.mesh_id = NO_HIT;

  TraceRay(acceleration_structure, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, ray_desc, payload);

  if (payload.mesh_id != NO_HIT) {
    const uint32_t triangle_depth = triangle_voxelized_depth_arr[payload.mesh_id][payload.triangle_id];
    const uint32_t global_triangle_id = payload.triangle_id + mesh_triangle_offsets[payload.mesh_id];

    if (triangle_depth == 0) {
      dense_set.Insert(global_triangle_id, global_triangle_id);
    }
  }
}

[[shader("closesthit")]]
void ClosestHitMain(inout PayloadData payload, in BuiltInTriangleIntersectionAttributes attr) {
  payload.triangle_id = PrimitiveIndex();
  payload.mesh_id = InstanceID();
}

[[shader("miss")]]
void MissMain() {}
