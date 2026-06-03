#include "Core/Util/terrain.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/other_buffer.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/types.h"
#include "Core/Util/compress.h"
#include "Core/Util/context.h"
#include "Core/Util/timer.h"
#include "Core/Util/types.h"

namespace Core {
void GenerateTerrainObject(const Vec2u32 extent, const Vec2f32 density, const i32 seed,
                           ObjectData &object_data) {
  SCOPED_TIMER("terrain gen time");

  MeshData &mesh_data = object_data.mesh_data_arr.emplace_back();
  mesh_data.name = "terrain mesh";
  const u32 vertex_count = (extent.x * density.x) * (extent.y * density.y);
  const u32 index_count = vertex_count * 6;
  mesh_data.vertex_host_buffer->Create(vertex_count * sizeof(Vertex), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  mesh_data.index_host_buffer->Create(index_count * sizeof(Index), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  util_context->generate_terrain_descriptor.Update<DeviceResourceType::Buffer>(
      0, mesh_data.vertex_host_buffer.get());
  util_context->generate_terrain_descriptor.Update<DeviceResourceType::Buffer>(
      1, mesh_data.index_host_buffer.get());

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    {
      cmd.BeginDebugPass("generate terrain pass");
      VulkanSubPass<SubPassType::Compute> pass;
      pass.AddDependency<DeviceResourceType::RWBuffer>(*mesh_data.vertex_host_buffer);
      pass.AddDependency<DeviceResourceType::RWBuffer>(*mesh_data.index_host_buffer);

      cmd.BindSubPass(pass);

      cmd.BindPipeline(util_context->generate_terrain_pipeline);
      cmd.BindDescriptors({
          util_context->generate_terrain_descriptor,
      });

      GenerateTerrainPushConstants push_constants;
      push_constants.extent = VecTypeCast<u32>(VecTypeCast<f32>(extent) * density);
      push_constants.seed = seed;

      mesh_data.vertex_count = push_constants.extent.x * push_constants.extent.y;
      mesh_data.index_count = mesh_data.vertex_count * 6;
      mesh_data.material_index = 0;

      cmd.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(GenerateTerrainPushConstants), &push_constants);
      cmd.Dispatch(Vec3u32(push_constants.extent / 8 + 1, 1));

      cmd.EndDebugPass();
    }
  });

  Core::Log("terrain object: vertex count {} index count {}", vertex_count, index_count);

  InstanceData &instance_data = object_data.instance_data_arr.emplace_back();
  instance_data.matrix = InstanceMatrix(0.0f, Quat(), Vec3f32(1.0f / density.x, 1.0f, 1.0f / density.y));
  instance_data.mesh_index = 0;

  MaterialData &material_data = object_data.material_data_arr.emplace_back();
  material_data.albedo_extent = 4;
  material_data.initialized = true;
  material_data.name = "terrain material";

  VulkanBuffer<BufferType::StagingBuffer> image_data_staging_buffer = "instance staging buffer";
  image_data_staging_buffer.Create(sizeof(u32) * 4 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  for (u32 x = 0; x < 4; x++) {
    for (u32 y = 0; y < 4; y++) {
      ((u32 *)image_data_staging_buffer.host_address)[x + 4 * y] = PackRGBA8(Vec4f32(0.0f, 1.0f, 0.0f, 1.0f));
    }
  }

  CompressBC1(material_data.albedo_extent, image_data_staging_buffer,
              *material_data.compressed_albedo_data_buffer);
}
} // namespace Core
