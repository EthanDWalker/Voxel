#include "Core/Render/context.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/Vulkan/pipeline.h"
#include "Core/Render/camera.h"
#include "Core/Render/frame.h"
#include "Core/Render/types.h"
#include "Core/window.h"

namespace Core {
RenderContext *render_context = nullptr;

void RenderContext::RecreatePipelines() {
  ZoneScoped;
  vkDeviceWaitIdle(VulkanContext::device);
  PipelineBuildManager::RecreatePipelines();
}

void RenderContext::CreatePipelines() {
  ZoneScoped;
  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptor(image_descriptor);
    pipeline_builder.AddDescriptor(camera_descriptor);
    pipeline_builder.AddDescriptor(voxel_tree.descriptor);
    pipeline_builder.AddDescriptor(light_descriptor);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "main.slang");
    PipelineBuildManager::Build(pipeline_builder, main_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptor(image_descriptor);
    pipeline_builder.AddDescriptor(camera_descriptor);
    pipeline_builder.AddDescriptor(voxel_tree.descriptor);
    pipeline_builder.AddDescriptor(triangle_id_dense_set.descriptor);
    pipeline_builder.AddDescriptor(mesh_descriptor);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "beam.slang");
    PipelineBuildManager::Build(pipeline_builder, beam_prepass_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptor(voxel_tree.descriptor);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "cmd_clear_volume.slang");
    pipeline_builder.AddPushConstantRange(sizeof(VoxelVolume));
    PipelineBuildManager::Build(pipeline_builder, clear_volume_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptor(voxel_tree.descriptor);
    pipeline_builder.AddDescriptor(raycast_descriptor);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "cmd_raycast.slang");
    PipelineBuildManager::Build(pipeline_builder, raycast_pipeline);
  }
  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Graphic>();
    pipeline_builder.Default();
    pipeline_builder.SetNoDepthTest();
    pipeline_builder.SetCullMode(VK_CULL_MODE_NONE, {});
    pipeline_builder.SetInputTopology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
    pipeline_builder.AddDescriptor(mesh_descriptor);
    pipeline_builder.AddDescriptor(voxel_tree.descriptor);
    pipeline_builder.AddDescriptor(triangle_id_dense_set.descriptor);
    pipeline_builder.SetShaders(std::filesystem::path(SHADER_DIR) / "reallocate.slang",
                                std::filesystem::path(SHADER_DIR) / "reallocate.slang",
                                std::filesystem::path(SHADER_DIR) / "reallocate.slang");
    PipelineBuildManager::Build(pipeline_builder, reallocate_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Graphic>();
    pipeline_builder.Default();
    pipeline_builder.SetNoDepthTest();
    pipeline_builder.SetCullMode(VK_CULL_MODE_NONE, {});
    pipeline_builder.SetInputTopology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
    pipeline_builder.AddDescriptor(mesh_descriptor);
    pipeline_builder.AddDescriptor(voxel_tree.descriptor);
    pipeline_builder.AddPushConstantRange(VK_SHADER_STAGE_ALL_GRAPHICS, sizeof(AllocateInfo));
    pipeline_builder.SetShaders(std::filesystem::path(SHADER_DIR) / "voxelize.slang",
                                std::filesystem::path(SHADER_DIR) / "allocate.slang",
                                std::filesystem::path(SHADER_DIR) / "voxelize.slang");
    PipelineBuildManager::Build(pipeline_builder, allocate_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Graphic>();
    pipeline_builder.Default();
    pipeline_builder.SetNoDepthTest();
    pipeline_builder.SetCullMode(VK_CULL_MODE_NONE, {});
    pipeline_builder.SetInputTopology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
    pipeline_builder.AddDescriptor(mesh_descriptor);
    pipeline_builder.AddDescriptor(voxel_tree.descriptor);
    pipeline_builder.AddPushConstantRange(VK_SHADER_STAGE_ALL_GRAPHICS, sizeof(AllocateInfo));
    pipeline_builder.SetShaders(std::filesystem::path(SHADER_DIR) / "voxelize.slang",
                                std::filesystem::path(SHADER_DIR) / "allocate_child_mask.slang",
                                std::filesystem::path(SHADER_DIR) / "voxelize.slang");
    PipelineBuildManager::Build(pipeline_builder, allocate_child_mask_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Raytrace>();
    pipeline_builder.SetShaders(std::filesystem::path(SHADER_DIR) / "triangle_id.slang",
                                std::filesystem::path(SHADER_DIR) / "triangle_id.slang",
                                std::filesystem::path(SHADER_DIR) / "triangle_id.slang");
    pipeline_builder.AddDescriptor(top_level_acceleration_structure_descriptor);
    pipeline_builder.AddDescriptor(camera_descriptor);
    pipeline_builder.AddDescriptor(image_descriptor);
    pipeline_builder.AddDescriptor(mesh_descriptor);
    pipeline_builder.AddDescriptor(triangle_id_dense_set.descriptor);
    pipeline_builder.SetMaxRecursion(1);
    PipelineBuildManager::Build(pipeline_builder, triangle_id_pipeline, triangle_id_shader_binding_table);
  }
}

void RenderContext::Create(const Spec &spec) {
  ZoneScoped;
  const Vec2u32 window_size = Window::GetSize();
  current_spec = spec;

  main_image.Create(window_size, VK_FORMAT_B8G8R8A8_UNORM,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    /*referenced=*/true);

  beam_prepass_image.Create(window_size / PREPASS_SCALE, VK_FORMAT_R32_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                            /*referenced=*/true);

  swapchain.Create(window_size);

  directional_light_buffer.Create(sizeof(DirectionalLight) * spec.max_directional_lights + sizeof(u32),
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  mesh_triangle_offset_buffer.Create(sizeof(u32) * spec.max_meshes,
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  camera_buffer.Create(sizeof(Camera::UBO),
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

  instance_buffer.Create(sizeof(Instance) * spec.max_instances,
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

  raycast_results_buffer.Create(sizeof(RaycastResult) * spec.max_raycasts,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  raycast_cmds_buffer.Create(sizeof(Raycast) * spec.max_raycasts,
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  raycast_staging_buffer.Create(Max(sizeof(RaycastResult), sizeof(Raycast)) * spec.max_raycasts,
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                /*host=*/true);

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&camera_buffer);
  DescriptorBuilder::Build(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR, &camera_descriptor);

  DescriptorBuilder::Bind<DeviceResourceType::RWStorageImage>(&main_image);
  DescriptorBuilder::Bind<DeviceResourceType::RWStorageImage>(&beam_prepass_image);
  DescriptorBuilder::Build(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR, &image_descriptor);

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&directional_light_buffer);
  DescriptorBuilder::Build(VK_SHADER_STAGE_COMPUTE_BIT, &light_descriptor);

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&raycast_cmds_buffer);
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&raycast_results_buffer);
  DescriptorBuilder::Build(VK_SHADER_STAGE_COMPUTE_BIT, &raycast_descriptor);

  top_level_acceleration_structure.CreateTopLevel(instance_buffer, instance_count);

  DescriptorBuilder::Bind<DeviceResourceType::AccelerationStructure>(&top_level_acceleration_structure);
  DescriptorBuilder::Build(VK_SHADER_STAGE_RAYGEN_BIT_KHR, &top_level_acceleration_structure_descriptor);

  albedo_sampler.Create(SamplerFilter::Linear, SamplerFilter::Linear);

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&mesh_triangle_offset_buffer);
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, spec.max_meshes);
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, spec.max_meshes);
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, spec.max_meshes);
  DescriptorBuilder::Bind<DeviceResourceType::SampledImage>(nullptr, spec.max_meshes);
  DescriptorBuilder::Bind<DeviceResourceType::Sampler>(&albedo_sampler);
  DescriptorBuilder::Build(VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT |
                               VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                           &mesh_descriptor);

  frame_staging_buffer.Create(sizeof(Camera::UBO), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, /*host=*/true);

  CreatePipelines();
}

RenderContext::~RenderContext() {
  ZoneScoped;
  WaitIdle();
}

} // namespace Core
