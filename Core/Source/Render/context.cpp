#include "Core/Render/context.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/Vulkan/pipeline.h"
#include "Core/Render/Vulkan/swapchain.h"
#include "Core/Render/camera.h"
#include "Core/Render/frame.h"
#include "Core/Render/types.h"
#include "Core/window.h"
#include <chrono>

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
    pipeline_builder.AddDescriptorLayout(image_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(camera_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(voxel_tree.descriptor_layout);
    pipeline_builder.AddDescriptorLayout(light_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(indirect_light_hash_set.descriptor_layout);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "main.slang");
    PipelineBuildManager::Build(pipeline_builder, main_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptorLayout(image_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(camera_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(voxel_tree.descriptor_layout);
    pipeline_builder.AddDescriptorLayout(light_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(indirect_light_hash_set.descriptor_layout);
    pipeline_builder.AddPushConstantRange(sizeof(u32)); // frame number
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "indirect_lighting.slang");
    PipelineBuildManager::Build(pipeline_builder, indirect_lighting_prepass_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptorLayout(image_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(camera_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(voxel_tree.descriptor_layout);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "beam.slang");
    PipelineBuildManager::Build(pipeline_builder, beam_prepass_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptorLayout(voxel_tree.descriptor_layout);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "cmd_clear_volume.slang");
    pipeline_builder.AddPushConstantRange(sizeof(VoxelVolume));
    PipelineBuildManager::Build(pipeline_builder, clear_volume_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptorLayout(voxel_tree.descriptor_layout);
    pipeline_builder.AddDescriptorLayout(raycast_descriptor_layout);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "cmd_raycast.slang");
    PipelineBuildManager::Build(pipeline_builder, raycast_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Graphic>();
    pipeline_builder.Default();
    pipeline_builder.SetNoDepthTest();
    pipeline_builder.SetCullMode(VK_CULL_MODE_NONE, {});
    pipeline_builder.SetInputTopology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
    pipeline_builder.AddDescriptorLayout(mesh_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(voxel_tree.descriptor_layout);
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
    pipeline_builder.AddDescriptorLayout(mesh_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(voxel_tree.descriptor_layout);
    pipeline_builder.AddPushConstantRange(VK_SHADER_STAGE_ALL_GRAPHICS, sizeof(AllocateInfo));
    pipeline_builder.SetShaders(std::filesystem::path(SHADER_DIR) / "voxelize.slang",
                                std::filesystem::path(SHADER_DIR) / "allocate_child_mask.slang",
                                std::filesystem::path(SHADER_DIR) / "voxelize.slang");
    PipelineBuildManager::Build(pipeline_builder, allocate_child_mask_pipeline);
  }
}

void RenderContext::Create(const Spec &spec) {
  ZoneScoped;

  start_time = std::chrono::high_resolution_clock::now();
  last_frame_time = start_time;

  const Vec2u32 window_size = Window::GetSize();
  current_spec = spec;

  main_image.Create(window_size, VK_FORMAT_B8G8R8A8_UNORM,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    /*referenced=*/true);

  beam_prepass_image.Create(window_size >> BEAM_PREPASS_SCALE_EXP, VK_FORMAT_R32_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

  swapchain.Create(window_size);

  directional_light_buffer.Create(sizeof(DirectionalLight) * spec.max_directional_lights + sizeof(u32),
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  for (u32 i = 0; i < VulkanSwapchain::FRAME_OVERLAP; i++) {
    camera_buffer[i].Create(sizeof(Camera::UBO),
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  }

  indirect_light_hash_set.Create((main_image.height * main_image.width) >> INDIRECT_LIGHT_SCALE_EXP,
                                 VK_SHADER_STAGE_COMPUTE_BIT);

  instance_buffer.Create(sizeof(Instance) * spec.max_instances,
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

  raycast_results_buffer.Create(sizeof(RaycastResult) * spec.max_raycasts,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  raycast_cmds_buffer.Create(sizeof(Raycast) * spec.max_raycasts,
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  raycast_staging_buffer.BuildAddStagingBinding(Max(sizeof(RaycastResult), sizeof(Raycast)) *
                                                spec.max_raycasts);
  raycast_staging_buffer.Build();

  for (u32 i = 0; i < VulkanSwapchain::FRAME_OVERLAP; i++) {
    DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&camera_buffer[i]);
    DescriptorBuilder::Bind<DeviceResourceType::Buffer>(
        &camera_buffer[(i + VulkanSwapchain::FRAME_OVERLAP - 1) % VulkanSwapchain::FRAME_OVERLAP]);
    if (i == 0) {
      DescriptorBuilder::BuildLayout(VK_SHADER_STAGE_COMPUTE_BIT, camera_descriptor_layout);
    }
    DescriptorBuilder::BuildSet(VK_SHADER_STAGE_COMPUTE_BIT, camera_descriptor_layout, camera_descriptor[i]);
    DescriptorBuilder::Reset();
  }

  DescriptorBuilder::Bind<DeviceResourceType::RWStorageImage>(&main_image);
  DescriptorBuilder::Bind<DeviceResourceType::RWStorageImage>(&beam_prepass_image);
  DescriptorBuilder::BuildLayout(VK_SHADER_STAGE_COMPUTE_BIT, image_descriptor_layout);
  DescriptorBuilder::BuildSet(VK_SHADER_STAGE_COMPUTE_BIT, image_descriptor_layout, image_descriptor);
  DescriptorBuilder::Reset();

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&directional_light_buffer);
  DescriptorBuilder::BuildLayout(VK_SHADER_STAGE_COMPUTE_BIT, light_descriptor_layout);
  DescriptorBuilder::BuildSet(VK_SHADER_STAGE_COMPUTE_BIT, light_descriptor_layout, light_descriptor);
  DescriptorBuilder::Reset();

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&raycast_cmds_buffer);
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&raycast_results_buffer);
  DescriptorBuilder::BuildLayout(VK_SHADER_STAGE_COMPUTE_BIT, raycast_descriptor_layout);
  DescriptorBuilder::BuildSet(VK_SHADER_STAGE_COMPUTE_BIT, raycast_descriptor_layout, raycast_descriptor);
  DescriptorBuilder::Reset();

  albedo_sampler.Create(SamplerFilter::Linear, SamplerFilter::Linear);

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, spec.max_meshes);
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, spec.max_meshes);
  DescriptorBuilder::Bind<DeviceResourceType::SampledImage>(nullptr, spec.max_meshes);
  DescriptorBuilder::Bind<DeviceResourceType::Sampler>(&albedo_sampler);
  DescriptorBuilder::BuildLayout(VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT,
                                 mesh_descriptor_layout);
  DescriptorBuilder::BuildSet(VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT,
                              mesh_descriptor_layout, mesh_descriptor);
  DescriptorBuilder::Reset();

  for (u32 i = 0; i < VulkanSwapchain::FRAME_OVERLAP; i++) {
    frame_staging_buffer[i].BuildAddStagingBinding(sizeof(Camera::UBO));
    frame_staging_buffer[i].Build();
  }

  CreatePipelines();
}

RenderContext::~RenderContext() {
  ZoneScoped;
  WaitIdle();
}

} // namespace Core
