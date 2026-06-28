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
std::unique_ptr<RenderContext> render_context;

void RenderContext::RecreatePipelines() {
  ZoneScoped;
  vkDeviceWaitIdle(VulkanContext::device);
  PipelineBuildManager::RecreatePipelines();
}

void RenderContext::CreatePipelines() {
  ZoneScoped;
  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Graphic>();
    pipeline_builder.Default();
    pipeline_builder.SetNoDepthTest();
    pipeline_builder.SetCullMode(VK_CULL_MODE_NONE, {});
    pipeline_builder.SetInputTopology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
    pipeline_builder.AddDescriptorLayout(emissive_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(camera_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(image_descriptor_layout);
    pipeline_builder.SetShaders(std::filesystem::path(SHADER_DIR) / "debug_quad.slang",
                                std::filesystem::path(SHADER_DIR) / "debug_quad.slang",
                                std::filesystem::path(SHADER_DIR) / "debug_quad.slang");
    PipelineBuildManager::Build(pipeline_builder, debug_quad_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptorLayout(image_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(camera_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(voxel_tree.descriptor_layout);
    pipeline_builder.AddDescriptorLayout(light_hash_set.descriptor_layout);
    pipeline_builder.AddDescriptorLayout(frame_luminance_descriptor_layout);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "main.slang");
    PipelineBuildManager::Build(pipeline_builder, main_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptorLayout(image_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(camera_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(voxel_tree.descriptor_layout);
    pipeline_builder.AddDescriptorLayout(light_hash_set.descriptor_layout);
    pipeline_builder.AddDescriptorLayout(emissive_descriptor_layout);
    pipeline_builder.AddPushConstantRange(sizeof(LightingPushConstants));
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "lighting.slang");
    PipelineBuildManager::Build(pipeline_builder, lighting_pipeline);
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
    pipeline_builder.AddDescriptorLayout(emissive_descriptor_layout);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "cmd_fill_volume.slang");
    pipeline_builder.AddPushConstantRange(sizeof(CmdVoxelVolumeFillPushConstants));
    PipelineBuildManager::Build(pipeline_builder, fill_volume_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptorLayout(voxel_tree.descriptor_layout);
    pipeline_builder.AddDescriptorLayout(raycast_descriptor_layout);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "cmd_raycast.slang");
    PipelineBuildManager::Build(pipeline_builder, raycast_cmd_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptorLayout(image_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(swapchain_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(frame_luminance_descriptor_layout);
    pipeline_builder.AddPushConstantRange(sizeof(ToneMapPushConstants));
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "tone_map.slang");
    PipelineBuildManager::Build(pipeline_builder, tone_map_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptorLayout(voxel_tree.descriptor_layout);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "allocate_chunk.slang");
    pipeline_builder.AddPushConstantRange(sizeof(ChunkAllocationInfo));
    PipelineBuildManager::Build(pipeline_builder, chunk_allocate_pipeline);
  }
}

RenderContext::RenderContext(const RenderSpec &spec) {
  ZoneScoped;

  start_time = std::chrono::high_resolution_clock::now();
  last_frame_time = start_time;

  const Vec2u32 window_size = Window::GetSize();
  current_spec = spec;

  main_image.Create(window_size, VK_FORMAT_R16G16B16A16_UNORM,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    /*referenced=*/true);

  beam_prepass_image.Create(window_size >> BEAM_PREPASS_SCALE_EXP, VK_FORMAT_R32_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

  swapchain.Create(window_size);

  for (u32 i = 0; i < VulkanSwapchain::FRAME_OVERLAP; i++) {
    camera_buffer[i].Create(sizeof(Camera::UBO),
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  }

  light_hash_set.Create(((main_image.height * main_image.width) >> INDIRECT_LIGHT_SCALE_EXP) * 3,
                        VK_SHADER_STAGE_COMPUTE_BIT);

  raycast_results_buffer.Create(sizeof(RaycastResult) * spec.max_raycasts,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  raycast_cmds_buffer.Create(sizeof(Raycast) * spec.max_raycasts,
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  raycast_staging_buffer.Create(Max(sizeof(RaycastResult), sizeof(Raycast)) * spec.max_raycasts);

  for (u32 i = 0; i < VulkanSwapchain::FRAME_OVERLAP; i++) {
    DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&camera_buffer[i]);
    DescriptorBuilder::Bind<DeviceResourceType::Buffer>(
        &camera_buffer[(i + VulkanSwapchain::FRAME_OVERLAP - 1) % VulkanSwapchain::FRAME_OVERLAP]);
    if (i == 0) {
      DescriptorBuilder::BuildLayout(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_GEOMETRY_BIT,
                                     camera_descriptor_layout);
    }
    DescriptorBuilder::BuildSet(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_GEOMETRY_BIT,
                                camera_descriptor_layout, camera_descriptor[i]);
    DescriptorBuilder::Reset();
  }

  emissive_quad_buffer.Create(10'000, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

  for (u32 i = 0; i < VulkanSwapchain::FRAME_OVERLAP; i++) {
    frame_luminance_data_buffer[i].Create(1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  }

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&emissive_quad_buffer);
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr);
  DescriptorBuilder::BuildLayout(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_GEOMETRY_BIT,
                                 emissive_descriptor_layout);
  DescriptorBuilder::BuildSet(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_GEOMETRY_BIT,
                              emissive_descriptor_layout, emissive_descriptor);
  DescriptorBuilder::Reset();

  DescriptorBuilder::Bind<DeviceResourceType::RWStorageImage>(&main_image);
  DescriptorBuilder::Bind<DeviceResourceType::RWStorageImage>(&beam_prepass_image);
  DescriptorBuilder::BuildLayout(VK_SHADER_STAGE_COMPUTE_BIT, image_descriptor_layout);
  DescriptorBuilder::BuildSet(VK_SHADER_STAGE_COMPUTE_BIT, image_descriptor_layout, image_descriptor);
  DescriptorBuilder::Reset();

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&raycast_cmds_buffer);
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&raycast_results_buffer);
  DescriptorBuilder::BuildLayout(VK_SHADER_STAGE_COMPUTE_BIT, raycast_descriptor_layout);
  DescriptorBuilder::BuildSet(VK_SHADER_STAGE_COMPUTE_BIT, raycast_descriptor_layout, raycast_descriptor);
  DescriptorBuilder::Reset();

  albedo_sampler.Create(SamplerFilter::Linear, SamplerFilter::Linear);

  for (u32 i = 0; i < VulkanSwapchain::FRAME_OVERLAP; i++) {
    DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&frame_luminance_data_buffer[i]);
    DescriptorBuilder::Bind<DeviceResourceType::Buffer>(
        &frame_luminance_data_buffer[LastIndex(i, VulkanSwapchain::FRAME_OVERLAP)]);
    if (i == 0) {
      DescriptorBuilder::BuildLayout(VK_SHADER_STAGE_COMPUTE_BIT, frame_luminance_descriptor_layout);
    }
    DescriptorBuilder::BuildSet(VK_SHADER_STAGE_COMPUTE_BIT, frame_luminance_descriptor_layout,
                                frame_luminance_descriptor[i]);
    DescriptorBuilder::Reset();
  }

  for (u32 i = 0; i < VulkanSwapchain::FRAME_OVERLAP; i++) {
    DescriptorBuilder::Bind<DeviceResourceType::RWStorageImage>(&swapchain.images[i]);
    if (i == 0) {
      DescriptorBuilder::BuildLayout(VK_SHADER_STAGE_COMPUTE_BIT, swapchain_descriptor_layout);
    }
    DescriptorBuilder::BuildSet(VK_SHADER_STAGE_COMPUTE_BIT, swapchain_descriptor_layout,
                                swapchain_descriptor[i]);
    DescriptorBuilder::Reset();
  }

  for (u32 i = 0; i < VulkanSwapchain::FRAME_OVERLAP; i++) {
    frame_staging_buffer[i].Create(camera_buffer[i].size);
  }

  CreatePipelines();
}

RenderContext::~RenderContext() {
  ZoneScoped;
  WaitIdle();
}

} // namespace Core
