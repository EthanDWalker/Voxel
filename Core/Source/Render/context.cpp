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
    pipeline_builder.AddDescriptorLayout(voxelize_descriptor_layout);
    pipeline_builder.AddPushConstantRange(VK_SHADER_STAGE_ALL_GRAPHICS, sizeof(AllocateInfo));
    pipeline_builder.SetShaders(std::filesystem::path(SHADER_DIR) / "allocate.slang",
                                std::filesystem::path(SHADER_DIR) / "allocate.slang",
                                std::filesystem::path(SHADER_DIR) / "allocate.slang");
    PipelineBuildManager::Build(pipeline_builder, allocate_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptorLayout(image_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(mesh_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(camera_descriptor_layout);
    pipeline_builder.AddDescriptorLayout(light_descriptor_layout);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "main.slang");
    PipelineBuildManager::Build(pipeline_builder, main_pipeline);
  }
}

RenderContext::RenderContext(const RenderSpec &spec) {
  ZoneScoped;

  start_time = std::chrono::high_resolution_clock::now();
  last_frame_time = start_time;

  const Vec2u32 window_size = Window::GetSize();
  current_spec = spec;

  main_image.Create(window_size, VK_FORMAT_B8G8R8A8_UNORM,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    /*referenced=*/true);

  swapchain.Create(window_size);

  directional_light_buffer.Create(spec.max_directional_lights,
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  for (u32 i = 0; i < VulkanSwapchain::FRAME_OVERLAP; i++) {
    camera_buffer[i].Create(sizeof(Camera::UBO),
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  }

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

  instance_counted_buffer.Create(
      spec.max_meshes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

  albedo_sampler.Create(SamplerFilter::Linear, SamplerFilter::Linear);

  top_level_acceleration_structure.Create(instance_counted_buffer);

  DescriptorBuilder::Bind<DeviceResourceType::RWStorageImage>(&main_image);
  DescriptorBuilder::BuildLayout(VK_SHADER_STAGE_COMPUTE_BIT, image_descriptor_layout);
  DescriptorBuilder::BuildSet(VK_SHADER_STAGE_COMPUTE_BIT, image_descriptor_layout, image_descriptor);
  DescriptorBuilder::Reset();

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&directional_light_buffer);
  DescriptorBuilder::BuildLayout(VK_SHADER_STAGE_COMPUTE_BIT, light_descriptor_layout);
  DescriptorBuilder::BuildSet(VK_SHADER_STAGE_COMPUTE_BIT, light_descriptor_layout, light_descriptor);
  DescriptorBuilder::Reset();

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, spec.max_meshes);       // voxelize info
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, spec.max_meshes);       // voxel aabb
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, spec.max_meshes);       // vertex
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, spec.max_meshes);       // index
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, spec.max_meshes);       // mesh aabb
  DescriptorBuilder::Bind<DeviceResourceType::SampledImage>(nullptr, spec.max_meshes); // albedo
  DescriptorBuilder::Bind<DeviceResourceType::Sampler>(&albedo_sampler);               // sampler
  DescriptorBuilder::BuildLayout(VK_SHADER_STAGE_ALL_GRAPHICS, voxelize_descriptor_layout);
  DescriptorBuilder::BuildSet(VK_SHADER_STAGE_ALL_GRAPHICS, voxelize_descriptor_layout, voxelize_descriptor);
  DescriptorBuilder::Reset();

  DescriptorBuilder::Bind<DeviceResourceType::AccelerationStructure>(&top_level_acceleration_structure);
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, spec.max_meshes);
  DescriptorBuilder::BuildLayout(VK_SHADER_STAGE_COMPUTE_BIT, mesh_descriptor_layout);
  DescriptorBuilder::BuildSet(VK_SHADER_STAGE_COMPUTE_BIT, mesh_descriptor_layout, mesh_descriptor);
  DescriptorBuilder::Reset();

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
