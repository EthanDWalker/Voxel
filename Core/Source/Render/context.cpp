#include "Core/Render/context.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/Vulkan/pipeline.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/camera.h"
#include "Core/Render/frame.h"
#include "Core/Render/types.h"
#include "Core/window.h"

namespace Core {
RenderContext *render_context = nullptr;

void RenderContext::RecreatePipelines() {
  vkDeviceWaitIdle(VulkanContext::device);
  PipelineBuildManager::RecreatePipelines();
}

void RenderContext::CreatePipelines() {
  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptor(image_descriptor);
    pipeline_builder.AddDescriptor(camera_descriptor);
    pipeline_builder.AddDescriptor(voxel_tree.tree_descriptor);
    pipeline_builder.AddDescriptor(light_descriptor);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "main.slang");
    PipelineBuildManager::Build(pipeline_builder, main_pipeline);
  }
}

void RenderContext::Create(const Spec &spec) {
  const Vec2u32 window_size = Window::GetSize();
  current_spec = spec;

  main_image.Create(window_size, VK_FORMAT_B8G8R8A8_UNORM,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

  swapchain.Create(window_size);

  directional_light_buffer.Create(sizeof(DirectionalLight) * spec.max_directional_lights + sizeof(u32),
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  camera_buffer.Create(sizeof(Camera::UBO),
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&camera_buffer);
  DescriptorBuilder::Build(VK_SHADER_STAGE_COMPUTE_BIT, &camera_descriptor);

  DescriptorBuilder::Bind<DeviceResourceType::RWStorageImage>(&main_image);
  DescriptorBuilder::Build(VK_SHADER_STAGE_COMPUTE_BIT, &image_descriptor);

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&directional_light_buffer);
  DescriptorBuilder::Build(VK_SHADER_STAGE_COMPUTE_BIT, &light_descriptor);

  frame_staging_buffer.Create(sizeof(Camera::UBO), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, /*host=*/true);

  CreatePipelines();

  upload_pass.AddDependency<DeviceResourceType::TransferSrc>(frame_staging_buffer);
  upload_pass.AddDependency<DeviceResourceType::TransferDst>(camera_buffer);

  main_draw_pass.AddDependency<DeviceResourceType::Buffer>(camera_buffer);
}

RenderContext::~RenderContext() { WaitIdle(); }

} // namespace Core
