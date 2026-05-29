#include "Core/Render/frame.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/other_buffer.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/Vulkan/swapchain.h"
#include "Core/Render/camera.h"
#include "Core/Render/context.h"
#include "Core/Render/device_hash_set.h"
#include "Core/Render/sparse_voxel_tree.h"
#include "Core/Render/types.h"
#include "Core/Util/log.h"
#include "Core/window.h"
#include <chrono>
#include <cstring>

namespace Core {
void BeginFrame(bool &resize) {
  FrameMark;
  ZoneScoped;
  render_context->swapchain.AcquireNextImage(resize);
  render_context->swapchain.BeginCommandBuffer();
}

void WaitIdle() {
  ZoneScoped;
  vkDeviceWaitIdle(VulkanContext::device);
}

void EndFrame(bool &resize) {
  ZoneScoped;
  render_context->swapchain.SubmitCommandBuffer();
  render_context->swapchain.Present(resize);
  if (resize) {
    render_context->swapchain.Recreate(Window::GetSize());
  }
}

void Frame(Camera &camera) {
  ZoneScoped;
  camera.Update();

  const auto new_last_frame_time = std::chrono::high_resolution_clock::now();
  const float delta_time =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::duration<f32>(new_last_frame_time - render_context->last_frame_time))
          .count() *
      0.001f * 0.001f;
  render_context->last_frame_time = new_last_frame_time;

  const float acc_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::duration<f32>(new_last_frame_time - render_context->start_time))
                             .count() *
                         0.001f * 0.001f;

  VulkanCommandBuffer &cmd = render_context->swapchain.GetActiveCommandBuffer();

  const u32 resource_index = render_context->swapchain.frame_index;
  const u32 last_resource_index =
      (render_context->swapchain.frame_index + (VulkanSwapchain::FRAME_OVERLAP - 1)) %
      VulkanSwapchain::FRAME_OVERLAP;

  {
    cmd.BeginDebugPass("upload pass");

    VulkanSubPass<SubPassType::Transfer> upload_pass;
    upload_pass.AddDependency<DeviceResourceType::TransferSrc>(
        render_context->frame_staging_buffer[resource_index]);
    upload_pass.AddDependency<DeviceResourceType::TransferDst>(render_context->main_image);
    upload_pass.AddDependency<DeviceResourceType::TransferDst>(render_context->camera_buffer[resource_index]);

    cmd.BindSubPass(upload_pass);

    u64 offset = 0;
    memcpy((char *)render_context->frame_staging_buffer[resource_index].host_address + offset, &camera.ubo,
           sizeof(Camera::UBO));
    cmd.UploadBufferToBuffer(render_context->frame_staging_buffer[resource_index],
                             render_context->camera_buffer[resource_index], sizeof(Camera::UBO), offset);
    offset += sizeof(Camera::UBO);

    cmd.ClearImage(render_context->main_image);
  }

  {
    cmd.BeginDebugPass("main pass");

    VulkanSubPass<SubPassType::Compute> main_pass;
    main_pass.AddDependency<DeviceResourceType::Buffer>(render_context->camera_buffer[resource_index]);
    main_pass.AddDependency<DeviceResourceType::RWStorageImage>(render_context->main_image);
    main_pass.AddDependency<DeviceResourceType::Buffer>(render_context->directional_light_buffer);

    cmd.BindSubPass(main_pass);

    cmd.BindPipeline(render_context->main_pipeline);
    cmd.BindDescriptors({
        render_context->image_descriptor,
        render_context->mesh_descriptor,
        render_context->camera_descriptor[resource_index],
        render_context->light_descriptor,
    });
    cmd.Dispatch(render_context->main_image.GetVec3u32() / 8 + 1);

    cmd.EndDebugPass();
  }

  {
    cmd.BeginDebugPass("transfer pass");
    VulkanSubPass<SubPassType::Transfer> transer_pass;
    transer_pass.AddDependency<DeviceResourceType::TransferSrc>(render_context->main_image);
    transer_pass.AddDependency<DeviceResourceType::TransferDst>(render_context->swapchain.GetImage());

    cmd.BindSubPass(transer_pass);

    cmd.CopyImageToImage(render_context->main_image, render_context->swapchain.GetImage());

    cmd.EndDebugPass();
  }
}

void Resize(Vec2u32 extent) {
  ZoneScoped;
  WaitIdle();

  Core::Log("{}", extent.String());
  render_context->main_image.Recreate(extent, render_context->main_image.format,
                                      render_context->main_image.usage);
  render_context->image_descriptor.Update<DeviceResourceType::RWStorageImage>(0, &render_context->main_image);
}
} // namespace Core
