#include "Core/Render/frame.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/camera.h"
#include "Core/Render/context.h"
#include "Core/Render/types.h"
#include "Core/window.h"
#include <cstring>

namespace Core {
VulkanCommandBuffer &BeginFrame(bool &resize) {
  render_context->swapchain.AcquireNextImage(resize);
  return render_context->swapchain.BeginCommandBuffer();
}

void WaitIdle() { vkDeviceWaitIdle(VulkanContext::device); }

void EndFrame(bool &resize) {
  render_context->swapchain.SubmitCommandBuffer();
  render_context->swapchain.Present(resize);
  if (resize) {
    render_context->swapchain.Resize(Window::GetSize());
  }
}

void Frame(Camera &camera) {
  camera.Update();

  VulkanCommandBuffer &cmd = render_context->swapchain.GetActiveCommandBuffer();

  {
    cmd.BindSubPass(render_context->upload_pass);

    u64 offset = 0;
    memcpy((char *)render_context->frame_staging_buffer.address + offset, &camera.ubo, sizeof(Camera::UBO));

    cmd.UploadBufferToBuffer(render_context->frame_staging_buffer, render_context->camera_buffer,
                             sizeof(Camera::UBO), offset);

    offset += sizeof(Camera::UBO);
  }

  {
    cmd.BindSubPass(render_context->main_draw_pass);

    VulkanSubPass<SubPassType::Compute> main_pass;
    main_pass.AddDependency<DeviceResourceType::RWStorageImage>(render_context->main_image);

    cmd.BindSubPass(main_pass);

    cmd.BindPipeline(render_context->main_pipeline);
    cmd.BindDescriptors({render_context->image_descriptor, render_context->camera_descriptor,
                         render_context->voxel_tree.tree_descriptor, render_context->light_descriptor});
    cmd.Dispatch(Vec3u32(render_context->main_image.GetVec2u32() / 8 + 1, 1));
  }

  {
    VulkanSubPass<SubPassType::Transfer> transer_pass;
    transer_pass.AddDependency<DeviceResourceType::TransferSrc>(render_context->main_image);
    transer_pass.AddDependency<DeviceResourceType::TransferDst>(render_context->swapchain.GetImage());

    cmd.BindSubPass(transer_pass);

    cmd.CopyImageToImage(render_context->main_image, render_context->swapchain.GetImage());
  }
}

void Resize(Vec2u32 extent) {
  WaitIdle();
  render_context->main_image.Recreate(extent, render_context->main_image.format,
                                      render_context->main_image.usage);
  render_context->image_descriptor.Update<DeviceResourceType::RWStorageImage>(0, &render_context->main_image);
}
} // namespace Core
