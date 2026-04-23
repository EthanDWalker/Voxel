#include "Core/Render/frame.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/camera.h"
#include "Core/Render/context.h"
#include "Core/Render/sparse_voxel_tree.h"
#include "Core/Render/types.h"
#include "Core/Util/timer.h"
#include "Core/window.h"
#include <cstring>
#include <tracy/Tracy.hpp>

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
    render_context->swapchain.Resize(Window::GetSize());
  }
}

void CalculateRadiance() {
  ZoneScoped;
  SCOPED_TIMER("radiance calculation")
  VulkanContext::Submit([](VulkanCommandBuffer &cmd) {
    cmd.BindPipeline(render_context->calculate_radiance_pipeline);
    cmd.BindDescriptors({render_context->voxel_tree.tree_descriptor, render_context->light_descriptor});
    cmd.Dispatch(Vec3u32(Vec2u32((1 << (render_context->voxel_tree.MAX_VOXLELIZE_DEPTH * 2)) / 8 + 1), 1));
  });
}

void Frame(Camera &camera) {
  ZoneScoped;
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
    VulkanSubPass<SubPassType::Compute> beam_pass;
    beam_pass.AddDependency<DeviceResourceType::RWStorageImage>(render_context->beam_prepass_image);

    cmd.BindSubPass(beam_pass);
    cmd.BindSubPass(render_context->beam_pass);

    cmd.BindPipeline(render_context->beam_prepass_pipeline);
    cmd.BindDescriptors({render_context->image_descriptor, render_context->camera_descriptor,
                         render_context->voxel_tree.tree_descriptor});
    cmd.Dispatch(Vec3u32(render_context->beam_prepass_image.GetVec2u32() / 8 + 1, 1));
  }

  {
    cmd.BindSubPass(render_context->main_draw_pass);

    VulkanSubPass<SubPassType::Compute> main_pass;
    main_pass.AddDependency<DeviceResourceType::RWStorageImage>(render_context->main_image);
    main_pass.AddDependency<DeviceResourceType::StorageImage>(render_context->beam_prepass_image);

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
  ZoneScoped;
  WaitIdle();

  render_context->main_image.Recreate(extent, render_context->main_image.format,
                                      render_context->main_image.usage);
  render_context->image_descriptor.Update<DeviceResourceType::RWStorageImage>(0, &render_context->main_image);

  render_context->beam_prepass_image.Recreate(extent / BEAM_PREPASS_SCALE,
                                              render_context->beam_prepass_image.format,
                                              render_context->beam_prepass_image.usage);
  render_context->image_descriptor.Update<DeviceResourceType::RWStorageImage>(
      1, &render_context->beam_prepass_image);
}
} // namespace Core
