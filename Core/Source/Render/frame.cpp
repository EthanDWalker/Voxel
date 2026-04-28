#include "Core/Render/frame.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/camera.h"
#include "Core/Render/context.h"
#include "Core/Render/sparse_voxel_tree.h"
#include "Core/Render/types.h"
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

void Frame(Camera &camera) {
  ZoneScoped;
  camera.Update();

  VulkanCommandBuffer &cmd = render_context->swapchain.GetActiveCommandBuffer();

  {
    VulkanSubPass<SubPassType::Transfer> upload_pass;
    upload_pass.AddDependency<DeviceResourceType::TransferDst>(render_context->main_image);

    cmd.BindSubPass(upload_pass);
    cmd.BindSubPass(render_context->upload_pass);

    u64 offset = 0;
    memcpy((char *)render_context->frame_staging_buffer.address + offset, &camera.ubo, sizeof(Camera::UBO));

    cmd.UploadBufferToBuffer(render_context->frame_staging_buffer, render_context->camera_buffer,
                             sizeof(Camera::UBO), offset);

    cmd.ClearImage(render_context->main_image);

    offset += sizeof(Camera::UBO);
  }

  {
    cmd.BindPipeline(render_context->clear_volume_pipeline);
    cmd.BindDescriptors({render_context->voxel_tree.tree_descriptor});

    for (u32 i = 0; i < render_context->clear_volume_cmds.size(); i++) {
      cmd.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(VoxelVolume),
                        &render_context->clear_volume_cmds[i]);
      cmd.Dispatch(VecTypeCast<u32>(
          Ceil((render_context->clear_volume_cmds[i].max - render_context->clear_volume_cmds[i].min) /
               render_context->voxel_tree.VOXEL_SIZE)));
      cmd.ClearPushConstants();
    }

    render_context->clear_volume_cmds.clear();
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
                                      render_context->main_image.usage, /*referenced=*/true);
  render_context->image_descriptor.Update<DeviceResourceType::RWStorageImage>(0, &render_context->main_image);

  render_context->beam_prepass_image.Recreate(extent / BEAM_PREPASS_SCALE,
                                              render_context->beam_prepass_image.format,
                                              render_context->beam_prepass_image.usage);
  render_context->image_descriptor.Update<DeviceResourceType::RWStorageImage>(
      1, &render_context->beam_prepass_image);
}
} // namespace Core
