#include "Core/Render/frame.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
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
    render_context->swapchain.Resize(Window::GetSize());
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
    upload_pass.AddDependency<DeviceResourceType::TransferDst>(
        render_context->indirect_light_hash_set.swapped_data[resource_index].set_buffer);

    upload_pass.AddDependency<DeviceResourceType::TransferSrc>(
        render_context->indirect_light_hash_set.swapped_data[resource_index].header_staging_buffer);
    upload_pass.AddDependency<DeviceResourceType::TransferDst>(
        render_context->indirect_light_hash_set.swapped_data[resource_index].header_buffer);

    cmd.BindSubPass(upload_pass);

    u64 offset = 0;
    memcpy((char *)render_context->frame_staging_buffer[resource_index].host_address + offset, &camera.ubo,
           sizeof(Camera::UBO));
    cmd.UploadBufferToBuffer(render_context->frame_staging_buffer[resource_index],
                             render_context->camera_buffer[resource_index], sizeof(Camera::UBO), offset);

    DeviceHashSetHeader *last_header =
        (DeviceHashSetHeader *)render_context->indirect_light_hash_set.swapped_data[resource_index]
            .header_staging_buffer.host_address;

    if (last_header->insertion_failures != 0) {
      cmd.FillBuffer(
          render_context->indirect_light_hash_set.swapped_data[resource_index].set_buffer,
          render_context->indirect_light_hash_set.swapped_data[resource_index].set_buffer.size,
          DeviceHashSet::EMPTY_KEY);
    }
    last_header->insertion_failures = 0;

    cmd.UploadBufferToBuffer(
        render_context->indirect_light_hash_set.swapped_data[resource_index].header_staging_buffer,
        render_context->indirect_light_hash_set.swapped_data[resource_index].header_buffer,
        render_context->indirect_light_hash_set.swapped_data[resource_index].header_buffer.size);

    offset += sizeof(Camera::UBO);

    cmd.EndDebugPass();
  }

  {
    cmd.BeginDebugPass("beam pass");
    VulkanSubPass<SubPassType::Compute> beam_pass;
    beam_pass.AddDependency<DeviceResourceType::RWStorageImage>(render_context->beam_prepass_image);
    beam_pass.AddDependency<DeviceResourceType::Buffer>(render_context->camera_buffer[resource_index]);
    beam_pass.AddDependency<DeviceResourceType::Buffer>(render_context->camera_buffer[last_resource_index]);

    cmd.BindSubPass(beam_pass);

    cmd.BindPipeline(render_context->beam_prepass_pipeline);
    cmd.BindDescriptors({
        render_context->image_descriptor,
        render_context->camera_descriptor[resource_index],
        render_context->voxel_tree.descriptor,
    });
    cmd.Dispatch(Vec3u32(render_context->beam_prepass_image.GetVec2u32() / 8 + 1, 1));
    cmd.EndDebugPass();
  }

  {
    cmd.BeginDebugPass("indirect pass");
    VulkanSubPass<SubPassType::Compute> indirect_lighting_pass;
    indirect_lighting_pass.AddDependency<DeviceResourceType::RWStorageImage>(render_context->main_image);
    indirect_lighting_pass.AddDependency<DeviceResourceType::Buffer>(
        render_context->camera_buffer[resource_index]);
    indirect_lighting_pass.AddDependency<DeviceResourceType::Buffer>(
        render_context->camera_buffer[last_resource_index]);
    indirect_lighting_pass.AddDependency<DeviceResourceType::Buffer>(
        render_context->directional_light_buffer);
    indirect_lighting_pass.AddDependency<DeviceResourceType::RWBuffer>(
        render_context->indirect_light_hash_set.swapped_data[resource_index].header_buffer);
    indirect_lighting_pass.AddDependency<DeviceResourceType::RWBuffer>(
        render_context->indirect_light_hash_set.swapped_data[resource_index].set_buffer);
    indirect_lighting_pass.AddDependency<DeviceResourceType::Buffer>(
        render_context->indirect_light_hash_set.swapped_data[last_resource_index].set_buffer);

    cmd.BindSubPass(indirect_lighting_pass);

    cmd.BindPipeline(render_context->indirect_lighting_prepass_pipeline);
    cmd.BindDescriptors({
        render_context->image_descriptor,
        render_context->camera_descriptor[resource_index],
        render_context->voxel_tree.descriptor,
        render_context->light_descriptor,
        render_context->indirect_light_hash_set.swapped_data[resource_index].descriptor,
    });
    cmd.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(u32), &render_context->swapchain.frame_number);
    cmd.Dispatch(Vec3u32((render_context->main_image.GetVec2u32() >> INDIRECT_LIGHT_SCALE_EXP) / 8 + 1, 1));
    cmd.EndDebugPass();
  }

  {
    cmd.BeginDebugPass("main pass");
    VulkanSubPass<SubPassType::Compute> main_pass;
    main_pass.AddDependency<DeviceResourceType::RWStorageImage>(render_context->main_image);
    main_pass.AddDependency<DeviceResourceType::Buffer>(render_context->camera_buffer[resource_index]);
    main_pass.AddDependency<DeviceResourceType::Buffer>(render_context->camera_buffer[last_resource_index]);
    main_pass.AddDependency<DeviceResourceType::Buffer>(render_context->directional_light_buffer);
    main_pass.AddDependency<DeviceResourceType::Buffer>(
        render_context->indirect_light_hash_set.swapped_data[resource_index].header_buffer);
    main_pass.AddDependency<DeviceResourceType::Buffer>(
        render_context->indirect_light_hash_set.swapped_data[resource_index].set_buffer);
    main_pass.AddDependency<DeviceResourceType::Buffer>(
        render_context->indirect_light_hash_set.swapped_data[last_resource_index].set_buffer);

    cmd.BindSubPass(main_pass);

    cmd.BindPipeline(render_context->main_pipeline);
    cmd.BindDescriptors({
        render_context->image_descriptor,
        render_context->camera_descriptor[resource_index],
        render_context->voxel_tree.descriptor,
        render_context->light_descriptor,
        render_context->indirect_light_hash_set.swapped_data[resource_index].descriptor,
    });
    cmd.Dispatch(Vec3u32(render_context->main_image.GetVec2u32() / 8 + 1, 1));
    cmd.EndDebugPass();
  }

  {
    cmd.BeginDebugPass("transfer pass");
    VulkanSubPass<SubPassType::Transfer> transer_pass;
    transer_pass.AddDependency<DeviceResourceType::TransferSrc>(render_context->main_image);
    transer_pass.AddDependency<DeviceResourceType::TransferDst>(render_context->swapchain.GetImage());

    transer_pass.AddDependency<DeviceResourceType::TransferSrc>(
        render_context->indirect_light_hash_set.swapped_data[resource_index].header_buffer);
    transer_pass.AddDependency<DeviceResourceType::TransferDst>(
        render_context->indirect_light_hash_set.swapped_data[resource_index].header_staging_buffer);

    cmd.BindSubPass(transer_pass);

    cmd.CopyImageToImage(render_context->main_image, render_context->swapchain.GetImage());

    cmd.UploadBufferToBuffer(
        render_context->indirect_light_hash_set.swapped_data[resource_index].header_buffer,
        render_context->indirect_light_hash_set.swapped_data[resource_index].header_staging_buffer,
        render_context->indirect_light_hash_set.swapped_data[resource_index].header_staging_buffer.size);

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
  render_context->indirect_light_hash_set.Recreate(
      render_context->main_image.height * render_context->main_image.width >> INDIRECT_LIGHT_SCALE_EXP,
      VK_SHADER_STAGE_COMPUTE_BIT);

  render_context->beam_prepass_image.Recreate(extent >> BEAM_PREPASS_SCALE_EXP,
                                              render_context->beam_prepass_image.format,
                                              render_context->beam_prepass_image.usage, /*referenced=*/true);
  render_context->image_descriptor.Update<DeviceResourceType::RWStorageImage>(
      1, &render_context->beam_prepass_image);

  Core::Log("{}", extent.String());
}
} // namespace Core
