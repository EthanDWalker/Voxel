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
      LastIndex(render_context->swapchain.frame_index, VulkanSwapchain::FRAME_OVERLAP);

  {
    cmd.BeginDebugPass("upload pass");

    VulkanSubPass<SubPassType::Transfer> pass;
    pass.AddDependency<DeviceResourceType::TransferSrc>(render_context->frame_staging_buffer[resource_index]);
    pass.AddDependency<DeviceResourceType::TransferSrc>(
        render_context->light_hash_set.swapped_data[resource_index].header_staging_buffer);

    pass.AddDependency<DeviceResourceType::TransferDst>(render_context->camera_buffer[resource_index]);
    pass.AddDependency<DeviceResourceType::TransferDst>(
        render_context->light_hash_set.swapped_data[resource_index].key_buffer);
    pass.AddDependency<DeviceResourceType::TransferDst>(
        render_context->light_hash_set.swapped_data[resource_index].diffuse_data_buffer);
    pass.AddDependency<DeviceResourceType::TransferDst>(
        render_context->light_hash_set.swapped_data[resource_index].specular_data_buffer);
    pass.AddDependency<DeviceResourceType::TransferDst>(
        render_context->light_hash_set.swapped_data[resource_index].header_buffer);

    pass.AddDependency<DeviceResourceType::TransferDst>(render_context->main_image);

    pass.AddDependency<DeviceResourceType::TransferDst>(
        render_context->frame_luminance_data_buffer[resource_index]);

    cmd.BindSubPass(pass);

    u64 offset = 0;
    memcpy((char *)render_context->frame_staging_buffer[resource_index].host_address + offset, &camera.ubo,
           sizeof(Camera::UBO));
    cmd.UploadBufferToBuffer(render_context->frame_staging_buffer[resource_index],
                             render_context->camera_buffer[resource_index], sizeof(Camera::UBO), offset);
    offset += sizeof(Camera::UBO);

    cmd.FillBuffer(render_context->frame_luminance_data_buffer[resource_index],
                   render_context->frame_luminance_data_buffer[resource_index].size, 0);

    cmd.FillBuffer(render_context->light_hash_set.swapped_data[resource_index].diffuse_data_buffer,
                   render_context->light_hash_set.swapped_data[resource_index].diffuse_data_buffer.size, 0);
    cmd.FillBuffer(render_context->light_hash_set.swapped_data[resource_index].specular_data_buffer,
                   render_context->light_hash_set.swapped_data[resource_index].specular_data_buffer.size, 0);
    cmd.FillBuffer(render_context->light_hash_set.swapped_data[resource_index].key_buffer,
                   render_context->light_hash_set.swapped_data[resource_index].key_buffer.size,
                   DeviceHashSet::EMPTY_KEY);

    cmd.EndDebugPass();
  }

  {
    cmd.BeginDebugPass("beam pass");
    VulkanSubPass<SubPassType::Compute> beam_pass;
    beam_pass.AddDependency<DeviceResourceType::RWStorageImage>(render_context->beam_prepass_image);
    beam_pass.AddDependency<DeviceResourceType::Buffer>(render_context->camera_buffer[resource_index]);

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
    cmd.BeginDebugPass("lighting pass");
    VulkanSubPass<SubPassType::Compute> pass;
    pass.AddDependency<DeviceResourceType::RWStorageImage>(render_context->main_image);
    pass.AddDependency<DeviceResourceType::Buffer>(render_context->camera_buffer[resource_index]);

    pass.AddDependency<DeviceResourceType::Buffer>(
        render_context->light_hash_set.swapped_data[resource_index].header_buffer);
    pass.AddDependency<DeviceResourceType::RWBuffer>(
        render_context->light_hash_set.swapped_data[resource_index].key_buffer);
    pass.AddDependency<DeviceResourceType::Buffer>(
        render_context->light_hash_set.swapped_data[last_resource_index].key_buffer);
    pass.AddDependency<DeviceResourceType::RWBuffer>(
        render_context->light_hash_set.swapped_data[resource_index].diffuse_data_buffer);
    pass.AddDependency<DeviceResourceType::Buffer>(
        render_context->light_hash_set.swapped_data[last_resource_index].diffuse_data_buffer);
    pass.AddDependency<DeviceResourceType::RWBuffer>(
        render_context->light_hash_set.swapped_data[resource_index].specular_data_buffer);
    pass.AddDependency<DeviceResourceType::Buffer>(
        render_context->light_hash_set.swapped_data[last_resource_index].specular_data_buffer);
    pass.AddDependency<DeviceResourceType::Buffer>(render_context->emissive_quad_buffer);

    cmd.BindSubPass(pass);

    cmd.BindPipeline(render_context->lighting_pipeline);
    cmd.BindDescriptors({
        render_context->image_descriptor,
        render_context->camera_descriptor[resource_index],
        render_context->voxel_tree.descriptor,
        render_context->light_hash_set.swapped_data[resource_index].descriptor,
        render_context->emissive_descriptor,
    });
    LightingPushConstants push_constants{};
    push_constants.diffuse_alpha = render_context->diffuse_light_alpha;
    push_constants.specular_alpha = render_context->specular_light_alpha;
    push_constants.frame_number = render_context->swapchain.frame_number;
    cmd.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(push_constants), &push_constants);
    cmd.Dispatch(Vec3u32((render_context->main_image.GetVec2u32() >> INDIRECT_LIGHT_SCALE_EXP) / 8 + 1, 1));
    cmd.EndDebugPass();
  }

  {
    cmd.BeginDebugPass("main pass");
    VulkanSubPass<SubPassType::Compute> main_pass;
    main_pass.AddDependency<DeviceResourceType::RWStorageImage>(render_context->main_image);
    main_pass.AddDependency<DeviceResourceType::Buffer>(render_context->camera_buffer[resource_index]);

    main_pass.AddDependency<DeviceResourceType::Buffer>(
        render_context->light_hash_set.swapped_data[resource_index].header_buffer);
    main_pass.AddDependency<DeviceResourceType::Buffer>(
        render_context->light_hash_set.swapped_data[resource_index].key_buffer);
    main_pass.AddDependency<DeviceResourceType::Buffer>(
        render_context->light_hash_set.swapped_data[resource_index].diffuse_data_buffer);
    main_pass.AddDependency<DeviceResourceType::Buffer>(
        render_context->light_hash_set.swapped_data[resource_index].specular_data_buffer);

    main_pass.AddDependency<DeviceResourceType::RWBuffer>(
        render_context->frame_luminance_data_buffer[resource_index]);

    cmd.BindSubPass(main_pass);

    cmd.BindPipeline(render_context->main_pipeline);
    cmd.BindDescriptors({
        render_context->image_descriptor,
        render_context->camera_descriptor[resource_index],
        render_context->voxel_tree.descriptor,
        render_context->light_hash_set.swapped_data[resource_index].descriptor,
        render_context->frame_luminance_descriptor[resource_index],
    });
    cmd.Dispatch(Vec3u32(render_context->main_image.GetVec2u32() / 8 + 1, 1));
    cmd.EndDebugPass();
  }

  {
    cmd.BeginDebugPass("tone map pass");
    VulkanSubPass<SubPassType::Compute> pass;
    pass.AddDependency<DeviceResourceType::RWStorageImage>(render_context->swapchain.GetImage());
    pass.AddDependency<DeviceResourceType::StorageImage>(render_context->main_image);
    pass.AddDependency<DeviceResourceType::Buffer>(
        render_context->frame_luminance_data_buffer[last_resource_index]);
    pass.AddDependency<DeviceResourceType::RWBuffer>(
        render_context->frame_luminance_data_buffer[resource_index]);

    cmd.BindSubPass(pass);
    cmd.BindPipeline(render_context->tone_map_pipeline);
    ToneMapPushConstants constants{};
    constants.delta_time = delta_time;
    cmd.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(ToneMapPushConstants), &constants);
    cmd.BindDescriptors({
        render_context->image_descriptor,
        render_context->swapchain_descriptor[resource_index],
        render_context->frame_luminance_descriptor[resource_index],
    });

    cmd.Dispatch(Vec3u32(render_context->main_image.GetVec2u32() / 8 + 1, 1));
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

  render_context->beam_prepass_image.Recreate(extent >> BEAM_PREPASS_SCALE_EXP,
                                              render_context->beam_prepass_image.format,
                                              render_context->beam_prepass_image.usage, /*referenced=*/true);
  render_context->image_descriptor.Update<DeviceResourceType::RWStorageImage>(
      1, &render_context->beam_prepass_image);

  for (u32 i = 0; i < VulkanSwapchain::FRAME_OVERLAP; i++) {
    render_context->swapchain_descriptor[i].Update<DeviceResourceType::RWStorageImage>(
        0, &render_context->swapchain.images[i]);
  }

  Core::Log("{}", extent.String());
}
} // namespace Core
