#include "Core/Render/edit.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/context.h"
#include "Core/Render/types.h"
#include <cstring>
#include <mutex>

namespace Core {
u32 AddDirectionalLight(const DirectionalLight &dir_light) {
  ZoneScoped;
  VulkanBuffer staging_buffer;
  staging_buffer.Create(sizeof(DirectionalLight), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, /*host=*/true);
  memcpy(staging_buffer.address, &dir_light, sizeof(DirectionalLight));

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    cmd.UploadBufferToBuffer(
        staging_buffer, render_context->directional_light_buffer, sizeof(DirectionalLight), 0,
        sizeof(DirectionalLight) * render_context->directional_light_count + sizeof(u32));

    cmd.FillBuffer(render_context->directional_light_buffer, sizeof(u32),
                   render_context->directional_light_count + 1);
  });

  return render_context->directional_light_count++;
}

void ClearVolume(const VoxelVolume &volume) {
  ZoneScoped;
  std::lock_guard<std::mutex> lock = std::lock_guard(render_context->clear_volume_cmd_mutex);
  render_context->clear_volume_cmds.emplace_back(volume);
}

void QueueRaycast(const Raycast &raycast, const std::function<void(const RaycastResult &result)> &&callback) {
  ZoneScoped;
  std::lock_guard<std::mutex> lock = std::lock_guard(render_context->raycast_mutex);
  render_context->raycast_cmds.emplace_back(raycast);
  render_context->raycast_callbacks.emplace_back(callback);
}

void FlushRaycasts() {
  ZoneScoped;

  std::lock_guard<std::mutex> lock = std::lock_guard(render_context->raycast_mutex);
  VulkanContext::Submit([](VulkanCommandBuffer &cmd) {
    {
      VulkanSubPass<SubPassType::Transfer> copy_pass;
      copy_pass.AddDependency<DeviceResourceType::TransferSrc>(render_context->raycast_staging_buffer);
      copy_pass.AddDependency<DeviceResourceType::TransferDst>(render_context->raycast_cmds_buffer);

      cmd.BindSubPass(copy_pass);

      memcpy(render_context->raycast_staging_buffer.address, render_context->raycast_cmds.data(),
             sizeof(Raycast) * render_context->raycast_cmds.size());
      cmd.UploadBufferToBuffer(render_context->raycast_staging_buffer, render_context->raycast_cmds_buffer,
                               sizeof(Raycast) * render_context->raycast_cmds.size());
    }

    {
      VulkanSubPass<SubPassType::Compute> raycast_pass;
      raycast_pass.AddDependency<DeviceResourceType::Buffer>(render_context->raycast_cmds_buffer);
      raycast_pass.AddDependency<DeviceResourceType::RWBuffer>(render_context->raycast_results_buffer);

      cmd.BindSubPass(raycast_pass);

      cmd.BindPipeline(render_context->raycast_pipeline);
      cmd.BindDescriptors({render_context->voxel_tree.tree_descriptor, render_context->raycast_descriptor});
      cmd.Dispatch(Vec3u32(render_context->raycast_cmds.size(), 1, 1));
    }

    {
      VulkanSubPass<SubPassType::Transfer> copy_pass;
      copy_pass.AddDependency<DeviceResourceType::TransferSrc>(render_context->raycast_results_buffer);
      copy_pass.AddDependency<DeviceResourceType::TransferDst>(render_context->raycast_staging_buffer);

      cmd.BindSubPass(copy_pass);

      cmd.UploadBufferToBuffer(render_context->raycast_results_buffer, render_context->raycast_staging_buffer,
                               sizeof(RaycastResult) * render_context->raycast_cmds.size());
    }
  });

  for (u32 i = 0; i < render_context->raycast_callbacks.size(); i++) {
    render_context->raycast_callbacks[i](*((RaycastResult *)render_context->raycast_staging_buffer.address));
  }

  render_context->raycast_cmds.clear();
  render_context->raycast_callbacks.clear();
}
}; // namespace Core
