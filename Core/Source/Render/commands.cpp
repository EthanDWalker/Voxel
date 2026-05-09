#include "Core/Render/commands.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/Vulkan/util.h"
#include "Core/Render/context.h"
#include "Core/Render/types.h"
#include "Core/Util/thread_pool.h"

namespace Core {
void QueueAddInstanceCmd(const Mesh &mesh) {
  std::lock_guard<std::mutex> lock(render_context->add_instance_cmd_mutex);
  Instance &instance = render_context->add_instance_cmds.emplace_back();
  instance.transform = Mat4ToVkTransform(1.0f);
  instance.instanceCustomIndex = mesh.id;
  instance.mask = 0xFF;
  instance.instanceShaderBindingTableRecordOffset = 0;
  instance.flags = 0;
  instance.accelerationStructureReference =
      GetDeviceAddress(render_context->bottom_level_acceleration_structures[mesh.id]->obj);
}

void FlushAddInstanceCmds() {
  ZoneScoped;
  if (render_context->add_instance_cmds.size() == 0)
    return;

  VulkanBuffer staging_buffer = "staging buffer";
  staging_buffer.Create(sizeof(Instance) * render_context->add_instance_cmds.size(),
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, /*host=*/true);
  memcpy(staging_buffer.host_address, render_context->add_instance_cmds.data(), staging_buffer.size);

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    {
      VulkanSubPass<SubPassType::Transfer> transfer_pass;
      transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(staging_buffer);
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(render_context->instance_buffer);

      cmd.BindSubPass(transfer_pass);

      cmd.UploadBufferToBuffer(staging_buffer, render_context->instance_buffer, staging_buffer.size, 0,
                               render_context->instance_count * sizeof(Instance));
    }
  });

  render_context->instance_count += render_context->add_instance_cmds.size();
  render_context->add_instance_cmds.clear();

  render_context->top_level_acceleration_structure.RecreateTopLevel(render_context->instance_buffer,
                                                                    render_context->instance_count);
  render_context->top_level_acceleration_structure_descriptor
      .Update<DeviceResourceType::AccelerationStructure>(0,
                                                         &render_context->top_level_acceleration_structure);
}

void QueueClearVolumeCmd(const VoxelVolume &volume) {
  ZoneScoped;
  std::lock_guard<std::mutex> lock = std::lock_guard(render_context->clear_volume_cmd_mutex);
  render_context->clear_volume_cmds.emplace_back(volume);
}

void FlushClearVolumeCmds() {
  ZoneScoped;
  std::vector<VoxelVolume> local_cmds;

  {
    std::lock_guard<std::mutex> lock = std::lock_guard(render_context->clear_volume_cmd_mutex);
    if (render_context->clear_volume_cmds.size() == 0)
      return;
    local_cmds = render_context->clear_volume_cmds;
    render_context->clear_volume_cmds.clear();
  }

  ThreadPool::QueueTask([local_cmds](u32 id) {
    SCOPED_TIMER("flush clear cmds");
    VulkanContext::Submit([local_cmds](VulkanCommandBuffer &cmd) {
      cmd.BindPipeline(render_context->clear_volume_pipeline);
      cmd.BindDescriptors({render_context->voxel_tree.descriptor});

      for (u32 i = 0; i < local_cmds.size(); i++) {
        cmd.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(VoxelVolume), &local_cmds[i]);

        const Vec3u32 min_tree_index = GetTreeIndex(local_cmds[i].min);
        const Vec3u32 max_tree_index = GetTreeIndex(local_cmds[i].max);
        cmd.Dispatch((max_tree_index - min_tree_index));
        cmd.ClearPushConstants();
      }
    });
  });
}

void QueueRaycastCmd(const Raycast &raycast,
                     const std::function<void(const RaycastResult &result)> &&callback) {
  ZoneScoped;
  std::lock_guard<std::mutex> lock = std::lock_guard(render_context->raycast_mutex);
  render_context->raycast_cmds.emplace_back(raycast);
  render_context->raycast_callbacks.emplace_back(callback);
}

void FlushRaycastCmds() {
  ZoneScoped;
  if (render_context->raycast_cmds.size() == 0)
    return;

  std::lock_guard<std::mutex> lock = std::lock_guard(render_context->raycast_mutex);
  VulkanContext::Submit([](VulkanCommandBuffer &cmd) {
    {
      VulkanSubPass<SubPassType::Transfer> copy_pass;
      copy_pass.AddDependency<DeviceResourceType::TransferSrc>(render_context->raycast_staging_buffer);
      copy_pass.AddDependency<DeviceResourceType::TransferDst>(render_context->raycast_cmds_buffer);

      cmd.BindSubPass(copy_pass);

      memcpy(render_context->raycast_staging_buffer.host_address, render_context->raycast_cmds.data(),
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
      cmd.BindDescriptors({render_context->voxel_tree.descriptor, render_context->raycast_descriptor});
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
    render_context->raycast_callbacks[i](
        *((RaycastResult *)render_context->raycast_staging_buffer.host_address));
  }

  render_context->raycast_cmds.clear();
  render_context->raycast_callbacks.clear();
}
} // namespace Core
