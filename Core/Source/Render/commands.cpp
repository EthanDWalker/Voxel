#include "Core/Render/commands.h"
#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/context.h"
#include "Core/Render/sparse_voxel_tree.h"
#include "Core/Render/types.h"
#include "Core/Util/thread_pool.h"

namespace Core {
void QueueFillVolumeCmd(const VoxelVolume &volume) {
  ZoneScoped;
  std::lock_guard<std::mutex> lock = std::lock_guard(render_context->fill_volume_cmd_mutex);
  render_context->fill_volume_cmds.emplace_back(volume);
}

void FlushFillVolumeCmds() {
  ZoneScoped;
  std::vector<VoxelVolume> local_cmds;

  {
    std::lock_guard<std::mutex> lock = std::lock_guard(render_context->fill_volume_cmd_mutex);
    if (render_context->fill_volume_cmds.size() == 0)
      return;
    local_cmds = render_context->fill_volume_cmds;
    render_context->fill_volume_cmds.clear();
  }

  for (u32 i = 0; i < SparseVoxelTree::MAX_DEPTH; i++) {
    VulkanContext::Submit([local_cmds](VulkanCommandBuffer &cmd) {
      {
        cmd.BeginDebugPass("fill volume cmds");

        cmd.BindPipeline(render_context->fill_volume_pipeline);
        cmd.BindDescriptors({render_context->voxel_tree.descriptor});

        for (u32 i = 0; i < local_cmds.size(); i++) {
          cmd.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(VoxelVolume), &local_cmds[i]);

          cmd.Dispatch((local_cmds[i].max_tree_index - local_cmds[i].min_tree_index));
          cmd.ClearPushConstants();
        }
        cmd.EndDebugPass();
      }

      {
        cmd.BeginDebugPass("fill volume cmds");
        VulkanSubPass<SubPassType::Transfer> pass;
        pass.AddDependency<DeviceResourceType::TransferSrc>(render_context->voxel_tree.tree_header_buffer);
        pass.AddDependency<DeviceResourceType::TransferDst>(
            render_context->voxel_tree.tree_header_host_buffer);

        cmd.BindSubPass(pass);

        cmd.UploadBufferToBuffer(render_context->voxel_tree.tree_header_buffer,
                                 render_context->voxel_tree.tree_header_host_buffer,
                                 render_context->voxel_tree.tree_header_buffer.size);

        cmd.EndDebugPass();
      }
    });

    const SparseVoxelTree::TreeHeader *const header =
        (SparseVoxelTree::TreeHeader *)render_context->voxel_tree.tree_header_host_buffer.host_address;

    render_context->voxel_tree.ResizeBranch(header->branch_count);
    render_context->voxel_tree.ResizeLeaf(header->leaf_count);
  }
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

  ThreadPool::QueueTask([local_cmds]() {
    VulkanContext::Submit([local_cmds](VulkanCommandBuffer &cmd) {
      cmd.BeginDebugPass("clear volume cmds");

      cmd.BindPipeline(render_context->clear_volume_pipeline);
      cmd.BindDescriptors({render_context->voxel_tree.descriptor});

      for (u32 i = 0; i < local_cmds.size(); i++) {
        cmd.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(VoxelVolume), &local_cmds[i]);

        cmd.Dispatch((local_cmds[i].max_tree_index - local_cmds[i].min_tree_index));
        cmd.ClearPushConstants();
      }
      cmd.EndDebugPass();
    });
  });
}

void QueueRaycastCmd(const Raycast &raycast,
                     const std::function<void(const RaycastResult &result)> &&callback) {
  ZoneScoped;
  std::lock_guard<std::mutex> lock = std::lock_guard(render_context->raycast_cmd_mutex);
  render_context->raycast_cmds.emplace_back(raycast);
  render_context->raycast_cmd_callbacks.emplace_back(callback);
}

void FlushRaycastCmds() {
  ZoneScoped;
  if (render_context->raycast_cmds.size() == 0)
    return;

  std::lock_guard<std::mutex> lock = std::lock_guard(render_context->raycast_cmd_mutex);
  VulkanContext::Submit([](VulkanCommandBuffer &cmd) {
    {
      cmd.BeginDebugPass("raycast cmd copy from host");
      VulkanSubPass<SubPassType::Transfer> copy_pass;
      copy_pass.AddDependency<DeviceResourceType::TransferSrc>(render_context->raycast_staging_buffer);
      copy_pass.AddDependency<DeviceResourceType::TransferDst>(render_context->raycast_cmds_buffer);

      cmd.BindSubPass(copy_pass);

      memcpy(render_context->raycast_staging_buffer.host_address, render_context->raycast_cmds.data(),
             sizeof(Raycast) * render_context->raycast_cmds.size());
      cmd.UploadBufferToBuffer(render_context->raycast_staging_buffer, render_context->raycast_cmds_buffer,
                               sizeof(Raycast) * render_context->raycast_cmds.size());
      cmd.EndDebugPass();
    }

    {
      cmd.BeginDebugPass("raycast cmd pass");
      VulkanSubPass<SubPassType::Compute> raycast_pass;
      raycast_pass.AddDependency<DeviceResourceType::Buffer>(render_context->raycast_cmds_buffer);
      raycast_pass.AddDependency<DeviceResourceType::RWBuffer>(render_context->raycast_results_buffer);

      cmd.BindSubPass(raycast_pass);

      cmd.BindPipeline(render_context->raycast_cmd_pipeline);
      cmd.BindDescriptors({render_context->voxel_tree.descriptor, render_context->raycast_descriptor});
      cmd.Dispatch(Vec3u32(render_context->raycast_cmds.size(), 1, 1));
      cmd.EndDebugPass();
    }

    {
      cmd.BeginDebugPass("raycast cmd copy to host");
      VulkanSubPass<SubPassType::Transfer> copy_pass;
      copy_pass.AddDependency<DeviceResourceType::TransferSrc>(render_context->raycast_results_buffer);
      copy_pass.AddDependency<DeviceResourceType::TransferDst>(render_context->raycast_staging_buffer);

      cmd.BindSubPass(copy_pass);

      cmd.UploadBufferToBuffer(render_context->raycast_results_buffer, render_context->raycast_staging_buffer,
                               sizeof(RaycastResult) * render_context->raycast_cmds.size());
      cmd.EndDebugPass();
    }
  });

  for (u32 i = 0; i < render_context->raycast_cmd_callbacks.size(); i++) {
    render_context->raycast_cmd_callbacks[i](
        *((RaycastResult *)render_context->raycast_staging_buffer.host_address));
  }

  render_context->raycast_cmds.clear();
  render_context->raycast_cmd_callbacks.clear();
}
} // namespace Core
