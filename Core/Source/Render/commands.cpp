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

Vec3u32 ExpandQuadAndClear(const Vec3u32 min, const u32 main_axis, const u32 seconadary_axis, u16 *const data,
                           const Vec3u32 extent) {
  u32 max_main_axis = extent[main_axis];
  for (u32 j = min[main_axis]; j < extent[main_axis]; j++) {
    if (data[j + min[seconadary_axis] * extent[main_axis]] == 0) {
      max_main_axis = j;
      break;
    }
  }

  Vec3u32 max = 0;
  max[main_axis] = max_main_axis;
  max[seconadary_axis] = min[seconadary_axis];

  for (u32 i = min[seconadary_axis]; i < extent[seconadary_axis]; i++) {
    for (u32 j = min[main_axis]; j < max_main_axis; j++) {
      if (data[j + i * extent[main_axis]] == 0) {
        return max;
      } else {
        data[j + i * extent[main_axis]] = 0;
      }
    }
    max[seconadary_axis]++;
  }

  return max;
}

void GreedyMeshSide(const u32 main_axis, const u32 secondary_axis, u16 *const data, const Vec3u32 extent,
                    std::vector<std::pair<Vec3u32, Vec3u32>> &quad_arr) {
  for (u32 i = 0; i < extent[secondary_axis]; i++) {
    for (u32 j = 0; j < extent[main_axis]; j++) {
      if (data[j + i * extent[main_axis]] == 1) {
        Vec3u32 min = 0;
        min[main_axis] = j;
        min[secondary_axis] = i;
        const Vec3u32 max = ExpandQuadAndClear(min, main_axis, secondary_axis, data, extent);
        Core::Log("{} - {}", min.String(), max.String());
        quad_arr.emplace_back(min, max);
      } else {
        data[j + i * extent[main_axis]] = 0;
      }
    }
  }
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

  SCOPED_TIMER("Fill Volume Cmds");

  const Vec3u32 extent = local_cmds[0].max_tree_index - local_cmds[0].min_tree_index;

  VulkanBuffer<BufferType::StagingBuffer> side_filled_buffer = "side filled buffer";
  side_filled_buffer.Create(sizeof(u16) * 2 *
                                (extent.x * extent.y + extent.x * extent.z + extent.z * extent.y),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  memset(side_filled_buffer.host_address, 0, side_filled_buffer.size);

  render_context->emissive_descriptor.Update<DeviceResourceType::Buffer>(1, &side_filled_buffer);

  SparseVoxelTree::TreeHeader *const header =
      (SparseVoxelTree::TreeHeader *)render_context->voxel_tree.tree_header_host_buffer.host_address;

  for (u32 depth = 1; depth < SparseVoxelTree::MAX_DEPTH; depth++) {
    VulkanContext::Submit([local_cmds, depth](VulkanCommandBuffer &cmd) {
      {
        cmd.BeginDebugPass("svo header transfer");
        VulkanSubPass<SubPassType::Transfer> pass;
        pass.AddDependency<DeviceResourceType::TransferSrc>(
            render_context->voxel_tree.tree_header_host_buffer);
        pass.AddDependency<DeviceResourceType::TransferDst>(render_context->voxel_tree.tree_header_buffer);

        cmd.BindSubPass(pass);

        cmd.UploadBufferToBuffer(render_context->voxel_tree.tree_header_host_buffer,
                                 render_context->voxel_tree.tree_header_buffer,
                                 render_context->voxel_tree.tree_header_host_buffer.size);

        cmd.EndDebugPass();
      }

      {
        cmd.BeginDebugPass("fill volume cmds");
        VulkanSubPass<SubPassType::Compute> pass;
        pass.AddDependency<DeviceResourceType::RWBuffer>(render_context->voxel_tree.tree_header_buffer);
        pass.AddDependency<DeviceResourceType::RWBuffer>(render_context->emissive_quad_buffer);

        cmd.BindSubPass(pass);

        cmd.BindPipeline(render_context->fill_volume_pipeline);
        cmd.BindDescriptors({
            render_context->voxel_tree.descriptor,
            render_context->emissive_descriptor,
        });

        for (u32 i = 0; i < local_cmds.size(); i++) {
          CmdVoxelVolumeFillPushConstants push_constants;
          push_constants.volume = local_cmds[i];
          push_constants.depth = depth;
          cmd.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(CmdVoxelVolumeFillPushConstants),
                            &push_constants);

          cmd.Dispatch(((local_cmds[i].max_tree_index - local_cmds[i].min_tree_index) >> 2) + 1);
          cmd.ClearPushConstants();
        }
        cmd.EndDebugPass();
      }

      {
        cmd.BeginDebugPass("svo header transfer");
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

    render_context->voxel_tree.ResizeBranch(header->branch_count);
    header->allocated_branch_count = render_context->voxel_tree.branch_pages.size()
                                     << SparseVoxelTree::PAGE_SIZE_EXP;

    render_context->voxel_tree.ResizeLeaf(header->leaf_count);
    header->allocated_leaf_count = render_context->voxel_tree.leaf_pages.size()
                                   << SparseVoxelTree::PAGE_SIZE_EXP;
  }

  u16 *const side_fill_arr = (u16 *const)side_filled_buffer.host_address;

  const VoxelVolume volume = local_cmds[0];

  u32 offset = 0;
  std::vector<AxisAlignedQuad> quad_arr;
  {
    std::vector<std::pair<Vec3u32, Vec3u32>> side_arr;
    GreedyMeshSide(1, 2, side_fill_arr, extent, side_arr);
    for (const auto &side : side_arr) {
      AxisAlignedQuad quad;
      quad.point_1 = VecTypeCast<f32>(side.first + volume.min_tree_index) * SparseVoxelTree::VOXEL_SIZE +
                     SparseVoxelTree::MIN_BOUND;
      quad.point_2 = VecTypeCast<f32>(side.second + volume.min_tree_index) * SparseVoxelTree::VOXEL_SIZE +
                     SparseVoxelTree::MIN_BOUND;
      quad.normal = 0;
      quad.material_index = volume.material_index;

      quad_arr.emplace_back(quad);
    }
  }
  offset += extent.y * extent.z;

  {
    std::vector<std::pair<Vec3u32, Vec3u32>> side_arr;
    GreedyMeshSide(1, 2, side_fill_arr + offset, extent, side_arr);
    for (const auto &side : side_arr) {
      AxisAlignedQuad quad;
      quad.point_1 = VecTypeCast<f32>(side.first + volume.min_tree_index + Vec3u32(extent.x, 0, 0)) *
                         SparseVoxelTree::VOXEL_SIZE +
                     SparseVoxelTree::MIN_BOUND;
      quad.point_2 = VecTypeCast<f32>(side.second + volume.min_tree_index + Vec3u32(extent.x, 0, 0)) *
                         SparseVoxelTree::VOXEL_SIZE +
                     SparseVoxelTree::MIN_BOUND;
      quad.normal = 1;
      quad.material_index = volume.material_index;

      quad_arr.emplace_back(quad);
    }
  }
  offset += extent.y * extent.z;

  {
    std::vector<std::pair<Vec3u32, Vec3u32>> side_arr;
    GreedyMeshSide(0, 2, side_fill_arr + offset, extent, side_arr);
    for (const auto &side : side_arr) {
      AxisAlignedQuad quad;
      quad.point_1 = VecTypeCast<f32>(side.first + volume.min_tree_index) * SparseVoxelTree::VOXEL_SIZE +
                     SparseVoxelTree::MIN_BOUND;
      quad.point_2 = VecTypeCast<f32>(side.second + volume.min_tree_index) * SparseVoxelTree::VOXEL_SIZE +
                     SparseVoxelTree::MIN_BOUND;
      quad.normal = 2;
      quad.material_index = volume.material_index;

      quad_arr.emplace_back(quad);
    }
  }
  offset += extent.x * extent.z;

  {
    std::vector<std::pair<Vec3u32, Vec3u32>> side_arr;
    GreedyMeshSide(0, 2, side_fill_arr + offset, extent, side_arr);
    for (const auto &side : side_arr) {
      AxisAlignedQuad quad;
      quad.point_1 = VecTypeCast<f32>(side.first + volume.min_tree_index + Vec3u32(0, extent.y, 0)) *
                         SparseVoxelTree::VOXEL_SIZE +
                     SparseVoxelTree::MIN_BOUND;
      quad.point_2 = VecTypeCast<f32>(side.second + volume.min_tree_index + Vec3u32(0, extent.y, 0)) *
                         SparseVoxelTree::VOXEL_SIZE +
                     SparseVoxelTree::MIN_BOUND;
      quad.normal = 3;
      quad.material_index = volume.material_index;

      quad_arr.emplace_back(quad);
    }
  }
  offset += extent.x * extent.z;

  {
    std::vector<std::pair<Vec3u32, Vec3u32>> side_arr;
    GreedyMeshSide(0, 1, side_fill_arr + offset, extent, side_arr);
    for (const auto &side : side_arr) {
      AxisAlignedQuad quad;
      quad.point_1 = VecTypeCast<f32>(side.first + volume.min_tree_index) * SparseVoxelTree::VOXEL_SIZE +
                     SparseVoxelTree::MIN_BOUND;
      quad.point_2 = VecTypeCast<f32>(side.second + volume.min_tree_index) * SparseVoxelTree::VOXEL_SIZE +
                     SparseVoxelTree::MIN_BOUND;
      quad.normal = 4;
      quad.material_index = volume.material_index;

      quad_arr.emplace_back(quad);
    }
  }
  offset += extent.x * extent.y;

  {
    std::vector<std::pair<Vec3u32, Vec3u32>> side_arr;
    GreedyMeshSide(0, 1, side_fill_arr + offset, extent, side_arr);
    for (const auto &side : side_arr) {
      AxisAlignedQuad quad;
      quad.point_1 = VecTypeCast<f32>(side.first + volume.min_tree_index + Vec3u32(0, 0, extent.z)) *
                         SparseVoxelTree::VOXEL_SIZE +
                     SparseVoxelTree::MIN_BOUND;
      quad.point_2 = VecTypeCast<f32>(side.second + volume.min_tree_index + Vec3u32(0, 0, extent.z)) *
                         SparseVoxelTree::VOXEL_SIZE +
                     SparseVoxelTree::MIN_BOUND;
      quad.material_index = 0;
      quad.normal = 5;
      quad.material_index = volume.material_index;

      quad_arr.emplace_back(quad);
    }
  }
  offset += extent.x * extent.y;

  if (quad_arr.size() == 0) {
    return;
  }
  VulkanBuffer<BufferType::StagingBuffer> quad_staging_buffer = "quad staging buffer";
  quad_staging_buffer.Create(sizeof(AxisAlignedQuad) * quad_arr.size());
  memcpy(quad_staging_buffer.host_address, quad_arr.data(), quad_staging_buffer.size);
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

        cmd.Dispatch(((local_cmds[i].max_tree_index - local_cmds[i].min_tree_index) >> 2) - 1);
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
