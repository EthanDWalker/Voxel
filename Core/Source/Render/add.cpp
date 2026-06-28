#include "Core/Render/add.h"
#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/other_buffer.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/context.h"
#include "Core/Render/sparse_voxel_tree.h"
#include "Core/Render/types.h"
#include <memory>

namespace Core {
void VoxelizeChunk(const Vec3u32 index, const i32 seed, const u32 max_depth) {
  ZoneScoped;

  SparseVoxelTree::TreeHeader *const header =
      (SparseVoxelTree::TreeHeader *const)render_context->voxel_tree.tree_header_host_buffer.host_address;

  ChunkAllocationInfo alloc_info{};
  alloc_info.chunk_index = index;
  alloc_info.seed = seed;
  alloc_info.max_depth = max_depth;

  for (u32 depth = 1; depth < max_depth; depth++) {
    render_context->voxel_tree.ResizeBranch(header->branch_count);
    header->allocated_branch_count = render_context->voxel_tree.branch_pages.size()
                                     << SparseVoxelTree::PAGE_SIZE_EXP;

    VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
      {
        cmd.BeginDebugPass("svo mesh allocate transfer pass");
        VulkanSubPass<SubPassType::Transfer> transfer_pass;
        transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(
            render_context->voxel_tree.tree_header_host_buffer);
        transfer_pass.AddDependency<DeviceResourceType::TransferDst>(
            render_context->voxel_tree.tree_header_buffer);

        cmd.BindSubPass(transfer_pass);

        cmd.UploadBufferToBuffer(render_context->voxel_tree.tree_header_host_buffer,
                                 render_context->voxel_tree.tree_header_buffer,
                                 render_context->voxel_tree.tree_header_buffer.size);
        cmd.EndDebugPass();
      }

      {
        cmd.BeginDebugPass("svo allocate pass");
        VulkanSubPass<SubPassType::Compute> allocate_pass;
        allocate_pass.AddDependency<DeviceResourceType::RWBuffer>(
            render_context->voxel_tree.tree_header_buffer);

        allocate_pass.ReserveBufferDependencies(render_context->voxel_tree.branch_pages.size());
        for (u32 i = 0; i < render_context->voxel_tree.branch_pages.size(); i++) {
          allocate_pass.AddDependency<DeviceResourceType::RWBuffer>(
              *render_context->voxel_tree.branch_pages[i]);
        }

        cmd.BindSubPass(allocate_pass);

        cmd.BindPipeline(render_context->chunk_allocate_pipeline);
        cmd.BindDescriptors({
            render_context->voxel_tree.descriptor,
        });

        alloc_info.lod = depth;
        alloc_info.alloc_leaf = (depth >= (max_depth - 1));

        cmd.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(ChunkAllocationInfo), &alloc_info);
        cmd.Dispatch(SparseVoxelTree::VOXEL_GRID_SIZE >>
                     (((SparseVoxelTree::MAX_DEPTH - max_depth) << 1) + 4));

        cmd.EndDebugPass();
      }

      {
        cmd.BeginDebugPass("svo allocate transfer to host");
        VulkanSubPass<SubPassType::Transfer> transfer_pass;
        transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(
            render_context->voxel_tree.tree_header_buffer);
        transfer_pass.AddDependency<DeviceResourceType::TransferDst>(
            render_context->voxel_tree.tree_header_host_buffer);

        cmd.BindSubPass(transfer_pass);

        cmd.UploadBufferToBuffer(render_context->voxel_tree.tree_header_buffer,
                                 render_context->voxel_tree.tree_header_host_buffer,
                                 render_context->voxel_tree.tree_header_host_buffer.size);
        cmd.EndDebugPass();
      }
    });
  }

  render_context->voxel_tree.ResizeLeaf(header->leaf_count);
  header->allocated_leaf_count = render_context->voxel_tree.leaf_pages.size()
                                 << SparseVoxelTree::PAGE_SIZE_EXP;

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    {
      cmd.BeginDebugPass("svo allocate child mask transfer");
      VulkanSubPass<SubPassType::Transfer> transfer_pass;
      transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(
          render_context->voxel_tree.tree_header_host_buffer);
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(
          render_context->voxel_tree.tree_header_buffer);

      cmd.BindSubPass(transfer_pass);

      cmd.UploadBufferToBuffer(render_context->voxel_tree.tree_header_host_buffer,
                               render_context->voxel_tree.tree_header_buffer,
                               render_context->voxel_tree.tree_header_buffer.size);
      cmd.EndDebugPass();
    }

    {
      cmd.BeginDebugPass("svo allocate child mask");
      VulkanSubPass<SubPassType::Compute> child_mask_pass;
      child_mask_pass.AddDependency<DeviceResourceType::RWBuffer>(
          render_context->voxel_tree.tree_header_buffer);

      child_mask_pass.ReserveBufferDependencies(render_context->voxel_tree.branch_pages.size());
      for (u32 i = 0; i < render_context->voxel_tree.branch_pages.size(); i++) {
        child_mask_pass.AddDependency<DeviceResourceType::RWBuffer>(
            *render_context->voxel_tree.branch_pages[i]);
      }

      child_mask_pass.ReserveBufferDependencies(render_context->voxel_tree.leaf_pages.size());
      for (u32 i = 0; i < render_context->voxel_tree.leaf_pages.size(); i++) {
        child_mask_pass.AddDependency<DeviceResourceType::RWBuffer>(
            *render_context->voxel_tree.leaf_pages[i]);
      }

      cmd.BindSubPass(child_mask_pass);

      cmd.BindPipeline(render_context->chunk_allocate_pipeline);
      cmd.BindDescriptors({
          render_context->voxel_tree.descriptor,
      });

      alloc_info.color = true;

      cmd.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(ChunkAllocationInfo), &alloc_info);
      cmd.Dispatch(SparseVoxelTree::VOXEL_GRID_SIZE >> (((SparseVoxelTree::MAX_DEPTH - max_depth) << 1) + 4));

      cmd.EndDebugPass();
    }
  });
}
}; // namespace Core
