#include "Core/Render/sparse_voxel_tree.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/Vulkan/sampler.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/types.h"
#include "Core/Util/Parse/gltf.h"
#include <memory>

namespace Core {
const u32 VERTEX_BUFFER_BINDING = 0;
const u32 ALBEDO_IMAGE_BINDING = 1;

const u32 TREE_BUFFER_BINDING = 0;
const u32 TREE_LEAF_BUFFER_BINDING = 1;

SparseVoxelTree::SparseVoxelTree() {
  ZoneScoped;
  constexpr bool host = true;

  sampler.Create(SamplerFilter::Linear, SamplerFilter::Linear);

  tree_header_buffer.Create(sizeof(TreeHeader), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  tree_header_host_buffer.Create(sizeof(TreeHeader),
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, host);

  empty_page_host_buffer.Create(sizeof(BranchNode) * PAGE_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host);

  for (u32 i = 0; i < PAGE_SIZE; i++) {
    ((BranchNode *)empty_page_host_buffer.address)[i] = {
        .child_mask = 0, .child_ptr = SENTINAL, .radiance = 0};
  }

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>();          // vertex buffer
  DescriptorBuilder::Bind<DeviceResourceType::SampledImage>();    // albedo image
  DescriptorBuilder::Bind<DeviceResourceType::Sampler>(&sampler); // sampler
  DescriptorBuilder::Build(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_ALL_GRAPHICS, &voxelize_descriptor);

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, MAX_PAGES); // voxel tree
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, MAX_PAGES); // voxel tree leafs
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&tree_header_buffer);
  DescriptorBuilder::Build(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_ALL_GRAPHICS, &tree_descriptor);

  TreeHeader header{};
  header._min_bound = MIN_BOUND;
  header._max_bound = MAX_BOUND;
  header._max_voxelize_depth = MAX_VOXLELIZE_DEPTH;
  header._page_size_exp = PAGE_SIZE_EXP;
  header.level_voxel_count[0] = 64;

  for (u32 i = 0; i < MAX_VOXLELIZE_DEPTH - 1; i++) {
    const u32 max_level_nodes = (1 << ((i + 1) * 6));
    if (max_level_nodes <= PAGE_SIZE) {
      pages[i]
          .emplace_back(std::make_unique<VulkanBuffer>())
          ->Create(sizeof(BranchNode) * max_level_nodes,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);
    } else {
      pages[i]
          .emplace_back(std::make_unique<VulkanBuffer>())
          ->Create(sizeof(BranchNode) * PAGE_SIZE,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);
    }
    tree_descriptor.Update<DeviceResourceType::Buffer>(TREE_BUFFER_BINDING, pages[i].back().get(), i);

    header.level_page_offset[i] = i;
  }

  leaf_pages.emplace_back(std::make_unique<VulkanBuffer>())
      ->Create(sizeof(LeafNode) * PAGE_SIZE,
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);
  tree_descriptor.Update<DeviceResourceType::Buffer>(TREE_LEAF_BUFFER_BINDING, leaf_pages.back().get(), 0);

  memcpy(tree_header_host_buffer.address, &header, sizeof(TreeHeader));

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    VulkanSubPass<SubPassType::Transfer> transfer_pass;
    transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(tree_header_host_buffer);
    transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(empty_page_host_buffer);
    transfer_pass.AddDependency<DeviceResourceType::TransferDst>(tree_header_buffer);
    for (u32 i = 0; i < pages.size(); i++) {
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(*pages[i][0]);
    }
    for (u32 i = 0; i < leaf_pages.size(); i++) {
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(*leaf_pages[i]);
    }

    cmd.BindSubPass(transfer_pass);

    cmd.UploadBufferToBuffer(tree_header_host_buffer, tree_header_buffer, sizeof(TreeHeader));
    for (u32 i = 0; i < pages.size(); i++) {
      cmd.UploadBufferToBuffer(empty_page_host_buffer, *pages[i][0], pages[i][0]->size);
    }
    for (u32 i = 0; i < leaf_pages.size(); i++) {
      cmd.FillBuffer(*leaf_pages[i], leaf_pages[i]->size, 0);
    }
  });

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Graphic>();
    pipeline_builder.Default();
    pipeline_builder.SetNoDepthTest();
    pipeline_builder.SetCullMode(VK_CULL_MODE_NONE, {});
    pipeline_builder.AddDescriptor(voxelize_descriptor);
    pipeline_builder.AddDescriptor(tree_descriptor);
    pipeline_builder.AddPushConstantRange(VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(u32));
    pipeline_builder.EnableConservativeRasterization();
    pipeline_builder.SetShaders(std::filesystem::path(SHADER_DIR) / "allocate.slang",
                                std::filesystem::path(SHADER_DIR) / "allocate.slang",
                                std::filesystem::path(SHADER_DIR) / "allocate.slang");
    PipelineBuildManager::Build(pipeline_builder, allocate_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Graphic>();
    pipeline_builder.Default();
    pipeline_builder.SetNoDepthTest();
    pipeline_builder.SetCullMode(VK_CULL_MODE_NONE, {});
    pipeline_builder.AddDescriptor(voxelize_descriptor);
    pipeline_builder.AddDescriptor(tree_descriptor);
    pipeline_builder.EnableConservativeRasterization();
    pipeline_builder.SetShaders(std::filesystem::path(SHADER_DIR) / "allocate_child_mask.slang",
                                std::filesystem::path(SHADER_DIR) / "allocate_child_mask.slang",
                                std::filesystem::path(SHADER_DIR) / "allocate_child_mask.slang");
    PipelineBuildManager::Build(pipeline_builder, allocate_child_mask_pipeline);
  }
}

void SparseVoxelTree::VoxelizeMesh(const MeshData &mesh_data) {
  ZoneScoped;
  TreeHeader *header = (TreeHeader *)tree_header_host_buffer.address;

  VulkanBuffer vertex_buffer;
  vertex_buffer.Create(sizeof(Vertex) * mesh_data.vertex_arr.size(),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  VulkanBuffer index_buffer;
  index_buffer.Create(sizeof(Index) * mesh_data.index_arr.size(),
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  VulkanImage<ImageType::Planar> albedo_image;
  albedo_image.Create(mesh_data.material.albedo_extent, VK_FORMAT_R8G8B8A8_UNORM,
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  voxelize_descriptor.Update<DeviceResourceType::Buffer>(VERTEX_BUFFER_BINDING, &vertex_buffer);
  voxelize_descriptor.Update<DeviceResourceType::SampledImage>(ALBEDO_IMAGE_BINDING, &albedo_image);

  const u64 image_data_size =
      mesh_data.material.albedo_extent.width * mesh_data.material.albedo_extent.height * 4;

  constexpr bool host = true;
  VulkanBuffer staging_buffer;
  staging_buffer.Create(vertex_buffer.size + index_buffer.size + image_data_size,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host);

  {
    for (u32 depth = 1; depth < MAX_VOXLELIZE_DEPTH; depth++) {
      TreeHeader *header = ((TreeHeader *)tree_header_host_buffer.address);

      const u32 page_offset = pages[depth - 1].size();
      const u32 new_page_offset = header->level_voxel_count[depth - 1] >> PAGE_SIZE_EXP;

      if (new_page_offset >= page_offset) {
        for (u32 i = page_offset; i <= new_page_offset; i++) {
          pages[depth - 1]
              .emplace_back(std::make_unique<VulkanBuffer>())
              ->Create(sizeof(BranchNode) * PAGE_SIZE,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);

          tree_descriptor.Update<DeviceResourceType::Buffer>(
              TREE_BUFFER_BINDING, pages[depth - 1].back().get(), header->level_page_offset[depth - 1] + i);
        }

        u32 index = header->level_page_offset[depth - 1] + pages[depth - 1].size();
        for (u32 i = depth; i < MAX_VOXLELIZE_DEPTH - 1; i++) {
          header->level_page_offset[i] = index;
          for (u32 j = 0; j < pages[i].size(); j++) {
            tree_descriptor.Update<DeviceResourceType::Buffer>(TREE_BUFFER_BINDING, pages[i][j].get(), index);
            index++;
          }
        }
      }

      VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
        {
          VulkanSubPass<SubPassType::Transfer> transfer_pass;
          transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(staging_buffer);
          transfer_pass.AddDependency<DeviceResourceType::TransferDst>(vertex_buffer);
          transfer_pass.AddDependency<DeviceResourceType::TransferDst>(index_buffer);
          transfer_pass.AddDependency<DeviceResourceType::TransferDst>(albedo_image);
          transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(empty_page_host_buffer);

          transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(tree_header_host_buffer);
          transfer_pass.AddDependency<DeviceResourceType::TransferDst>(tree_header_buffer);

          for (u32 i = page_offset; i <= new_page_offset; i++) {
            transfer_pass.AddDependency<DeviceResourceType::TransferDst>(*pages[depth - 1][i]);
          }

          cmd.BindSubPass(transfer_pass);

          u64 offset = 0;

          if (depth == 1) {
            memcpy((char *)staging_buffer.address + offset, mesh_data.vertex_arr.data(), vertex_buffer.size);
            cmd.UploadBufferToBuffer(staging_buffer, vertex_buffer, vertex_buffer.size, offset);
            offset += vertex_buffer.size;

            memcpy((char *)staging_buffer.address + offset, mesh_data.index_arr.data(), index_buffer.size);
            cmd.UploadBufferToBuffer(staging_buffer, index_buffer, index_buffer.size, offset);
            offset += index_buffer.size;

            memcpy((char *)staging_buffer.address + offset, mesh_data.material.albedo_data, image_data_size);
            cmd.UploadBufferToImage(staging_buffer, albedo_image, offset);
            offset += image_data_size;
          }

          for (u32 i = page_offset; i <= new_page_offset; i++) {
            cmd.UploadBufferToBuffer(empty_page_host_buffer, *pages[depth - 1][i], pages[depth - 1][i]->size);
          }

          cmd.UploadBufferToBuffer(tree_header_host_buffer, tree_header_buffer, tree_header_buffer.size);
        }

        {
          VulkanSubPass<SubPassType::Graphic> allocate_pass;
          allocate_pass.AddDependency<DeviceResourceType::Buffer>(vertex_buffer);
          allocate_pass.AddDependency<DeviceResourceType::RWBuffer>(tree_header_buffer);
          for (u32 i = 0; i < pages.size(); i++) {
            for (u32 j = 0; j < pages[i].size(); j++) {
              allocate_pass.AddDependency<DeviceResourceType::RWBuffer>(*pages[i][j]);
            }
          }

          cmd.BindSubPass(allocate_pass);

          cmd.BeginRendering({}, nullptr, Vec2u32(1 << (MAX_VOXLELIZE_DEPTH * 2)));
          cmd.BindPipeline(allocate_pipeline);
          cmd.BindDescriptors({voxelize_descriptor, tree_descriptor});
          cmd.PushConstants(VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(u32), &depth);
          cmd.BindIndexBuffer(index_buffer);
          cmd.DrawIndexed(mesh_data.index_arr.size());
          cmd.EndRendering();
        }

        {
          VulkanSubPass<SubPassType::Transfer> transfer_pass;
          transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(tree_header_buffer);
          transfer_pass.AddDependency<DeviceResourceType::TransferDst>(tree_header_host_buffer);

          cmd.BindSubPass(transfer_pass);

          cmd.UploadBufferToBuffer(tree_header_buffer, tree_header_host_buffer, tree_header_host_buffer.size);
        }
      });
    }
  }

  const u32 page_offset = leaf_pages.size();
  const u32 new_page_offset = header->leaf_voxel_count >> PAGE_SIZE_EXP;

  for (u32 i = page_offset; i <= new_page_offset; i++) {
    leaf_pages.emplace_back(std::make_unique<VulkanBuffer>())
        ->Create(sizeof(LeafNode) * PAGE_SIZE,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);

    tree_descriptor.Update<DeviceResourceType::Buffer>(TREE_LEAF_BUFFER_BINDING, leaf_pages.back().get(), i);
  }

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    {
      VulkanSubPass<SubPassType::Transfer> transfer_pass;
      for (u32 i = page_offset; i <= new_page_offset; i++) {
        transfer_pass.AddDependency<DeviceResourceType::TransferDst>(*leaf_pages[i]);
      }

      cmd.BindSubPass(transfer_pass);

      for (u32 i = page_offset; i <= new_page_offset; i++) {
        cmd.FillBuffer(*leaf_pages[i], leaf_pages[i]->size, 0);
      }
    }
    {
      VulkanSubPass<SubPassType::Graphic> child_mask_pass;
      child_mask_pass.AddDependency<DeviceResourceType::Buffer>(vertex_buffer);
      child_mask_pass.AddDependency<DeviceResourceType::RWBuffer>(tree_header_buffer);
      for (u32 i = 0; i < pages.size(); i++) {
        for (u32 j = 0; j < pages[i].size(); j++) {
          child_mask_pass.AddDependency<DeviceResourceType::RWBuffer>(*pages[i][j]);
        }
      }
      for (u32 i = 0; i < leaf_pages.size(); i++) {
        child_mask_pass.AddDependency<DeviceResourceType::RWBuffer>(*leaf_pages[i]);
      }

      cmd.BindSubPass(child_mask_pass);

      const u32 depth = MAX_VOXLELIZE_DEPTH;

      cmd.BeginRendering({}, nullptr, Vec2u32(1 << (depth * 2)));
      cmd.BindPipeline(allocate_child_mask_pipeline);
      cmd.BindDescriptors({voxelize_descriptor, tree_descriptor});
      cmd.BindIndexBuffer(index_buffer);
      cmd.DrawIndexed(mesh_data.index_arr.size());
      cmd.EndRendering();
    }
  });
}
} // namespace Core
